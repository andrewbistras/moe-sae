"""Capture CLI: stream the four frozen domains through DeepSeek-VL2-Tiny and
write the spec §3 capture manifest (or export §3 validation inputs).

Examples:
    uv run python scripts/run_capture.py --out captures/discovery \\
        --examples 20000 --visual-cap 512 --seed 7
    uv run python scripts/run_capture.py --out captures/text_control \\
        --examples 2000 --visual-cap 0 --seed 7 --text-control
    uv run python scripts/run_capture.py --export-validation-inputs \\
        captures/validation_inputs --validation-tokens 100000
"""

from __future__ import annotations

import argparse
import itertools
import logging
import os


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", help="capture root, e.g. captures/<name>")
    p.add_argument("--examples", type=int, help="number of examples to capture")
    p.add_argument("--visual-cap", type=int, help="max visual rows kept per example")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument(
        "--domains",
        default="coco,textvqa,chartqa,ai2d",
        help="comma-separated subset of the four frozen domains",
    )
    p.add_argument("--split", default="train", choices=["train", "validation", "test"])
    p.add_argument(
        "--text-control",
        action="store_true",
        help="capture the image-free control stream (separate capture root; "
        "domain=255; never mixed into training captures)",
    )
    p.add_argument("--device", default=None, help="cuda|mps|cpu (default: auto)")
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--shard-size", type=int, default=16384)
    p.add_argument(
        "--start",
        type=int,
        default=0,
        help="skip this many stream examples before capturing (parallel "
        "workers: worker w uses --start w*PER --examples PER into its own --out)",
    )
    p.add_argument(
        "--buckets",
        default=None,
        help="capture only examples with example_id %% 100 in lo-hi (e.g. "
        "'80-99' for the retained dev/eval capture; saves the skipped "
        "examples' model forwards entirely)",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="continue an interrupted capture at --out from its progress.json "
        "(identical parameters required; a completed capture is a no-op)",
    )
    p.add_argument(
        "--export-validation-inputs",
        metavar="DIR",
        default=None,
        help="export prompt-only validation inputs (no model load) and exit",
    )
    p.add_argument("--validation-tokens", type=int, default=100_000)
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )

    if args.export_validation_inputs is not None:
        from moe_sae.data.evalsets import export_validation_inputs

        export_validation_inputs(
            args.export_validation_inputs,
            n_tokens_target=args.validation_tokens,
            seed=args.seed,
        )
        _hard_exit()

    if args.out is None or args.examples is None or args.visual_cap is None:
        p.error("--out, --examples and --visual-cap are required for capture")

    from moe_sae.capture.driver import capture_examples
    from moe_sae.data.streams import domain_stream, interleave_domains, text_control_stream
    from moe_sae.model_loading import load_model, load_processor

    model, info = load_model(device=args.device)
    processor = load_processor()
    logging.getLogger(__name__).info("model loaded: %s", info)

    if args.text_control:
        examples = text_control_stream(split=args.split, seed=args.seed)
    else:
        domains = [d.strip() for d in args.domains.split(",") if d.strip()]
        examples = interleave_domains(
            {d: domain_stream(d, split=args.split) for d in domains}, seed=args.seed
        )

    if args.buckets:
        lo, hi = (int(x) for x in args.buckets.split("-"))
        base = examples

        def _bucket_filtered(src=base, lo=lo, hi=hi):
            for ex in src:
                if lo <= ex.example_id % 100 <= hi:
                    yield ex

        examples = _bucket_filtered()

    # The raw stream (not islice) goes to the driver: on --resume it
    # fast-forwards by the recorded consumed count, and n_examples_target
    # bounds CONSUMED examples so resumed runs stop at the same stream
    # position a single uninterrupted run would.
    capture_examples(
        model=model,
        processor=processor,
        examples_iter=examples,
        out_root=args.out,
        visual_cap_per_example=args.visual_cap,
        seed=args.seed,
        max_seq_len=args.max_seq_len,
        text_control=args.text_control,
        shard_size=args.shard_size,
        n_examples_target=args.examples,
        resume=args.resume,
        start_offset=args.start,
    )
    _hard_exit()


def _hard_exit() -> None:
    """Exit without running interpreter finalizers. Abandoned HF streaming
    iterators (itertools.islice over parquet-backed IterableDatasets) can leave
    pyarrow IO threads alive, and arrow's ThreadPool shutdown then deadlocks at
    interpreter exit (observed in this environment). All capture artifacts are
    written synchronously before this point (ShardWriter.close / meta.save)."""
    logging.shutdown()
    os._exit(0)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        # A crash must not fall through to normal interpreter exit: abandoned
        # HF streaming iterators leave pyarrow IO threads whose ThreadPool
        # shutdown deadlocks in exit() (observed: process hung in
        # __cxa_finalize after an exception). Print and hard-exit instead.
        import traceback

        traceback.print_exc()
        logging.shutdown()
        os._exit(1)
