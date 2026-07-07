"""Domain streams: field mapping on dict fixtures, frozen example-id offsets,
round-robin interleaving (spec §9). No network — records are injected; one
optional smoke test hits the Hub only when MOE_SAE_NET=1."""

import itertools
import os

import pytest

from moe_sae.data.streams import (
    COCO_CAPTION_INSTRUCTION,
    DOMAIN_OFFSETS,
    SPLIT_OFFSETS,
    domain_stream,
    interleave_domains,
    map_ai2d_record,
    map_chartqa_record,
    map_coco_record,
    map_textvqa_record,
    text_control_stream,
)

IMG = object()  # mapping functions pass the image field through untouched


# --------------------------------------------------------------------------- #
# Field mapping fixtures (shapes taken from real streamed records)             #
# --------------------------------------------------------------------------- #


def test_map_coco():
    rec = {"image": IMG, "filename": "x.jpg", "cocoid": 42, "caption": " A cat. "}
    image, prompt, answer = map_coco_record(rec)
    assert image is IMG
    assert prompt == COCO_CAPTION_INSTRUCTION
    assert answer == "A cat."


def test_map_textvqa_majority_answer():
    rec = {
        "image": IMG,
        "question": " what brand? ",
        "answers": ["nokia", "toshiba", "nokia", " nokia "],
    }
    image, prompt, answer = map_textvqa_record(rec)
    assert image is IMG
    assert prompt == "what brand?"
    assert answer == "nokia"  # modal answer, whitespace-normalized


def test_map_textvqa_no_answers():
    rec = {"image": IMG, "question": "q", "answers": []}
    assert map_textvqa_record(rec)[2] == ""


def test_map_chartqa():
    rec = {"image": IMG, "query": "Is it 38?", "label": ["Yes"], "human_or_machine": 0}
    image, prompt, answer = map_chartqa_record(rec)
    assert (image, prompt, answer) == (IMG, "Is it 38?", "Yes")


def test_map_ai2d():
    rec = {
        "image": IMG,
        "question": "which is dairy",
        "options": ["milk", "rock", "tree", "car"],
        "answer": "1",
    }
    image, prompt, answer = map_ai2d_record(rec)
    assert image is IMG
    assert answer == "rock"  # options[int(answer)]
    assert "(A) milk" in prompt and "(B) rock" in prompt and "(D) car" in prompt
    assert prompt.startswith("which is dairy")


# --------------------------------------------------------------------------- #
# Example ids                                                                  #
# --------------------------------------------------------------------------- #


def fake_coco_records(n):
    return [{"image": IMG, "caption": f"cap {i}"} for i in range(n)]


def test_domain_stream_ids_and_offsets():
    exs = list(domain_stream("coco", "train", records=fake_coco_records(3)))
    assert [e.example_id for e in exs] == [0, 1, 2]
    assert all(e.domain == "coco" for e in exs)
    assert [e.answer_text for e in exs] == ["cap 0", "cap 1", "cap 2"]

    exs_val = list(domain_stream("coco", "validation", records=fake_coco_records(2)))
    assert [e.example_id for e in exs_val] == [50_000_000, 50_000_001]

    recs = [
        {"image": IMG, "question": f"q{i}", "answers": ["a"]} for i in range(2)
    ]
    exs_tv = list(domain_stream("textvqa", "train", records=recs))
    assert [e.example_id for e in exs_tv] == [100_000_000, 100_000_001]


def test_ai2d_pseudo_splits_disjoint_and_offset():
    recs = [
        {"image": IMG, "question": f"q{i}", "options": ["x", "y"], "answer": "0"}
        for i in range(20)
    ]
    train = list(domain_stream("ai2d", "train", records=recs))
    val = list(domain_stream("ai2d", "validation", records=recs))
    test = list(domain_stream("ai2d", "test", records=recs))
    # 0-7 of every 10 → train; 8 → validation; 9 → test
    assert [e.prompt_text.split("\n")[0] for e in train] == [
        f"q{i}" for i in range(20) if i % 10 < 8
    ]
    assert [e.prompt_text.split("\n")[0] for e in val] == ["q8", "q18"]
    assert [e.prompt_text.split("\n")[0] for e in test] == ["q9", "q19"]
    # underlying records disjoint; ids in disjoint frozen ranges
    assert train[0].example_id == DOMAIN_OFFSETS["ai2d"]
    assert val[0].example_id == DOMAIN_OFFSETS["ai2d"] + SPLIT_OFFSETS["validation"]
    assert test[0].example_id == DOMAIN_OFFSETS["ai2d"] + SPLIT_OFFSETS["test"]


# --------------------------------------------------------------------------- #
# Interleaving                                                                 #
# --------------------------------------------------------------------------- #


def _fake_streams(counts):
    streams = {}
    for dom, n in counts.items():
        if dom == "coco":
            recs = fake_coco_records(n)
        else:
            recs = [
                {"image": IMG, "question": f"{dom}{i}", "answers": ["a"]}
                for i in range(n)
            ]
        streams[dom] = domain_stream(dom, "train", records=recs)
    return streams


def test_interleave_round_robin_even():
    streams = _fake_streams({"coco": 5, "textvqa": 5})
    out = list(interleave_domains(streams, seed=0))
    assert len(out) == 10
    # even at the example level: every consecutive pair covers both domains
    for i in range(0, 10, 2):
        assert {out[i].domain, out[i + 1].domain} == {"coco", "textvqa"}


def test_interleave_deterministic_and_seed_dependent():
    a = [e.example_id for e in interleave_domains(_fake_streams({"coco": 4, "textvqa": 4}), 3)]
    b = [e.example_id for e in interleave_domains(_fake_streams({"coco": 4, "textvqa": 4}), 3)]
    assert a == b
    c = [e.example_id for e in interleave_domains(_fake_streams({"coco": 4, "textvqa": 4}), 4)]
    assert sorted(a) == sorted(c)  # same examples, possibly different order


def test_interleave_drops_exhausted_stream():
    streams = _fake_streams({"coco": 2, "textvqa": 6})
    out = list(interleave_domains(streams, seed=1))
    assert len(out) == 8
    assert [e.domain for e in out].count("coco") == 2
    # after coco is exhausted the remaining examples are all textvqa
    tail = [e.domain for e in out[-4:]]
    assert tail == ["textvqa"] * 4


def test_text_control_stream_ids_and_no_images():
    streams = _fake_streams({"coco": 3, "textvqa": 3})
    out = list(text_control_stream(split="train", seed=0, streams=streams))
    assert len(out) == 6
    assert all(e.image is None for e in out)
    assert all(e.domain == "text_control" for e in out)
    assert [e.example_id for e in out] == [400_000_000 + i for i in range(6)]
    assert all(e.prompt_text for e in out)


# --------------------------------------------------------------------------- #
# Optional network smoke test                                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(os.environ.get("MOE_SAE_NET") != "1", reason="set MOE_SAE_NET=1")
def test_streams_smoke_network():
    for dom in ("coco", "textvqa", "chartqa", "ai2d"):
        ex = next(iter(domain_stream(dom, "train")))
        assert ex.example_id == DOMAIN_OFFSETS[dom]
        assert ex.image is not None and ex.prompt_text
    mixed = list(
        itertools.islice(
            interleave_domains({d: domain_stream(d, "train") for d in ("coco", "textvqa")}, 7),
            4,
        )
    )
    assert {e.domain for e in mixed} == {"coco", "textvqa"}
