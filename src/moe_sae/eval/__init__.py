"""Evaluation package: reconstruction metrics + FeatureDumps (spec §13),
downstream fidelity (spec §13, changelog #1), the A-vs-C implementation sanity
gate (spec §20 step 5), and the §17 subtraction intervention."""

from moe_sae.eval.fidelity import replacement_mask, run_fidelity
from moe_sae.eval.intervention import (
    matched_control_sets,
    run_intervention,
    select_enriched_latents,
)
from moe_sae.eval.reconstruction import (
    ThresholdedEncoding,
    ThresholdedSAE,
    dump_features,
    evaluate_reconstruction,
    split_mask,
)
from moe_sae.eval.sanity_gate import compare_sanity

__all__ = [
    "ThresholdedEncoding",
    "ThresholdedSAE",
    "compare_sanity",
    "dump_features",
    "evaluate_reconstruction",
    "matched_control_sets",
    "replacement_mask",
    "run_fidelity",
    "run_intervention",
    "select_enriched_latents",
    "split_mask",
]
