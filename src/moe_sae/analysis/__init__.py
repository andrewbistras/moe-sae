"""Statistical analyses over FeatureDump artifacts (spec §14–§16).

Pure numpy/scipy/sklearn over θ*-inference dumps + the router-statistics
artifact; no model, no network. Sub-modules:

- `mass`       — spec §14: per-token per-bank masses, q_M(e), example tables
- `enrichment` — spec §15: R_M(e) vs gate/route baselines, §16.3 load floor
- `stability`  — spec §16.1–§16.3: JSD, nulls, bootstrap CIs, concentration,
                 top-6 stability (Jaccard / Spearman / selection frequency)
- `probes`     — spec §16.4: incremental modality-information probes
"""

from moe_sae.analysis.enrichment import (
    EPSILON,
    RouterStats,
    enrichment,
    enrichment_analysis,
    load_floor_mask,
    load_stats,
)
from moe_sae.analysis.mass import (
    BANKED_CONDITIONS,
    GLOBAL_CONDITIONS,
    BankMasses,
    ExampleMasses,
    FeatureDump,
    domain_masks,
    load_dump,
    per_example_masses,
    q_by_domain,
    q_by_modality,
    q_from_mass,
    token_bank_masses,
)
from moe_sae.analysis.probes import example_split, run_probes
from moe_sae.analysis.stability import (
    bootstrap_analysis,
    entropy_stats,
    jaccard,
    jsd,
    shuffled_label_jsds,
    stability_analysis,
    top_enriched,
)

__all__ = [
    "BANKED_CONDITIONS",
    "GLOBAL_CONDITIONS",
    "EPSILON",
    "BankMasses",
    "ExampleMasses",
    "FeatureDump",
    "RouterStats",
    "bootstrap_analysis",
    "domain_masks",
    "enrichment",
    "enrichment_analysis",
    "entropy_stats",
    "example_split",
    "jaccard",
    "jsd",
    "load_dump",
    "load_floor_mask",
    "load_stats",
    "per_example_masses",
    "q_by_domain",
    "q_by_modality",
    "q_from_mass",
    "run_probes",
    "shuffled_label_jsds",
    "stability_analysis",
    "token_bank_masses",
    "top_enriched",
]
