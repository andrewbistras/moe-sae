"""Run manifest (determinism, hash stability) and logbook (format, 20-process
concurrent append integrity). Ops tooling only — no training math touched."""

import json
import multiprocessing

import yaml

from moe_sae.train.config import PIPELINE_F, PIPELINE_K_BAR, RunConfig
from moe_sae.util.manifest import (
    MANIFEST_NAME,
    config_content_hash,
    load_run_manifest,
    logbook_append,
    logbook_entries,
    write_run_manifest,
)


def make_cfg(**overrides) -> RunConfig:
    kw = dict(
        run_name="pipe_A_s1",
        condition="A",
        seed=1,
        f=PIPELINE_F,
        k_bar=PIPELINE_K_BAR,
        total_tokens=1_000_000,
        capture_root="captures/pipeline",
        stats_path="stats/pipeline/stats.safetensors",
        pipeline=True,
    )
    kw.update(overrides)
    return RunConfig(**kw)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def test_manifest_write_and_read_back(tmp_path):
    cfg = make_cfg()
    path = write_run_manifest(tmp_path / "run", cfg, extra={"orchestrator": "test"})
    assert path.name == MANIFEST_NAME
    m = load_run_manifest(tmp_path / "run")

    assert m["config"] == json.loads(cfg.to_json())  # resolved config echoed
    assert m["config_hash"].startswith("sha256:")
    assert m["extra"] == {"orchestrator": "test"}
    # spec version strings parsed from spec/ filenames (repo has v0.1.1 + v0.1.2)
    assert "v0.1.1" in m["spec_versions"] and "v0.1.2" in m["spec_versions"]
    # env fingerprint + accelerator inventory + git presence
    assert m["env"]["python"] and m["env"]["hostname"] and m["env"]["platform"]
    assert m["env"]["torch"] is not None  # torch is a repo dependency
    assert m["accelerators"]["cpu_count"] >= 1
    assert set(m["git"]) == {"commit", "dirty", "branch"}
    assert m["created_at"][:2] == "20"  # ISO timestamp
    # frozen-constants echo matches the config values (spec §5–§11)
    fc = m["frozen_constants"]
    assert fc["batch_tokens"] == cfg.batch_tokens
    assert fc["theta_ema_decay"] == cfg.theta_ema_decay
    assert fc["aux_token_budget"] == cfg.aux_token_budget
    assert fc["f"] == cfg.f and fc["k_bar"] == cfg.k_bar
    assert fc["condition"] == "A"


def test_manifest_deterministic_given_fixed_inputs(tmp_path):
    cfg = make_cfg()
    write_run_manifest(tmp_path / "run", cfg, extra={"n": 1})
    m1 = load_run_manifest(tmp_path / "run")
    write_run_manifest(tmp_path / "run", cfg, extra={"n": 1})
    m2 = load_run_manifest(tmp_path / "run")
    m1.pop("created_at"), m2.pop("created_at")
    assert m1 == m2  # only the timestamp may differ


def test_config_hash_stability_and_sensitivity(tmp_path):
    cfg = make_cfg()
    h = config_content_hash(cfg)
    # stable across calls and across YAML round-trip (resolved content hash)
    assert config_content_hash(cfg) == h
    p = tmp_path / "cfg.yaml"
    cfg.to_yaml(p)
    assert config_content_hash(RunConfig.from_yaml(p)) == h
    # key order in the source mapping must not matter (canonical sort)
    d = json.loads(cfg.to_json())
    reordered = dict(reversed(list(d.items())))
    assert config_content_hash(reordered) == h
    # any content change must change the hash
    assert config_content_hash(make_cfg(seed=2)) != h
    assert config_content_hash(make_cfg(run_name="other")) != h


def test_manifest_missing_spec_dir_is_empty_list(tmp_path):
    m_path = write_run_manifest(
        tmp_path / "run", make_cfg(), spec_dir=tmp_path / "nonexistent"
    )
    assert json.loads(m_path.read_text())["spec_versions"] == []


def test_manifest_config_yaml_hash_matches_manifest_field(tmp_path):
    """The manifest's hash is of the resolved YAML content, per contract."""
    import hashlib

    cfg = make_cfg()
    write_run_manifest(tmp_path / "run", cfg)
    m = load_run_manifest(tmp_path / "run")
    canonical = yaml.safe_dump(json.loads(cfg.to_json()), sort_keys=True)
    assert m["config_hash"] == "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Logbook
# ---------------------------------------------------------------------------


def test_logbook_append_format_and_parse(tmp_path):
    path = tmp_path / "LOGBOOK.md"
    logbook_append("phase 3 start", {"note": "hello", "step": 12}, path=path)
    logbook_append("gate PASS", {"nmse": 0.25}, path=path)
    logbook_append("bare event", None, path=path)

    text = path.read_text()
    assert text.count("## ") == 3
    entries = logbook_entries(path)
    assert [e["event"] for e in entries] == ["phase 3 start", "gate PASS", "bare event"]
    assert entries[0]["detail"] == {"note": "hello", "step": "12"}
    assert entries[1]["detail"] == {"nmse": "0.25"}
    assert entries[2]["detail"] == {}
    for e in entries:
        assert e["ts"] and e["ts"][:2] == "20"  # timestamped header


def test_logbook_multiline_value_stays_one_line(tmp_path):
    path = tmp_path / "LOGBOOK.md"
    logbook_append("evt", {"tb": "a\nb", "cfg": {"x": 1}}, path=path)
    (entry,) = logbook_entries(path)
    assert entry["detail"]["tb"] == '"a\\nb"'  # newline escaped, entry intact
    assert entry["detail"]["cfg"] == '{"x":1}'


def _worker(args):
    """Top-level so multiprocessing (spawn) can pickle it."""
    i, path = args
    from moe_sae.util.manifest import logbook_append

    logbook_append(f"evt-{i:02d}", {"idx": i, "payload": "x" * 256}, path=path)


def test_logbook_concurrent_append_20_processes(tmp_path):
    """20 processes append simultaneously; all 20 entries must be intact
    (single O_APPEND write per entry — no interleaving)."""
    path = str(tmp_path / "LOGBOOK.md")
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=10) as pool:
        pool.map(_worker, [(i, path) for i in range(20)])

    entries = logbook_entries(path)
    assert len(entries) == 20
    assert sorted(e["event"] for e in entries) == [f"evt-{i:02d}" for i in range(20)]
    for e in entries:
        i = int(e["event"].split("-")[1])
        assert e["detail"]["idx"] == str(i)
        assert e["detail"]["payload"] == "x" * 256  # full entry, no tearing
        assert e["ts"] is not None
