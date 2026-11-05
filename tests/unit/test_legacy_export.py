# mypy: disable-error-code="no-any-return,unused-ignore"
# ruff: noqa: E501
"""T:comparativo-legado: exportação no formato de arquivos de uma reprodução anterior."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from physics_llm_sr.analysis import legacy_export as lx
from physics_llm_sr.analysis.loading import AnalysisError

#: Pasta da reprodução anterior (formato de referência); sem a variável, os testes que a usam são pulados.
REF = Path(os.environ.get("PHYSICS_LLM_SR_LEGACY_REF", "/nonexistent-legacy-ref"))
RESULT_COLS = (
    "key,scenario,sr,llm,prompt,noise_mode,noise_level,biased,rep,MAE,MSE,one_minus_R2,R2,"
    "distance,size,plaus,loss,equation,raw_equation,n_generations,n_evaluated,sr_seconds,"
    "llm_seconds,llm_ratio,llm_calls,llm_cached,llm_fail_rate,timestamp"
).split(",")
TABLE_COLS = {
    "tabela2_prompts.csv": "prompt,llm,sr,MAE,MAE_artigo,MSE,MSE_artigo,one_minus_R2,one_minus_R2_artigo,distance,distance_artigo",
    "tabela5_benchmark.csv": "scenario,llm,sr,MAE,MAE_artigo,MSE,MSE_artigo,one_minus_R2,one_minus_R2_artigo,distance,distance_artigo",
    "tabela6_tempos.csv": "scenario,sr,llm,total_sr_s,total_llm_s,llm_sr_ratio,sr_per_call_s,llm_per_call_s,sr_per_iter_s,llm_per_iter_s,total_sr_s_artigo,total_llm_s_artigo,llm_sr_ratio_artigo,sr_per_call_s_artigo,llm_per_call_s_artigo,sr_per_iter_s_artigo,llm_per_iter_s_artigo",
    "overhead_llm.csv": "sr,llm,razao_LLM_SR_mediana,chamadas_LLM_media,falha_parse_media,execucoes",
}
KEYS = ("dimensional_consistency", "simplicity", "physical_realism")


def _header(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as fh:
        return next(csv.reader(fh))


def _rows() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    spec = [
        ("deap", "none", "none", "baseline,llm_integration", "zero_mean", "target", "x*2.5"),
        (
            "deap",
            "llama2",
            "E",
            "llm_integration,prompt_sensitivity",
            "zero_mean",
            "target",
            "sin(t)",
        ),
        (
            "pysr",
            "mistral",
            "A",
            "prompt_sensitivity",
            "zero_mean",
            "target",
            "__import__('os').system('echo hi')",
        ),
        ("deap", "none", "none", "noise_robustness", "zero_mean", "inputs", "x"),
        ("deap", "mistral", "E", "biased_noise", "biased", "both", "x + 1.0"),
    ]
    for i, (sr, llm, prompt, exps, nt, nl, expr) in enumerate(spec):
        rows.append(
            {
                "run_id": "rid", "cell_key": f"c{i}", "scenario": "free_fall", "sr": sr, "llm": llm,
                "prompt": prompt, "noise_type": nt, "noise_location": nl, "noise_level": 0.01,
                "repetition": i, "mode": "pilot", "track": "paper_replication",
                "normalization": "literal", "tree_distance": 0.5, "mae": 2.0, "mse": 5.0,
                "one_minus_r2": 0.4, "test_mae": 1.5, "test_mse": 4.0, "test_one_minus_r2": 0.3,
                "test_r2": 0.7, "expression": expr, "complexity": 3, "n_generations": 4,
                "final_loss": 1.0, "final_e": 1.0, "final_s": 1.0,
                "final_c": 0.0 if llm == "none" else 0.25, "train_seconds": 12.0 if llm != "none" else 2.0,
                "llm_seconds": 10.0 if llm != "none" else 0.0, "llm_calls": 5 if llm != "none" else 0,
                "llm_cache_hits": 1, "llm_invalid_responses": 1 if llm != "none" else 0,
                "llm_in_loop": llm != "none", "status": "ok", "experiments": exps,
            }
        )  # fmt: skip
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def export(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("leg")
    run = root / "results" / "pilot" / "rid"
    (run / "cells" / "c1").mkdir(parents=True)
    _rows().to_parquet(run / "results.parquet")
    (run / "cells" / "c1" / "manifest.json").write_text(
        json.dumps({"created_at": "2026-11-05T03:25:03.819390+00:00"})
    )
    cfg = {
        "llm": {
            "cache_dir": ".cache/llm",
            "models": [{"name": "mistral", "checkpoint": "mistral:7b"}],
        },
        "loss": {"w1": 1, "w2": 1, "w3": 1, "normalization": "literal"},
        "experiments": {},
    }
    (run / "run.json").write_text(
        json.dumps(
            {
                "run_id": "rid",
                "created_at": "2026-11-05T00:00:00+00:00",
                "git_sha": "abc123",
                "config": cfg,
            }
        )
    )
    cache = root / ".cache" / "llm"
    cache.mkdir(parents=True)
    ev = {
        "dimensional_consistency": 1.0,
        "simplicity": 0.5,
        "physical_realism": 0.0,
        "feedback": "f",
    }
    meta = {
        "expression": "t + 1.0",
        "model_id": "mistral:7b",
        "context_hash": "ab" * 32,
        "scenario": "free_fall",
    }
    recs = [
        {"key": "k1", "status": "ok", "evaluation": ev, "raw_response": "RAW", "metadata": meta},
        {"key": "k2", "status": "ok", "evaluation": ev, "raw_response": "", "metadata": meta},
        {"key": "k3", "status": "error", "error": "e", "raw_responses": ["bad"], "metadata": meta},
    ]
    (cache / "evaluations.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    lx.export_legacy([run], root / "out")
    return root / "out"


def test_result_columns_exact(export: Path) -> None:
    for name in (lx.BENCHMARK_FILE, lx.PROMPTS_FILE, lx.NOISE_FILE):
        assert _header(export / name) == RESULT_COLS
        if (REF / name).exists():
            assert _header(export / name) == _header(REF / name)


def test_table_columns_and_dir(export: Path) -> None:
    tdir = export / "tabelas_pilot"
    for name, cols in TABLE_COLS.items():
        assert _header(tdir / name) == cols.split(",")
        if (REF / "tabelas_pilot" / name).exists():
            assert _header(tdir / name) == _header(REF / "tabelas_pilot" / name)
    for png in ("fig_tabela2_prompts.png", "fig_tabela3_ruido.png", "fig_tabela4_ruido.png"):
        assert (tdir / png).stat().st_size > 1000


def test_key_format_and_mapping(export: Path) -> None:
    df = pd.read_csv(export / lx.BENCHMARK_FILE, keep_default_na=False)
    assert set(df["key"]) == {
        "ball_drop|deap|none|-|target|0.01|0|0",
        "ball_drop|deap|llama|E|target|0.01|0|1",
    }
    base = df[df["llm"] == "none"].iloc[0]
    assert (
        base["prompt"] == "-" and base["noise_mode"] == "target" and str(base["biased"]) == "False"
    )
    assert float(base["MAE"]) == 1.5 and float(base["R2"]) == 0.7  # métricas de TESTE
    noise = pd.read_csv(export / lx.NOISE_FILE, keep_default_na=False)
    assert "ball_drop|deap|mistral|E|both|0.01|1|4" in set(noise["key"])
    assert set(noise["noise_mode"]) == {"features", "both"}


def test_unavailable_fields_are_nan(export: Path) -> None:
    df = pd.read_csv(export / lx.BENCHMARK_FILE, keep_default_na=False)
    assert set(df["n_evaluated"]) == {"nan"}
    base = df[df["llm"] == "none"].iloc[0]
    assert base["plaus"] == "nan" and base["llm_fail_rate"] == 0.0
    llm = df[df["llm"] == "llama"].iloc[0]
    assert (
        float(llm["plaus"]) == 0.75
        and float(llm["sr_seconds"]) == 2.0
        and float(llm["llm_ratio"]) == 5.0
    )
    assert float(llm["llm_fail_rate"]) == 0.2 and llm["llm_cached"] == 1
    assert llm["timestamp"] == "2026-11-05 03:25:03"
    assert base["timestamp"] == "2026-11-05 00:00:00"  # sem manifest da célula: run.json


def test_raw_equation_no_eval(export: Path, tmp_path: Path) -> None:
    assert lx.raw_equation("x*2.5") == "mul(x, 2.5)"
    prompts = pd.read_csv(export / lx.PROMPTS_FILE, keep_default_na=False)
    evil = prompts[prompts["llm"] == "mistral"].iloc[0]
    assert evil["raw_equation"] == "nan" and "__import__" in evil["equation"]
    assert lx.raw_equation("__import__('os').system('touch /tmp/pwned_legacy')") == "nan"
    assert not Path("/tmp/pwned_legacy").exists()


def test_wide_noise_triple_header(export: Path) -> None:
    lines = (export / "tabelas_pilot" / "tabela4_ruido_enviesado.csv").read_text().splitlines()
    assert lines[0].startswith("noise_mode,,features,features,features,features,features,target")
    assert lines[1].startswith("noise_level,,0.01,0.02,0.03,0.04,0.05,0.01")
    assert lines[2].startswith("scenario,sr,,,")
    both = lines[3].split(",")  # ball_drop,deap
    assert both[:2] == ["ball_drop", "deap"] and both[12] == "0.5"
    assert len(lines) == 3 + 9
    base = (
        (export / "tabelas_pilot" / "tabela3_baseline_sem_llm.csv")
        .read_text()
        .splitlines()[3]
        .split(",")
    )
    assert base[2] == "0.5"  # features 0.01, baseline sem LLM


def test_cache_commit_and_md(export: Path) -> None:
    lines = [json.loads(x) for x in (export / lx.CACHE_OUT).read_text().splitlines()]
    assert lines[0] == {
        "key": f"mistral|{'ab' * 6}|velocity = t + 1.0",
        "score": 0.5,
        "ok": True,
        "raw": "RAW",
    }
    assert json.loads(lines[1]["raw"]) == {k: v for k, v in zip(KEYS, (1.0, 0.5, 0.0), strict=True)}
    assert lines[2]["ok"] is False and lines[2]["score"] is None and lines[2]["raw"] == "bad"
    assert (export / lx.COMMIT_FILE).read_text() == "abc123\n"
    md = (export / lx.MD_FILE).read_text()
    assert "execução reduzida, não é reprodução" in md and "n_evaluated" in md
    for col in RESULT_COLS:
        assert f"`{col}`" in md and col in (lx.__doc__ or "")
        assert col in dict(lx.MAPPING)


def test_mixed_arms_refused(tmp_path: Path) -> None:
    run1 = tmp_path / "a"
    run2 = tmp_path / "b"
    for run, track in ((run1, "paper_replication"), (run2, "extended_analysis")):
        run.mkdir()
        _rows().assign(track=track).to_parquet(run / "results.parquet")
    with pytest.raises(AnalysisError, match="uma saída"):
        lx.export_legacy([run1, run2], tmp_path / "o")
