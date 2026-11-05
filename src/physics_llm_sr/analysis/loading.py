"""Carga de N execuções (qualquer modo/trilha) num DataFrame único com a coluna ``arm``.

Lê SOMENTE arquivos persistidos (``results.parquet``, ``history.parquet``, ``run.json``,
``report.json``). Nada é recalculado aqui e nenhum número é inventado.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

RESULTS_FILE = "results.parquet"
HISTORY_FILE = "history.parquet"
RUN_FILE = "run.json"
REPORT_FILE = "report.json"
STATUS_OK = "ok"
BASELINE_LLM = "none"
REQUIRED_COLUMNS = (
    "scenario",
    "sr",
    "llm",
    "prompt",
    "noise_type",
    "noise_location",
    "noise_level",
    "repetition",
    "mode",
    "track",
    "status",
    "mae",
    "mse",
    "one_minus_r2",
    "tree_distance",
    "expression",
    "final_e",
    "final_s",
    "final_c",
    "train_seconds",
    "llm_seconds",
    "llm_calls",
)
"""Colunas mínimas de ``results.parquet`` necessárias às tabelas e figuras."""
HISTORY_REQUIRED = ("cell_key", "sr", "llm", "repetition", "generation", "best_loss")
MODE_NAMES = {"smoke": "SMOKE", "pilot": "PILOTO", "full": "FULL"}
EXTENDED_TRACK = "extended_analysis"
REPLICATION_TRACK = "paper_replication"
FULL_MODE = "full"


class AnalysisError(ValueError):
    """Execução inválida ou inconsistente para a análise."""


@dataclass(frozen=True)
class RunInfo:
    """Metadados de uma execução (tudo vindo dos arquivos persistidos)."""

    run_dir: Path
    arm: str
    arm_short: str
    mode: str
    track: str
    normalization: str
    label: str
    weights: tuple[float, float, float] | None
    experiments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AnalysisData:
    """Resultados e históricos de todas as execuções, com ``arm`` em cada linha."""

    results: pd.DataFrame
    history: pd.DataFrame
    runs: list[RunInfo]
    cache: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @property
    def ok(self) -> pd.DataFrame:
        """Células com ``status == 'ok'``."""
        out: pd.DataFrame = self.results[self.results["status"] == STATUS_OK].copy()
        return out

    def run_of(self, arm: str) -> RunInfo:
        """Metadados do braço ``arm``."""
        for run in self.runs:
            if run.arm == arm:
                return run
        raise KeyError(arm)

    def label_of(self, arm: str) -> str:
        """Rótulo de modo/trilha do braço."""
        return self.run_of(arm).label

    @property
    def arms(self) -> list[str]:
        """Braços na ordem de carga."""
        return [r.arm for r in self.runs]

    def combined_label(self) -> str:
        """Rótulos distintos dos braços, unidos (título de figuras agregadas)."""
        return " || ".join(dict.fromkeys(r.label for r in self.runs))


def make_label(mode: str, track: str, normalization: str, report_label: str | None) -> str:
    """Rótulo de modo/trilha derivado dos dados (e do ``report.json`` se for full incompleto)."""
    name = MODE_NAMES.get(mode, mode.upper())
    if mode != FULL_MODE:
        label = f"{name} — execução reduzida, não é reprodução"
    elif track == REPLICATION_TRACK and report_label:
        label = f"{name} — {report_label}"
    else:
        label = f"{name} — {track}"
    if track == EXTENDED_TRACK:
        label += f" | {EXTENDED_TRACK} — desvio declarado ({normalization})"
    return label


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _weights(config: dict[str, Any]) -> tuple[float, float, float] | None:
    loss = config.get("loss") or {}
    try:
        return (float(loss["w1"]), float(loss["w2"]), float(loss["w3"]))
    except (KeyError, TypeError, ValueError):
        return None


def _load_one(run_dir: Path) -> tuple[RunInfo, pd.DataFrame, pd.DataFrame]:
    run_dir = Path(run_dir)
    res_path = run_dir / RESULTS_FILE
    if not res_path.exists():
        raise AnalysisError(f"{res_path} não existe (rode `physics-llm-sr report` antes)")
    results = pd.read_parquet(res_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in results.columns]
    if missing:
        raise AnalysisError(f"{run_dir.name}: colunas ausentes em results.parquet: {missing}")
    if results.empty:
        raise AnalysisError(f"{run_dir.name}: results.parquet vazio")
    modes, tracks = results["mode"].unique(), results["track"].unique()
    if len(modes) != 1 or len(tracks) != 1:
        raise AnalysisError(f"{run_dir.name}: mode/track não únicos ({modes}, {tracks})")
    mode, track = str(modes[0]), str(tracks[0])
    run_json = _read_json(run_dir / RUN_FILE)
    config = run_json.get("config") or {}
    if "normalization" in results.columns:
        norm = str(results["normalization"].iloc[0])
    else:
        norm = str((config.get("loss") or {}).get("normalization", "desconhecida"))
    report = _read_json(run_dir / REPORT_FILE)
    arm = f"{track}/{mode}/{run_dir.name}"
    info = RunInfo(
        run_dir=run_dir,
        arm=arm,
        arm_short=f"{norm} ({track})",
        mode=mode,
        track=track,
        normalization=norm,
        label=make_label(mode, track, norm, report.get("label")),
        weights=_weights(config),
        experiments=dict(config.get("experiments") or {}),
    )
    results = results.copy()
    results.insert(0, "arm", arm)
    hist_path = run_dir / HISTORY_FILE
    history = pd.read_parquet(hist_path) if hist_path.exists() else pd.DataFrame()
    if not history.empty:
        bad = [c for c in HISTORY_REQUIRED if c not in history.columns]
        if bad:
            raise AnalysisError(f"{run_dir.name}: colunas ausentes em history.parquet: {bad}")
        history = history.copy()
        history.insert(0, "arm", arm)
    return info, results, history


def load_runs(run_dirs: list[Path] | list[str]) -> AnalysisData:
    """Carrega as execuções em um único conjunto, com a coluna ``arm`` (track/mode/nome)."""
    if not run_dirs:
        raise AnalysisError("nenhuma execução informada")
    infos: list[RunInfo] = []
    results: list[pd.DataFrame] = []
    histories: list[pd.DataFrame] = []
    for run_dir in run_dirs:
        info, res, hist = _load_one(Path(run_dir))
        if any(i.arm == info.arm for i in infos):
            raise AnalysisError(f"braço duplicado: {info.arm}")
        infos.append(info)
        results.append(res)
        histories.append(hist)
    shorts = [i.arm_short for i in infos]
    if len(set(shorts)) < len(shorts):  # desambigua: duas execuções da mesma trilha/escala
        infos = [
            RunInfo(**{**i.__dict__, "arm_short": f"{i.arm_short} {i.run_dir.name}"}) for i in infos
        ]
    nonempty = [h for h in histories if not h.empty]
    history = pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()
    return AnalysisData(pd.concat(results, ignore_index=True), history, infos)
