"""Experiment metrics, summaries and convergence plots."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any


@dataclass(frozen=True)
class EpisodeMetrics:
    """Metrics captured for one training or evaluation episode."""

    episode: int
    reward: float
    steps: int
    success: bool
    epsilon: float
    duration_seconds: float
    optimal_steps: int | None = None


def summarize(records: Sequence[EpisodeMetrics]) -> dict[str, Any]:
    """Build a JSON-serializable aggregate summary."""
    if not records:
        raise ValueError("At least one episode record is required")
    successful = [record for record in records if record.success]
    optimal = [
        record.steps / record.optimal_steps
        for record in successful
        if record.optimal_steps and record.optimal_steps > 0
    ]
    return {
        "episodes": len(records),
        "successes": len(successful),
        "success_rate": len(successful) / len(records),
        "average_reward": fmean(record.reward for record in records),
        "average_steps": fmean(record.steps for record in records),
        "average_success_steps": (
            fmean(record.steps for record in successful) if successful else None
        ),
        "path_efficiency_ratio": fmean(optimal) if optimal else None,
        "total_duration_seconds": sum(record.duration_seconds for record in records),
    }


def write_metrics(
    records: Sequence[EpisodeMetrics],
    output_dir: str | Path,
    *,
    prefix: str,
    metadata: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Write episode-level CSV and aggregate JSON files."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / f"{prefix}_episodes.csv"
    json_path = directory / f"{prefix}_summary.json"

    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)

    payload = {"summary": summarize(records), "metadata": metadata or {}}
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return csv_path, json_path


def moving_average(values: Sequence[float], window: int = 20) -> list[float]:
    """Return a trailing moving average with a shorter initial window."""
    if window <= 0:
        raise ValueError("window must be positive")
    return [fmean(values[max(0, index - window + 1) : index + 1]) for index in range(len(values))]


def plot_convergence(
    records: Sequence[EpisodeMetrics],
    destination: str | Path,
    *,
    window: int = 20,
) -> Path:
    """Save reward and success convergence curves as a PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    episodes = [record.episode for record in records]
    rewards = [record.reward for record in records]
    success = [float(record.success) for record in records]

    figure, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(episodes, rewards, alpha=0.3, label="Episode reward")
    axes[0].plot(episodes, moving_average(rewards, window), label=f"Moving average ({window})")
    axes[0].set_ylabel("Reward")
    axes[0].legend()
    axes[0].grid(alpha=0.2)
    axes[1].plot(episodes, moving_average(success, window), color="seagreen")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Success rate")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(alpha=0.2)
    figure.suptitle("Q-learning convergence")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path
