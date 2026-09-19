"""Command-line interface for training, evaluation, demos and benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from static_maze_solver.agent import QLearningAgent
from static_maze_solver.environment import MazeEnv
from static_maze_solver.maze import MazeConfig, generate_maze, get_preset
from static_maze_solver.metrics import plot_convergence, summarize, write_metrics
from static_maze_solver.training import evaluate_agent, train_agent
from static_maze_solver.visualization import record_demo, save_snapshot


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return number


def _probability(value: str) -> float:
    number = float(value)
    if not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("value must be between 0 and 1")
    return number


def _add_maze_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--maze", choices=("classic", "small", "generated"), default="classic")
    parser.add_argument("--rows", type=_positive_int, default=10)
    parser.add_argument("--cols", type=_positive_int, default=10)
    parser.add_argument("--obstacle-density", type=_probability, default=0.2)
    parser.add_argument("--max-steps", type=_positive_int)


def _maze_from_args(args: argparse.Namespace, *, seed: int | None = None) -> MazeConfig:
    actual_seed = args.seed if seed is None else seed
    if args.maze == "generated":
        return generate_maze(args.rows, args.cols, args.obstacle_density, actual_seed)
    return get_preset(args.maze)


def _environment(
    config: MazeConfig,
    args: argparse.Namespace,
    *,
    render_mode: str | None = None,
) -> MazeEnv:
    return MazeEnv(
        grid_size=config.grid_size,
        start=config.start,
        goal=config.goal,
        obstacles=config.obstacles,
        max_steps=args.max_steps,
        render_mode=render_mode,
    )


def _metadata(config: MazeConfig, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "maze": asdict(config),
        "seed": args.seed,
        "max_steps": args.max_steps,
    }


def _progress(record: Any) -> None:
    if record.episode == 1 or record.episode % 50 == 0:
        print(
            f"episode={record.episode:4d} reward={record.reward:7.1f} "
            f"steps={record.steps:4d} success={record.success} epsilon={record.epsilon:.3f}"
        )


def command_train(args: argparse.Namespace) -> int:
    """Train and persist one reproducible experiment."""
    output = Path(args.output)
    config = _maze_from_args(args)
    env = _environment(config, args, render_mode="human" if args.render else None)
    agent = QLearningAgent(
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        seed=args.seed,
    )
    try:
        result = train_agent(
            env,
            agent,
            episodes=args.episodes,
            seed=args.seed,
            render=args.render,
            early_stop_window=args.early_stop_window,
            callback=_progress,
        )
    finally:
        env.close()

    model_path = agent.save(output / "q_table.json")
    csv_path, summary_path = write_metrics(
        result.records,
        output,
        prefix="train",
        metadata={**_metadata(config, args), "stopped_early": result.stopped_early},
    )
    plot_path = plot_convergence(result.records, output / "convergence.png")
    print(f"Model: {model_path}")
    print(f"Metrics: {csv_path}, {summary_path}")
    print(f"Plot: {plot_path}")
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    """Evaluate a saved model greedily and write metrics."""
    output = Path(args.output)
    config = _maze_from_args(args)
    env = _environment(config, args, render_mode="human" if args.render else None)
    agent = QLearningAgent.load(args.model)
    try:
        result = evaluate_agent(
            env,
            agent,
            episodes=args.episodes,
            seed=args.seed,
            render=args.render,
            callback=_progress,
        )
    finally:
        env.close()
    _, summary_path = write_metrics(
        result.records,
        output,
        prefix="evaluation",
        metadata=_metadata(config, args),
    )
    summary = summarize(result.records)
    print(f"Success rate: {summary['success_rate']:.1%}")
    print(f"Average steps: {summary['average_steps']:.1f}")
    print(f"Report: {summary_path}")
    return 0 if summary["success_rate"] > 0 else 2


def command_demo(args: argparse.Namespace) -> int:
    """Run a human demonstration or record reusable portfolio assets."""
    config = _maze_from_args(args)
    agent = QLearningAgent.load(args.model)
    if args.record:
        env = _environment(config, args, render_mode="rgb_array")
        try:
            gif_path, learned_path, success = record_demo(
                env, agent, args.record, seed=args.seed, fps=args.fps
            )
            if args.snapshot:
                save_snapshot(env, args.snapshot, learned_path=learned_path)
        finally:
            env.close()
        print(f"Demo: {gif_path}")
        return 0 if success else 2

    env = _environment(config, args, render_mode="human")
    try:
        result = evaluate_agent(env, agent, episodes=1, seed=args.seed, render=True)
    finally:
        env.close()
    return 0 if result.records[0].success else 2


def _parse_int_list(value: str) -> list[int]:
    try:
        result = [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not result or any(item <= 1 for item in result):
        raise argparse.ArgumentTypeError("all values must be greater than 1")
    return result


def command_benchmark(args: argparse.Namespace) -> int:
    """Benchmark Q-learning over several seeds and generated maze sizes."""
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for size in args.sizes:
        for seed in args.seeds:
            config = generate_maze(size, size, args.obstacle_density, seed)
            env = MazeEnv(
                grid_size=config.grid_size,
                start=config.start,
                goal=config.goal,
                obstacles=config.obstacles,
                max_steps=args.max_steps,
            )
            agent = QLearningAgent(
                alpha=args.alpha,
                gamma=args.gamma,
                epsilon=args.epsilon,
                epsilon_min=args.epsilon_min,
                epsilon_decay=args.epsilon_decay,
                seed=seed,
            )
            trained = train_agent(env, agent, episodes=args.episodes, seed=seed)
            evaluated = evaluate_agent(env, agent, episodes=args.eval_episodes, seed=seed + 10_000)
            summary = summarize(evaluated.records)
            rows.append(
                {
                    "size": size,
                    "seed": seed,
                    "training_episodes": len(trained.records),
                    "success_rate": summary["success_rate"],
                    "average_steps": summary["average_steps"],
                    "optimal_steps": evaluated.records[0].optimal_steps,
                    "path_efficiency_ratio": summary["path_efficiency_ratio"],
                }
            )
            env.close()
            print(f"size={size} seed={seed} success={summary['success_rate']:.1%}")

    csv_path = output / "benchmark.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path = output / "benchmark.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Benchmark: {csv_path}, {json_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the public CLI parser."""
    parser = argparse.ArgumentParser(
        prog="maze-solver",
        description="Train and evaluate a tabular Q-learning maze solver.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(command: argparse.ArgumentParser) -> None:
        _add_maze_arguments(command)
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--output", default="artifacts/latest")

    train = subparsers.add_parser("train", help="train and save a Q-table")
    add_common(train)
    train.add_argument("--episodes", type=_positive_int, default=500)
    train.add_argument("--alpha", type=float, default=0.1)
    train.add_argument("--gamma", type=_probability, default=0.95)
    train.add_argument("--epsilon", type=_probability, default=1.0)
    train.add_argument("--epsilon-min", type=_probability, default=0.05)
    train.add_argument("--epsilon-decay", type=_probability, default=0.99)
    train.add_argument("--early-stop-window", type=_positive_int, default=50)
    train.add_argument("--render", action="store_true")
    train.set_defaults(handler=command_train)

    evaluate = subparsers.add_parser("evaluate", help="evaluate a saved model")
    add_common(evaluate)
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--episodes", type=_positive_int, default=50)
    evaluate.add_argument("--render", action="store_true")
    evaluate.set_defaults(handler=command_evaluate)

    demo = subparsers.add_parser("demo", help="show or record a greedy run")
    add_common(demo)
    demo.add_argument("--model", required=True)
    demo.add_argument("--record", help="GIF destination; uses headless RGB rendering")
    demo.add_argument("--snapshot", help="optional PNG destination")
    demo.add_argument("--fps", type=_positive_int, default=6)
    demo.set_defaults(handler=command_demo)

    benchmark = subparsers.add_parser("benchmark", help="run a multi-seed benchmark")
    benchmark.add_argument("--sizes", type=_parse_int_list, default=[5, 8, 10])
    benchmark.add_argument("--seeds", type=_parse_int_list, default=[7, 21, 42])
    benchmark.add_argument("--episodes", type=_positive_int, default=500)
    benchmark.add_argument("--eval-episodes", type=_positive_int, default=30)
    benchmark.add_argument("--obstacle-density", type=_probability, default=0.15)
    benchmark.add_argument("--max-steps", type=_positive_int)
    benchmark.add_argument("--alpha", type=float, default=0.1)
    benchmark.add_argument("--gamma", type=_probability, default=0.95)
    benchmark.add_argument("--epsilon", type=_probability, default=1.0)
    benchmark.add_argument("--epsilon-min", type=_probability, default=0.05)
    benchmark.add_argument("--epsilon-decay", type=_probability, default=0.99)
    benchmark.add_argument("--output", default="artifacts/benchmark")
    benchmark.set_defaults(handler=command_benchmark)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point with concise user-facing errors."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (FileNotFoundError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
