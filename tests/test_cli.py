"""End-to-end smoke tests for public CLI commands."""

from static_maze_solver.cli import main


def test_train_evaluate_and_record_demo(tmp_path) -> None:
    output = tmp_path / "run"
    assert (
        main(
            [
                "train",
                "--maze",
                "small",
                "--episodes",
                "400",
                "--epsilon-decay",
                "0.9",
                "--early-stop-window",
                "10",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    model = output / "q_table.json"
    assert model.exists()
    assert (output / "train_summary.json").exists()
    assert (output / "convergence.png").exists()

    evaluation = tmp_path / "evaluation"
    assert (
        main(
            [
                "evaluate",
                "--maze",
                "small",
                "--model",
                str(model),
                "--episodes",
                "3",
                "--output",
                str(evaluation),
            ]
        )
        == 0
    )

    gif = tmp_path / "demo.gif"
    png = tmp_path / "demo.png"
    assert (
        main(
            [
                "demo",
                "--maze",
                "small",
                "--model",
                str(model),
                "--record",
                str(gif),
                "--snapshot",
                str(png),
            ]
        )
        == 0
    )
    assert gif.exists()
    assert png.exists()


def test_benchmark_command_writes_reports(tmp_path) -> None:
    output = tmp_path / "benchmark"
    assert (
        main(
            [
                "benchmark",
                "--sizes",
                "3",
                "--seeds",
                "7",
                "--episodes",
                "5",
                "--eval-episodes",
                "2",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert (output / "benchmark.csv").exists()
    assert (output / "benchmark.json").exists()
