"""Backward-compatible project entry point.

Prefer the installed ``maze-solver`` command or ``python -m static_maze_solver``.
"""

from __future__ import annotations

import sys

from static_maze_solver.cli import main

if __name__ == "__main__":
    arguments = sys.argv[1:] or ["train"]
    raise SystemExit(main(arguments))
