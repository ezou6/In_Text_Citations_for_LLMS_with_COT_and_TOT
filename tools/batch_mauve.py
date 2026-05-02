#!/usr/bin/env python3
"""
Run MAUVE-only eval (eval.py --mauve_only) on many result JSONs and merge scores.

Example (from repo root, after pip install -r requirements.txt):

  .venv/bin/python tools/batch_mauve.py \\
    --glob 'result/qasa-*-quick_test200.json' \\
    --mauve-device -1 \\
    -o citation_results/mauve_scores.json

  .venv/bin/python tools/batch_mauve.py result/a.json result/b.json -o out.json

Each input must be run.py-style JSON with top-level {\"data\": [...]} and rows with
question, answer, output. Uses eval.py --mauve_only (MAUVE only; no ROUGE). Per-file
metrics go to {path}.score.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _collect_paths(globs: list[str], files: list[Path]) -> list[Path]:
    root = Path(".")
    seen: set[Path] = set()
    out: list[Path] = []
    for g in globs:
        for p in root.glob(g):
            if not p.is_file():
                continue
            if p.suffix != ".json" or p.name.endswith(".score.json"):
                continue
            rp = p.resolve()
            if rp not in seen:
                seen.add(rp)
                out.append(p)
    for p in files:
        if not p.is_file():
            continue
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(p)
    return sorted(out, key=lambda x: str(x))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("files", nargs="*", type=Path, help="Result JSON paths")
    ap.add_argument(
        "--glob",
        action="append",
        dest="globs",
        default=[],
        metavar="PATTERN",
        help="Glob relative to cwd (repeatable), e.g. result/qasa-*-quick_test200.json",
    )
    ap.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path("citation_results/mauve_scores.json"),
        help="Write merged {\"<basename>\": {\"mauve\": ...}, ...}",
    )
    ap.add_argument(
        "--eval-py",
        type=Path,
        default=Path("eval.py"),
        help="Path to eval.py",
    )
    ap.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Python interpreter to run eval.py",
    )
    ap.add_argument(
        "--mauve-device",
        type=int,
        default=None,
        help="Passed to eval.py --mauve_device (omit for auto CPU/GPU)",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a file if its .score already exists",
    )
    args = ap.parse_args()

    paths = _collect_paths(args.globs, args.files)
    if not paths:
        print("No input JSON files matched.", file=sys.stderr)
        sys.exit(1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged: dict[str, dict] = {}

    for p in paths:
        score_path = Path(str(p) + ".score")
        if args.skip_existing and score_path.is_file():
            row = json.loads(score_path.read_text(encoding="utf-8"))
            merged[p.name] = {"mauve": row.get("mauve"), "from_existing_score": True}
            continue

        cmd = [
            str(args.python),
            str(args.eval_py),
            "--f",
            str(p),
            "--mauve_only",
        ]
        if args.mauve_device is not None:
            cmd += ["--mauve_device", str(args.mauve_device)]

        print("Running:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        row = json.loads(score_path.read_text(encoding="utf-8"))
        merged[p.name] = {k: row[k] for k in row if k in ("mauve", "length", "str_em", "str_hit", "rougeLsum")}

    args.out.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(merged)} entries to {args.out}")


if __name__ == "__main__":
    main()
