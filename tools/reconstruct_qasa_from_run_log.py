#!/usr/bin/env python3
"""
Reconstruct a run.py-style QASA result JSON from stdout captured in a run log.

Logs contain blocks like:
  ... - INFO - Question: ...
  ... - INFO - Gold answer: ...
  ... - INFO - CoT output: ... (possibly multiline until Final model output)
  ... - INFO - Final model output: ...
  ... - INFO - Gold ctxs: ['1']

Rows are merged with eval JSON in the same order as quick_test + indices file.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from copy import deepcopy
from pathlib import Path


def _after(haystack: str, needle: str) -> str | None:
    i = haystack.find(needle)
    if i < 0:
        return None
    return haystack[i + len(needle) :].strip()


def _find_field_line(lines: list[str], start: int, field: str) -> tuple[int, str] | None:
    """Return (index, full line) of first line at or after start containing ' - INFO - {field}: '."""
    prefix = f" - INFO - {field}: "
    for j in range(start, len(lines)):
        if prefix in lines[j]:
            return j, lines[j]
    return None


def parse_log_records(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    records: list[dict] = []
    i = 0
    final_tag = " - INFO - Final model output: "

    while i < len(lines):
        q_hit = _find_field_line(lines, i, "Question")
        if not q_hit:
            break
        i, q_line = q_hit
        q = _after(q_line, " - INFO - Question: ")
        if q is None:
            i += 1
            continue

        ga_hit = _find_field_line(lines, i + 1, "Gold answer")
        if not ga_hit:
            raise ValueError(f"No Gold answer after Question at line {i + 1}: {q[:80]!r}")
        i, ga_line = ga_hit
        gold_answer_log = _after(ga_line, " - INFO - Gold answer: ") or ""

        cot_hit = _find_field_line(lines, i + 1, "CoT output")
        if not cot_hit:
            raise ValueError(f"No CoT output after Gold answer for question: {q[:80]!r}")
        i, cot_line = cot_hit
        cot = _after(cot_line, " - INFO - CoT output: ") or ""
        i += 1

        while i < len(lines):
            line = lines[i]
            if final_tag in line:
                idx = line.find(final_tag)
                prefix = line[:idx]
                final_out = line[idx + len(final_tag) :].strip()
                rest = prefix.strip()
                if rest:
                    first_word = rest.split()[0] if rest.split() else ""
                    if not re.match(r"^\d{4}-\d{2}-\d{2}", first_word):
                        cot = (cot + "\n" + rest).strip() if cot else rest
                i += 1
                break
            if " - INFO - Question: " in line:
                raise ValueError("Unterminated CoT block before next Question")
            cot = (cot + "\n" + line).strip() if cot else line
            i += 1
        else:
            raise ValueError(f"EOF while reading CoT for question: {q[:80]!r}")

        gc_hit = _find_field_line(lines, i, "Gold ctxs")
        if not gc_hit:
            raise ValueError(f"No Gold ctxs after Final model output for: {q[:80]!r}")
        i, gc_line = gc_hit
        raw_ctx = _after(gc_line, " - INFO - Gold ctxs: ")
        if raw_ctx is None:
            raise ValueError("Malformed Gold ctxs line")
        try:
            gold_ctxs = ast.literal_eval(raw_ctx.strip())
        except (SyntaxError, ValueError) as e:
            raise ValueError(f"Could not parse Gold ctxs: {raw_ctx!r}") from e
        if not isinstance(gold_ctxs, list):
            gold_ctxs = [str(gold_ctxs)]

        records.append(
            {
                "question": q,
                "gold_answer_log": gold_answer_log,
                "cot_output": cot,
                "output": final_out,
                "gold_ctxs": [str(x) for x in gold_ctxs],
            }
        )
        i += 1

    return records


def _normalize_cot_output(cot: str) -> str:
    """Strip duplicate final-answer tail from logs; match run.py style blank line before Most critical."""
    m = re.search(r"^(.*Most critical documents \(most to least\):[^\n]+)", cot, re.DOTALL)
    body = m.group(1).strip() if m else cot.strip()
    if "\n\nMost critical documents" not in body and "\nMost critical documents" in body:
        body = body.replace("\nMost critical documents", "\n\nMost critical documents", 1)
    return body


def _cot_eval_refs(row: dict) -> dict:
    docs = row.get("docs") or []
    by_rank = []
    for j, d in enumerate(docs):
        if not isinstance(d, dict):
            continue
        suf = d.get("id_suffix")
        rank = str(suf).strip() if suf is not None and str(suf).strip() else str(j + 1)
        by_rank.append({"rank": rank, "passage_id": d.get("id", "")})
    return {"gold_ctxs": list(row.get("gold_ctxs") or []), "docs_by_rank": by_rank}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--log", type=Path, required=True, help="Captured run.py stdout log")
    ap.add_argument("--eval", type=Path, default=Path("data/qasa_eval.json"), help="Full eval JSON list")
    ap.add_argument(
        "--indices",
        type=Path,
        default=Path("configs/qasa_eval_quick200_seed42_indices.json"),
        help="Fixed index list JSON",
    )
    ap.add_argument("--quick-test", type=int, default=200, help="First N indices to use")
    ap.add_argument(
        "--args-from",
        type=Path,
        default=None,
        help="Existing result JSON to copy top-level 'args' from (optional)",
    )
    ap.add_argument("-o", "--out", type=Path, required=True, help="Output JSON path")
    args = ap.parse_args()

    records = parse_log_records(args.log)
    spec = json.loads(args.indices.read_text())
    indices = spec["indices"] if isinstance(spec, dict) else spec
    indices = [int(x) for x in indices[: args.quick_test]]

    eval_data = json.loads(args.eval.read_text())
    base_rows = [deepcopy(eval_data[i]) for i in indices]

    if len(records) != len(base_rows):
        print(
            f"Warning: log has {len(records)} Question blocks but eval slice has {len(base_rows)} rows.",
            file=sys.stderr,
        )

    n = min(len(records), len(base_rows))
    merged = []
    for k in range(n):
        r = records[k]
        row = base_rows[k]
        if row["question"].strip() != r["question"].strip():
            raise ValueError(
                f"Row {k}: question mismatch log vs eval:\n  LOG: {r['question']!r}\n  EVAL: {row['question']!r}"
            )
        row["cot_output"] = _normalize_cot_output(r["cot_output"])
        row["output"] = r["output"]
        row["gold_ctxs"] = r["gold_ctxs"]
        row["cot_eval_refs"] = _cot_eval_refs(row)
        merged.append(row)

    args_obj: dict | None = None
    if args.args_from is not None:
        prev = json.loads(args.args_from.read_text())
        args_obj = prev.get("args")
    if args_obj is None:
        args_obj = {
            "note": "Reconstructed from log; fill args from original run if needed.",
            "reconstructed_from_log": str(args.log),
        }

    out_obj = {"args": args_obj, "data": merged}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_obj, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {n} rows to {args.out}")


if __name__ == "__main__":
    main()
