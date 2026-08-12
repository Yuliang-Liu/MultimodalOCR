#!/usr/bin/env python3
"""Synchronize the static leaderboard data from MDPBench's README table.

The README is the source of truth for the initial official results. This tool
parses its HTML table without third-party dependencies and preserves the stable
evaluation identifiers already present in leaderboard.json.

Examples:
    python tools/sync_leaderboard.py --check
    python tools/sync_leaderboard.py --write
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


LANGUAGES = [
    "DE", "EN", "ES", "FR", "ID", "IT", "NL", "PT", "VI",
    "AR", "HI", "JP", "KO", "RU", "TH", "ZH", "ZH-T",
]
SCORE_KEYS = ["overall", "digital", "photo", "latin", "non_latin", "private"]


class ResultsTableParser(HTMLParser):
    """Collect table rows and ``rowspan`` values from an HTML README table."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[tuple[str, int]]] = []
        self._row: list[tuple[str, int]] | None = None
        self._cell_text: list[str] | None = None
        self._cell_rowspan = 1

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            self._cell_text = []
            attrs_dict = dict(attrs)
            self._cell_rowspan = int(attrs_dict.get("rowspan") or 1)

    def handle_data(self, data: str) -> None:
        if self._cell_text is not None:
            self._cell_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._cell_text is not None and self._row is not None:
            text = " ".join("".join(self._cell_text).split())
            self._row.append((text, self._cell_rowspan))
            self._cell_text = None
        elif tag == "tr" and self._row:
            self.rows.append(self._row)
            self._row = None


def parse_number(value: str, *, model_name: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{model_name}: expected a numeric score, got {value!r}") from exc


def parse_results(readme: Path) -> list[dict[str, Any]]:
    parser = ResultsTableParser()
    parser.feed(readme.read_text(encoding="utf-8"))

    current_type: str | None = None
    parsed: list[dict[str, Any]] = []
    for row in parser.rows:
        cells = [text for text, _ in row]
        if not cells or cells[0] in {"Model Type", "All"}:
            continue

        # A category row has model type, model name, and 23 scores. Subsequent
        # rows have the model name and the same 23 scores.
        if len(cells) == 25:
            current_type = re.sub(r"\s+", " ", cells[0])
            model_name, score_values = cells[1], cells[2:]
        elif len(cells) == 24 and current_type is not None:
            model_name, score_values = cells[0], cells[1:]
        else:
            continue

        if len(score_values) != 23:
            raise ValueError(
                f"{model_name}: expected 23 score columns from the README table, "
                f"found {len(score_values)}"
            )

        numeric = [parse_number(value, model_name=model_name) for value in score_values]
        latin_languages = numeric[4:13]
        non_latin_languages = numeric[14:22]
        parsed.append(
            {
                "type": current_type,
                "source_name": model_name,
                "overall": numeric[0],
                "digital": numeric[1],
                "photo": numeric[2],
                "latin": numeric[3],
                "non_latin": numeric[13],
                "private": numeric[22],
                "languages": dict(zip(LANGUAGES, latin_languages + non_latin_languages, strict=True)),
            }
        )

    if not parsed:
        raise ValueError("No Main Results rows were found in the README.")
    return parsed


def type_label(value: str) -> str:
    compact = value.replace(" ", "")
    return {
        "GeneralVLMs": "General VLM",
        "SpecializedVLMs": "Specialized VLM",
        "PipelineTools": "Pipeline Tool",
    }[compact]


def build_document(source_rows: list[dict[str, Any]], existing: dict[str, Any]) -> dict[str, Any]:
    existing_models = existing.get("models", [])
    identifiers = {
        model.get("source_name", model["name"]): model["name"]
        for model in existing_models
    }
    source_names = {model.get("source_name") for model in existing_models if model.get("source_name")}
    models = []

    for source in source_rows:
        source_name = source["source_name"]
        model: dict[str, Any] = {
            "type": type_label(source["type"]),
            "name": identifiers.get(source_name, source_name),
            "overall": source["overall"],
            "digital": source["digital"],
            "photo": source["photo"],
            "latin": source["latin"],
            "non_latin": source["non_latin"],
            "private": source["private"],
            "languages": source["languages"],
        }
        if source_name in source_names:
            model["source_name"] = source_name
        models.append(model)

    missing = set(identifiers) - {row["source_name"] for row in source_rows}
    if missing:
        raise ValueError("Models in leaderboard.json absent from README: " + ", ".join(sorted(missing)))

    benchmark = dict(existing.get("benchmark", {}))
    benchmark["updated"] = date.today().isoformat()
    return {"benchmark": benchmark, "models": models}


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Sync leaderboard.json from the MDPBench README results table.")
    parser.add_argument("--readme", type=Path, default=root.parent / "README.md")
    parser.add_argument("--output", type=Path, default=root / "leaderboard.json")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true", help="Fail if the generated data differs from the current JSON.")
    action.add_argument("--write", action="store_true", help="Write the generated data to leaderboard.json.")
    args = parser.parse_args()

    existing = json.loads(args.output.read_text(encoding="utf-8"))
    generated = build_document(parse_results(args.readme), existing)
    current = json.dumps(existing, ensure_ascii=False, sort_keys=True)
    candidate = json.dumps(generated, ensure_ascii=False, sort_keys=True)
    changed = current != candidate

    if args.check:
        if changed:
            print("Leaderboard JSON is out of sync with README. Run with --write to update it.", file=sys.stderr)
            return 1
        print(f"In sync: {len(generated['models'])} models, {len(LANGUAGES)} language scores each.")
        return 0

    args.output.write_text(json.dumps(generated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(generated['models'])} models to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
