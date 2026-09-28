"""Apply editable, directed-edge lighting and sidewalk data to a walking graph."""

from pathlib import Path

import pandas as pd

from route_core import _tag


REQUIRED_COLUMNS = {"Segment ID", "Lighting edit", "Sidewalk edit"}
VALID_VALUES = {"yes", "no", "unknown"}


def _status(value, column, segment_id):
    status = "unknown" if pd.isna(value) else str(value).strip().lower()
    if status == "":
        status = "unknown"
    if status not in VALID_VALUES:
        raise ValueError(
            f"Invalid {column} value {value!r} for segment {segment_id}. "
            "Use yes, no, or unknown."
        )
    return status


def apply_condition_table(graph, table):
    """Modify a graph copy using rows from the workbook's Segments sheet."""
    missing = REQUIRED_COLUMNS - set(table.columns)
    if missing:
        raise ValueError(f"Segments sheet is missing columns: {', '.join(sorted(missing))}")

    rows = table.dropna(subset=["Segment ID"])
    if rows["Segment ID"].duplicated().any():
        raise ValueError("Segments sheet contains duplicate Segment IDs.")

    edits = {}
    for _, row in rows.iterrows():
        segment_id = str(row["Segment ID"]).strip()
        edits[segment_id] = (
            _status(row["Lighting edit"], "Lighting edit", segment_id),
            _status(row["Sidewalk edit"], "Sidewalk edit", segment_id),
        )

    matched = set()
    for u, v, key, edge in graph.edges(keys=True, data=True):
        # The user requested that missing/unknown lighting count as unlit.
        if _tag(edge.get("lit")) not in {"yes", "24/7", "automatic", "no", "disused"}:
            edge["lit"] = "no"

        segment_id = f"{u}|{v}|{key}"
        if segment_id not in edits:
            continue
        matched.add(segment_id)
        lighting, sidewalk = edits[segment_id]
        edge["lit"] = "yes" if lighting == "yes" else "no"
        for field in ("sidewalk", "sidewalk:left", "sidewalk:right"):
            edge.pop(field, None)
        if sidewalk != "unknown":
            edge["sidewalk"] = sidewalk

    unmatched = set(edits) - matched
    if unmatched:
        example = next(iter(unmatched))
        raise ValueError(
            f"{len(unmatched)} Segment IDs in Excel were not found in the GraphML "
            f"(for example {example}). Keep the matching graph and workbook together."
        )
    return graph


def load_excel_conditions(graph, workbook_path: Path):
    table = pd.read_excel(
        workbook_path,
        sheet_name="Segments",
        usecols=lambda column: column in REQUIRED_COLUMNS,
        dtype={"Segment ID": str},
    )
    return apply_condition_table(graph, table)
