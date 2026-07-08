import json
import os
import re
from typing import Any


def read_schema(schema_path: str, *, mode: str = "compact", max_chars: int | None = 4000) -> str:
    '''
    Read the flight database schema file and return a compact string suitable for prompts.

    The provided `data/flight_database.schema` is JSON containing:
      - ents: tables and their columns
      - links: lightweight foreign-key style hints
    '''
    with open(schema_path, "r") as f:
        schema_obj: Any = json.load(f)

    ents: dict[str, dict[str, Any]] = schema_obj.get("ents", {}) or {}
    links: dict[str, dict[str, Any]] = schema_obj.get("links", {}) or {}

    if mode not in {"compact", "full", "none"}:
        raise ValueError(f"schema mode must be one of compact/full/none, got: {mode}")
    if mode == "none":
        return ""

    lines: list[str] = []
    lines.append("Tables and columns:")
    for table in sorted(ents.keys()):
        cols = sorted(ents[table].keys())
        if mode == "full":
            # Include utterances for columns when available (more human-readable).
            col_parts = []
            for col in cols:
                utt = ents[table].get(col, {}).get("utt")
                if utt and utt != col:
                    col_parts.append(f"{col} ({utt})")
                else:
                    col_parts.append(col)
            col_str = ", ".join(col_parts)
        else:
            col_str = ", ".join(cols)
        lines.append(f"- {table}({col_str})")

    nonempty_links = {t: m for t, m in links.items() if m}
    if nonempty_links:
        lines.append("")
        lines.append("Links (join hints):")
        for table in sorted(nonempty_links.keys()):
            # Format: fare: fare_basis->fare_basis.fare_basis_code, airline->fare.fare_airline, ...
            parts = []
            for other_table, other_col in sorted(nonempty_links[table].items()):
                parts.append(f"{other_table}.{other_col}")
            lines.append(f"- {table}: " + ", ".join(parts))

    schema_text = "\n".join(lines).strip()
    if max_chars is not None and len(schema_text) > max_chars:
        schema_text = schema_text[: max_chars - 3].rstrip() + "..."
    return schema_text


_SPECIAL_TOKEN_RE = re.compile(r"<\s*(?:start_of_turn|end_of_turn)\s*>", re.IGNORECASE)


def extract_sql_query(response: str) -> str:
    '''
    Extract the SQL query from the model's response
    '''
    if response is None:
        return ""

    text = str(response)
    text = _SPECIAL_TOKEN_RE.sub("", text)
    text = text.replace("\r\n", "\n").strip()

    # Prefer fenced SQL blocks: ```sql ... ```
    fence_match = re.search(r"```(?:sql)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fence_match:
        candidate = fence_match.group(1).strip()
    else:
        # Fall back to the first SELECT/WITH occurrence.
        sel_match = re.search(r"\b(SELECT|WITH)\b", text, flags=re.IGNORECASE)
        if not sel_match:
            # Fall back to first line after common labels.
            text = re.sub(r"^\s*(SQL|Query)\s*:\s*", "", text, flags=re.IGNORECASE)
            candidate = text.strip().split("\n", 1)[0].strip()
        else:
            candidate = text[sel_match.start() :].strip()

    # Remove leading labels and stray fences.
    candidate = re.sub(r"^\s*(SQL|Query)\s*:\s*", "", candidate, flags=re.IGNORECASE).strip()
    candidate = candidate.strip("`").strip()

    # If there are multiple statements, keep the first terminated by ';' when present.
    semi_idx = candidate.find(";")
    if semi_idx != -1:
        candidate = candidate[: semi_idx + 1].strip()

    # Normalize whitespace a bit (helps sqlite parser and evaluation stability).
    candidate = re.sub(r"\s+", " ", candidate).strip()
    return candidate

def save_logs(output_path, sql_em, record_em, record_f1, error_msgs):
    '''
    Save the logs of the experiment to files.
    You can change the format as needed.
    '''
    with open(output_path, "w") as f:
        f.write(f"SQL EM: {sql_em}\nRecord EM: {record_em}\nRecord F1: {record_f1}\nModel Error Messages: {error_msgs}\n")
