# ripgrep executor
# 25/02/2026 02:10PM Nikhil Kapila

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, List

from wp_content_engine.state.state import AgentState, RipgrepMatch
from wp_content_engine.prompts.prompts import rg_summary_prompt
from wp_content_engine.llm import llm


def exec_ripgrep_queries_node(
    state: AgentState, *_args, **_kwargs
) -> Dict[str, Dict[str, List[RipgrepMatch]]]:
    """
    Node: execute ripgrep over kb_root for each query in rg_queries.

    Reads:
      - kb_root: str
      - rg_queries: List[str]

    Writes:
      - rg_results: Dict[str, List[RipgrepMatch]]
    """
    kb_root = state.get("kb_root")
    queries = state.get("rg_queries") or []

    if not kb_root or not queries:
        return {}

    root = Path(kb_root).expanduser().resolve()
    if not root.exists():
        errors = list(state.get("errors", []))
        errors.append(f"exec_ripgrep_queries_node: kb_root does not exist: {kb_root}")
        return {"errors": errors}

    results_by_query: Dict[str, List[RipgrepMatch]] = {}

    for query in queries:
        words = query.split()
        if len(words) > 2:
            pattern = "|".join(words)
        else:
            pattern = query

        cmd = ["rg", "-i", "-uuu", "-C", "3", "--json", pattern, str(root)]
        proc = subprocess.run(cmd, capture_output=True, text=True)

        # 0 = matches, 1 = no matches, others = error
        if proc.returncode not in (0, 1):
            errors = list(state.get("errors", []))
            errors.append(
                f"exec_ripgrep_queries_node: rg error for '{query}': {proc.stderr.strip()}"
            )
            results_by_query[query] = []
            continue

        matches: List[RipgrepMatch] = []
        context_buf: List[str] = []

        for line in proc.stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            etype = event.get("type", "")

            if etype == "context":
                data = event.get("data", {})
                ctx_text = data.get("lines", {}).get("text", "").rstrip("\n")
                context_buf.append(ctx_text)

            elif etype == "match":
                data = event.get("data", {})
                path = data.get("path", {}).get("text", "")
                line_text = data.get("lines", {}).get("text", "").rstrip("\n")
                line_number = data.get("line_number") or 0

                matches.append(
                    {
                        "file_path": path,
                        "line": line_number,
                        "line_text": line_text,
                        "context_before": list(context_buf),
                        "context_after": [],
                    }
                )
                context_buf.clear()

            elif etype == "end":
                if matches and context_buf:
                    matches[-1]["context_after"] = list(context_buf)
                context_buf.clear()

        results_by_query[query] = matches

    return {"rg_results": results_by_query}


def rg_summary_node(state: AgentState, config) -> Dict[str, str]:
    """
    Node: summarize ripgrep KB search results into a synthesized summary.

    Reads from AgentState:
      - rg_results: Dict[str, List[RipgrepMatch]]
      - topic: str

    Writes:
      - rg_result_summary: str
    """
    rg_results = state.get("rg_results") or {}
    topic = state.get("topic", "")

    has_matches = any(matches for matches in rg_results.values())
    if not rg_results or not topic or not has_matches:
        return {}

    try:
        system_msg, user_msg = rg_summary_prompt(topic, rg_results)
        response = llm.invoke([system_msg, user_msg], config=config)
        summary = response.content.strip()
        return {"rg_result_summary": summary}
    except Exception as e:
        errors = list(state.get("errors", []))
        errors.append(f"rg_summary_node error: {e}")
        return {"errors": errors}
