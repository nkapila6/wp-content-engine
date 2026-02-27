# dddgs lib node definitions
# 25/02/2026 01:15PM Nikhil Kapila

from __future__ import annotations

import time
from typing import Dict, List

from ddgs import DDGS
from langchain_core.runnables import RunnableConfig

from wp_content_engine.state.state import AgentState, SearchResult
from wp_content_engine.utils.fetch import fetch_all_content
from wp_content_engine.prompts.prompts import ddgs_summary_prompt
from wp_content_engine.llm import llm

DDGS_BACKEND = "api"
DDGS_DELAY = 2


def ddgs_search_node(
    state: AgentState, *_args, **_kwargs
) -> Dict[str, Dict[str, List[SearchResult]]]:
    """
    Node: run DDGS web search for each query in ddgs_queries and fetch page content.

    Reads from AgentState:
      - ddgs_queries: List[str]
      - ddgs_num_results: int (optional, default 10)

    Writes:
      - ddgs_results: Dict[str, List[SearchResult]]
    """
    queries = state.get("ddgs_queries") or []
    if not queries:
        return {}

    num_results = state.get("ddgs_num_results", 5)
    ddgs = DDGS()

    results_by_query: Dict[str, List[SearchResult]] = {}

    for idx, query in enumerate(queries):
        if idx > 0:
            time.sleep(DDGS_DELAY)

        try:
            raw_results = list(ddgs.text(query, max_results=num_results, backend=DDGS_BACKEND))
        except Exception as e:
            print(f"DDGS search failed for '{query}': {e}")
            results_by_query[query] = []
            continue

        docs = fetch_all_content(raw_results)

        search_docs: List[SearchResult] = []
        for doc in docs:
            content = doc.get("text", "")
            if not content:
                continue
            search_docs.append(
                {
                    "content": content,
                    "url": doc.get("url"),
                }
            )

        results_by_query[query] = search_docs

    return {"ddgs_results": results_by_query}


def ddgs_summary_node(state: AgentState, config) -> Dict[str, str]:
    """
    Node: summarize DDGS web search results into a focused summary.

    Reads from AgentState:
      - ddgs_results: Dict[str, List[SearchResult]]
      - topic: str
      - target_words_total: int

    Writes:
      - ddgs_result_summary: str
    """
    ddgs_results = state.get("ddgs_results") or {}
    topic = state.get("topic", "")
    target_words_total = state.get("target_words_total", 1500)

    if not ddgs_results or not topic:
        return {}

    try:
        system_msg, user_msg = ddgs_summary_prompt(
            topic, ddgs_results, target_words_total
        )
        response = llm.invoke([system_msg, user_msg], config=config)
        summary = response.content.strip()
        return {"ddgs_result_summary": summary}
    except Exception as e:
        errors = list(state.get("errors", []))
        errors.append(f"ddgs_summary_node error: {e}")
        return {"errors": errors}
