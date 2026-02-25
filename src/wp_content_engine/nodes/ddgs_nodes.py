# dddgs lib node definitions
# 25/02/2026 01:15PM Nikhil Kapila

from __future__ import annotations

from typing import Dict, List
from ddgs import DDGS
from langchain_core.runnables import RunnableConfig

from wp_content_engine.state.state import AgentState, SearchResult
from wp_content_engine.utils.fetch import fetch_all_content

def ddgs_search_node(state: AgentState, *_args, **_kwargs) -> Dict[str, Dict[str, List[SearchResult]]]:
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

    for query in queries:
        raw_results = list(ddgs.text(query, max_results=num_results, backend="google"))
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

