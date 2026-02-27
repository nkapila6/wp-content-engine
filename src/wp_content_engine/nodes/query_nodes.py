# generates queries for both ripgrep and ddgs
# 25/02/2026 01:23PM Nikhil Kapila

from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field

from langchain_core.runnables import RunnableConfig

from wp_content_engine.state.state import AgentState
from wp_content_engine.llm import llm


class QueryResponse(BaseModel):
    web_queries: List[str] = Field(..., min_length=1)
    rg_queries: List[str] = Field(..., min_length=1)


def enhance_queries_node(
    state: AgentState, config: RunnableConfig | None = None
) -> Dict[str, List[str]]:
    """
    LLM step to generate and refine queries for web search and ripgrep.

    Reads:
        - topic
        - raw_prompt
        - persona, example_post (optional)
        - primary_keyword / seo_keywords

    Writes:
        - ddgs_queries:List[str]
        - rg_queries:List[str]
    """

    topic = state.get("topic", "")
    raw_prompt = state.get("raw_prompt", "")
    persona = state.get("persona", "")
    example_post = state.get("example_post", "")

    primary_keyword = state.get("primary_keyword", "")
    seo_keywords = state.get("seo_keywords", [])

    existing_ddgs = state.get("ddgs_queries", []) or []
    existing_rg = state.get("rg_queries", []) or []

    user_msg = f"""
Topic: {topic}
Raw prompt: {raw_prompt}
Persona (optional): {persona}
Example post (optional, can be empty): {example_post}
Primary SEO keyword (optional): {primary_keyword}
Other SEO keywords (optional): {", ".join(seo_keywords)}
Existing web queries (optional): {existing_ddgs}
Existing ripgrep queries (optional): {existing_rg}

Task:
1. Propose 3-6 high-signal web search queries for DDGS/Google.
   - These should be natural search phrases a human would type.

2. Propose 5-10 ripgrep search patterns for a LOCAL text/markdown knowledge base.
   CRITICAL ripgrep rules:
   - Each pattern must be 1-2 words MAXIMUM (e.g. "curriculum", "EYFS", "HPL", "fees", "pastoral")
   - Multi-word phrases WILL FAIL to match. Keep patterns SHORT.
   - Use key domain terms, school names, acronyms, or single concepts.
   - Good examples: "British curriculum", "KHDA", "Al Quoz", "sibling discount", "BSO"
   - Bad examples: "Best British Schools in Dubai 2026" (too long, won't match)

3. Avoid near-duplicates.
""".strip()

    try:
        structured_llm = llm.with_structured_output(QueryResponse)
        resp = structured_llm.invoke(user_msg, config=config)
    except Exception as e:
        errors = list(state.get("errors", []))
        errors.append(f"enhance_queries_node error: {e}")
        return {"errors": errors}

    def dedupe(existing: List[str], new: List[str]) -> List[str]:
        seen = set()
        combined: List[str] = []
        for q in existing + new:
            q_norm = q.strip()
            if not q_norm or q_norm in seen:
                continue
            seen.add(q_norm)
            combined.append(q_norm)
        return combined

    ddgs_queries = dedupe(existing_ddgs, resp.web_queries)
    rg_queries = dedupe(existing_rg, resp.rg_queries)

    return {
        "ddgs_queries": ddgs_queries,
        "rg_queries": rg_queries,
    }
