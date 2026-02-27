# node definitions for different nodes
# 24/02/2026 01:06PM Nikhil Kapila

from __future__ import annotations

import os
import re
from typing import Dict, List

from pydantic import BaseModel, Field

from langchain_core.runnables import RunnableConfig

from wp_content_engine.state.state import AgentState, Plan
from wp_content_engine.prompts.prompts import (
    condenser_prompt,
    planner_prompt,
    draft_task_prompt,
    task_revision_prompt,
    stitcher_prompt,
    styler_prompt,
    seo_prompt,
)
from wp_content_engine.llm import llm


def condenser_node(
    state: AgentState, config: RunnableConfig | None = None
) -> Dict[str, str]:
    """
    Node: condense DDGS and RG summaries into budgeted context for planner/writers.

    Reads from AgentState:
      - ddgs_result_summary: str
      - rg_result_summary: str
      - topic: str
      - persona: str
      - example_post: str
      - primary_keyword: str
      - secondary_keywords: List[str]
      - seo_keywords: List[str]

    Writes:
      - condensed_content: str
    """
    ddgs_summary = state.get("ddgs_result_summary", "")
    rg_summary = state.get("rg_result_summary", "")
    topic = state.get("topic", "")
    persona = state.get("persona", "")
    example_post = state.get("example_post", "")
    primary_keyword = state.get("primary_keyword", "")
    secondary_keywords = state.get("secondary_keywords", [])
    seo_keywords = state.get("seo_keywords", [])

    brand_name = state.get("brand_name", "")
    brand_context = state.get("brand_context", "")
    token_limit = int(os.getenv("CONDENSER_TOKEN_LIMIT", "3000"))

    if not topic or (not ddgs_summary and not rg_summary):
        return {}

    try:
        system_msg, user_msg = condenser_prompt(
            topic=topic,
            ddgs_summary=ddgs_summary,
            rg_summary=rg_summary,
            persona=persona,
            example_post=example_post,
            primary_keyword=primary_keyword,
            secondary_keywords=secondary_keywords,
            seo_keywords=seo_keywords,
            token_limit=token_limit,
            brand_name=brand_name,
            brand_context=brand_context,
        )
        response = llm.invoke([system_msg, user_msg], config=config)
        condensed = response.content.strip()
        return {"condensed_content": condensed}
    except Exception as e:
        errors = list(state.get("errors", []))
        errors.append(f"condenser_node error: {e}")
        return {"errors": errors}


def planner_node(
    state: AgentState, config: RunnableConfig | None = None
) -> Dict[str, object]:
    """
    Node: generate structured Plan with tasks for blog post.

    Reads from AgentState:
      - topic: str
      - raw_prompt: str
      - persona: str
      - example_post: str
      - target_words_total: int
      - condensed_content: str
      - primary_keyword: str
      - secondary_keywords: List[str]
      - seo_keywords: List[str]

    Writes:
      - plan: Plan
      - current_task_id: int
      - completed_task_ids: List[int]
    """
    topic = state.get("topic", "")
    raw_prompt = state.get("raw_prompt", "")
    persona = state.get("persona", "")
    example_post = state.get("example_post", "")
    target_words_total = state.get("target_words_total", 1500)
    condensed_content = state.get("condensed_content", "")
    primary_keyword = state.get("primary_keyword", "")
    secondary_keywords = state.get("secondary_keywords", [])
    seo_keywords = state.get("seo_keywords", [])

    brand_name = state.get("brand_name", "")
    blog_kind_hint = state.get("blog_kind_hint", "")

    if not topic or not raw_prompt:
        return {}

    try:
        user_msg = planner_prompt(
            topic=topic,
            raw_prompt=raw_prompt,
            persona=persona,
            example_post=example_post,
            target_words_total=target_words_total,
            condensed_content=condensed_content,
            primary_keyword=primary_keyword,
            secondary_keywords=secondary_keywords,
            seo_keywords=seo_keywords,
            brand_name=brand_name,
            blog_kind_hint=blog_kind_hint,
        )
        structured_llm = llm.with_structured_output(Plan)
        plan = structured_llm.invoke(user_msg, config=config)

        if not plan.tasks:
            raise ValueError("Plan must have at least one task")

        current_task_id = plan.tasks[0].id
        completed_task_ids = []

        return {
            "plan": plan,
            "current_task_id": current_task_id,
            "completed_task_ids": completed_task_ids,
        }
    except Exception as e:
        errors = list(state.get("errors", []))
        errors.append(f"planner_node error: {e}")
        return {"errors": errors}


def draft_task_node(
    state: AgentState, config: RunnableConfig | None = None
) -> Dict[str, Dict[int, str]]:
    """
    Node: draft a single task/section based on current task ID.

    Reads from AgentState:
      - plan: Plan
      - current_task_id: int
      - condensed_content: str
      - ddgs_results: Dict[str, List[SearchResult]]
      - rg_results: Dict[str, List[RipgrepMatch]]
      - primary_keyword: str
      - secondary_keywords: List[str]

    Writes:
      - task_drafts[current_task_id]: str (merge semantics)
    """
    plan = state.get("plan")
    current_task_id = state.get("current_task_id")
    condensed_content = state.get("condensed_content", "")
    ddgs_results = state.get("ddgs_results", {})
    rg_results = state.get("rg_results", {})
    primary_keyword = state.get("primary_keyword", "")
    secondary_keywords = state.get("secondary_keywords", [])

    if not plan or current_task_id is None:
        return {}

    task = next((t for t in plan.tasks if t.id == current_task_id), None)
    if not task:
        errors = list(state.get("errors", []))
        errors.append(
            f"draft_task_node error: Task {current_task_id} not found in plan"
        )
        return {"errors": errors}

    try:
        system_msg, user_msg = draft_task_prompt(
            plan=plan,
            current_task_id=current_task_id,
            condensed_content=condensed_content,
            ddgs_results=ddgs_results,
            rg_results=rg_results,
            primary_keyword=primary_keyword,
            secondary_keywords=secondary_keywords,
        )
        response = llm.invoke([system_msg, user_msg], config=config)
        draft = response.content.strip()

        return {"task_drafts": {current_task_id: draft}}
    except Exception as e:
        errors = list(state.get("errors", []))
        errors.append(f"draft_task_node error: {e}")
        return {"errors": errors}


def task_revision_node(
    state: AgentState, config: RunnableConfig | None = None
) -> Dict[str, Dict[int, str]]:
    """
    Node: fact-check and revise a task draft.

    Reads from AgentState:
      - task_drafts: Dict[int, str]
      - plan: Plan
      - current_task_id: int
      - ddgs_results: Dict[str, List[SearchResult]]
      - rg_results: Dict[str, List[RipgrepMatch]]

    Writes:
      - task_drafts[current_task_id]: str (updated version)
    """
    task_drafts = state.get("task_drafts") or {}
    plan = state.get("plan")
    current_task_id = state.get("current_task_id")
    ddgs_results = state.get("ddgs_results", {})
    rg_results = state.get("rg_results", {})

    if not plan or current_task_id is None:
        return {}

    task_draft = task_drafts.get(current_task_id)
    if not task_draft:
        return {}

    try:
        system_msg, user_msg = task_revision_prompt(
            task_draft=task_draft,
            plan=plan,
            current_task_id=current_task_id,
            ddgs_results=ddgs_results,
            rg_results=rg_results,
        )
        response = llm.invoke([system_msg, user_msg], config=config)
        revised_draft = response.content.strip()

        return {"task_drafts": {current_task_id: revised_draft}}
    except Exception as e:
        errors = list(state.get("errors", []))
        errors.append(f"task_revision_node error: {e}")
        return {"errors": errors}


def advance_task_node(state: AgentState, *_args, **_kwargs) -> Dict[str, object]:
    """
    Node: advance to next task after current task is complete.

    Reads from AgentState:
      - plan: Plan
      - current_task_id: int
      - completed_task_ids: List[int]

    Writes:
      - completed_task_ids: List[int]
      - current_task_id: int | None
    """
    plan = state.get("plan")
    current_task_id = state.get("current_task_id")
    completed_task_ids = list(state.get("completed_task_ids") or [])

    if not plan or current_task_id is None:
        return {}

    completed_task_ids.append(current_task_id)

    all_task_ids = [t.id for t in plan.tasks]
    pending_task_ids = [tid for tid in all_task_ids if tid not in completed_task_ids]

    next_task_id = pending_task_ids[0] if pending_task_ids else None

    return {
        "completed_task_ids": completed_task_ids,
        "current_task_id": next_task_id,
    }


def stitcher_node(
    state: AgentState, config: RunnableConfig | None = None
) -> Dict[str, str]:
    """
    Node: stitch all task drafts into coherent article with transitions.

    Reads from AgentState:
      - plan: Plan
      - task_drafts: Dict[int, str]
      - persona: str
      - example_post: str
      - target_words_total: int

    Writes:
      - stitched_draft: str
    """
    plan = state.get("plan")
    task_drafts = state.get("task_drafts") or {}
    persona = state.get("persona", "")
    example_post = state.get("example_post", "")
    target_words_total = state.get("target_words_total", 1500)

    if not plan or not task_drafts:
        return {}

    try:
        system_msg, user_msg = stitcher_prompt(
            plan=plan,
            task_drafts=task_drafts,
            persona=persona,
            example_post=example_post,
            target_words_total=target_words_total,
        )
        response = llm.invoke([system_msg, user_msg], config=config)
        stitched = response.content.strip()
        return {"stitched_draft": stitched}
    except Exception as e:
        errors = list(state.get("errors", []))
        errors.append(f"stitcher_node error: {e}")
        return {"errors": errors}


def styler_node(
    state: AgentState, config: RunnableConfig | None = None
) -> Dict[str, str]:
    """
    Node: apply persona/style to match example post.

    Reads from AgentState:
      - stitched_draft: str
      - persona: str
      - example_post: str
      - primary_keyword: str
      - secondary_keyword: str

    Writes:
      - styled_draft: str
    """
    stitched_draft = state.get("stitched_draft")
    persona = state.get("persona", "")
    example_post = state.get("example_post", "")
    plan = state.get("plan")
    primary_keyword = state.get("primary_keyword", "")
    secondary_keywords = state.get("secondary_keywords", [])

    if not stitched_draft:
        return {}

    if not example_post:
        return {"styled_draft": stitched_draft}

    tone_profile = plan.tone_profile if plan else ""

    try:
        system_msg, user_msg = styler_prompt(
            stitched_draft=stitched_draft,
            persona=persona,
            example_post=example_post,
            tone_profile=tone_profile,
            primary_keyword=primary_keyword,
            secondary_keywords=secondary_keywords,
        )
        response = llm.invoke([system_msg, user_msg], config=config)
        styled = response.content.strip()
        return {"styled_draft": styled}
    except Exception as e:
        errors = list(state.get("errors", []))
        errors.append(f"styler_node error: {e}")
        return {"errors": errors}


class WPFormatOutput(BaseModel):
    subtitle: str = Field(..., description="One-line subtitle / deck that complements the title (max 120 chars)")
    excerpt: str = Field(..., description="2-3 sentence summary for WP excerpt (max 300 chars)")
    categories: List[str] = Field(
        ..., min_length=1, max_length=3,
        description="1-3 WordPress categories (broad topic buckets, e.g. 'Education', 'School Guide', 'Parenting')"
    )
    tags: List[str] = Field(
        ..., min_length=3, max_length=10,
        description="3-10 WordPress tags (specific, searchable terms relevant to the article)"
    )


def wp_format_node(
    state: AgentState, config: RunnableConfig | None = None
) -> Dict[str, object]:
    """
    Node: package styled article into WP-ready fields.

    Uses LLM to generate subtitle, excerpt, categories, and tags
    informed by the article content and KB research summary.

    Reads: styled_draft, plan, seo_meta_title, seo_meta_description,
           rg_result_summary, primary_keyword, secondary_keywords, seo_keywords
    Writes: wp_title, wp_subtitle, wp_body, wp_excerpt, wp_tags, wp_categories
    """
    styled_draft = state.get("styled_draft", "")
    plan = state.get("plan")
    seo_title = state.get("seo_meta_title", "")
    seo_desc = state.get("seo_meta_description", "")
    rg_summary = state.get("rg_result_summary", "")
    primary_keyword = state.get("primary_keyword", "")
    secondary_keywords = state.get("secondary_keywords", [])
    seo_keywords = state.get("seo_keywords", [])

    if not styled_draft:
        return {}

    wp_title = seo_title or (plan.blog_title if plan else "")

    keywords_context = ""
    if primary_keyword:
        keywords_context += f"Primary keyword: {primary_keyword}\n"
    if secondary_keywords:
        keywords_context += f"Secondary keywords: {', '.join(secondary_keywords)}\n"
    if seo_keywords:
        keywords_context += f"SEO keywords: {', '.join(seo_keywords)}\n"

    kb_context = ""
    if rg_summary:
        kb_context = f"\nKnowledge base summary (use this to inform tag/category choices):\n{rg_summary[:1500]}\n"

    try:
        prompt_msg = (
            f"Article title: {wp_title}\n\n"
            f"Article body (first 1500 chars):\n{styled_draft[:1500]}\n\n"
            f"{keywords_context}"
            f"{kb_context}\n"
            "Tasks:\n"
            "1. Write a one-line subtitle (complementing the title, max 120 chars)\n"
            "2. Write a 2-3 sentence excerpt summary for WordPress (max 300 chars)\n"
            "3. Choose 1-3 WordPress CATEGORIES — these are broad topic buckets.\n"
            "   Good categories: 'Education', 'School Guide', 'Parenting', 'Dubai Schools', 'Curriculum'\n"
            "   Bad categories: 'best british schools' (too specific — that's a tag)\n"
            "4. Choose 3-10 WordPress TAGS — these are specific, searchable terms.\n"
            "   Mix of: brand terms, location terms, curriculum terms, and topic-specific phrases.\n"
            "   Draw from the keywords and KB summary above.\n"
        )
        structured_llm = llm.with_structured_output(WPFormatOutput)
        result = structured_llm.invoke(prompt_msg, config=config)
        return {
            "wp_title": wp_title,
            "wp_subtitle": result.subtitle,
            "wp_body": styled_draft,
            "wp_excerpt": result.excerpt if not seo_desc else seo_desc,
            "wp_tags": result.tags,
            "wp_categories": result.categories,
        }
    except Exception as e:
        errors = list(state.get("errors", []))
        errors.append(f"wp_format_node error: {e}")
        return {
            "wp_title": wp_title,
            "wp_subtitle": "",
            "wp_body": styled_draft,
            "wp_excerpt": seo_desc,
            "wp_tags": [primary_keyword] + secondary_keywords[:5] if primary_keyword else [],
            "wp_categories": [],
            "errors": errors,
        }


class SEOOutput(BaseModel):
    primary_keyword: str = Field(..., description="The single best primary keyword")
    secondary_keywords: List[str] = Field(
        ..., min_length=3, max_length=7, description="3-7 relevant secondary keywords"
    )
    seo_keywords: List[str] = Field(
        ..., min_length=5, max_length=10, description="5-10 long-tail SEO keywords"
    )
    seo_meta_title: str = Field(..., description="SEO-optimized title (50-60 chars)")
    seo_meta_description: str = Field(
        ..., description="Compelling description (150-160 chars)"
    )
    seo_slug: str = Field(
        ..., description="URL-friendly slug (lowercase, hyphens, no stop words)"
    )


def seo_node(
    state: AgentState, config: RunnableConfig | None = None
) -> Dict[str, object]:
    """
    Node: generate SEO metadata from styled article.

    Reads from AgentState:
      - styled_draft: str
      - topic: str
      - primary_keyword: str
      - secondary_keywords: List[str]
      - seo_keywords: List[str]
      - plan: Plan

    Writes:
      - primary_keyword: str
      - secondary_keywords: List[str]
      - seo_keywords: List[str]
      - seo_meta_title: str
      - seo_meta_description: str
      - seo_slug: str
    """
    styled_draft = state.get("styled_draft")
    topic = state.get("topic", "")
    primary_keyword = state.get("primary_keyword", "")
    secondary_keywords = state.get("secondary_keywords", [])
    seo_keywords = state.get("seo_keywords", [])
    plan = state.get("plan")

    if not styled_draft or not topic:
        return {}

    try:
        user_msg = seo_prompt(
            styled_draft=styled_draft,
            topic=topic,
            primary_keyword=primary_keyword,
            secondary_keywords=secondary_keywords,
            seo_keywords=seo_keywords,
            plan=plan,
        )
        structured_llm = llm.with_structured_output(SEOOutput)
        seo_output = structured_llm.invoke(user_msg, config=config)

        return {
            "primary_keyword": seo_output.primary_keyword,
            "secondary_keywords": seo_output.secondary_keywords,
            "seo_keywords": seo_output.seo_keywords,
            "seo_meta_title": seo_output.seo_meta_title,
            "seo_meta_description": seo_output.seo_meta_description,
            "seo_slug": seo_output.seo_slug,
        }
    except Exception as e:
        errors = list(state.get("errors", []))
        errors.append(f"seo_node error: {e}")
        return {"errors": errors}
