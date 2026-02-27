# Shared agent state definitins in LangGraph
# 24/02/2026 01:06PM Nikhil Kapila

from __future__ import annotations
from typing import Type, TypedDict, Literal, Any, Dict, List, Optional
from typing_extensions import NotRequired

from pydantic import BaseModel, Field

class Source(TypedDict):
    id: int
    url: str
    title: str

class SearchResult(TypedDict):
    url: NotRequired[str]
    title: NotRequired[str]
    content: str

class RipgrepMatch(TypedDict):
    file_path:str
    line:int
    line_text:str
    context_before:List[str]
    context_after: List[str]

class Task(BaseModel):
    id: int
    title:str
    goal:str=Field(
        ...,
        description="One sentence describing what the reader should do or understand."
    )
    bullets: List[str]=Field(...,min_length=3, max_length=6)
    target_words:int = Field(..., description="Target words (120-550)")

    tags:List[str]=Field(default_factory=list)
    requires_research:bool=False
    requires_citations:bool=False
    requires_code:bool=False

class Plan(BaseModel):
    blog_title:str
    audience:str
    tone:str
    blog_kind: Literal[
    "concept_explainer",  # The "What": Defining a single idea/topic.
    "procedural_guide",   # The "How": Step-by-step instructions or tutorials.
    "analytical_compare", # The "Versus": Evaluating two or more options.
    "digest_roundup",     # The "Latest": News, links, or weekly highlights.
    "structural_deepdive",# The "Architecture": How a complex system/org is built.
    "narrative_log",      # The "Story": Case studies, project retrospectives, or field trips.
    "inquiry_response",   # The "Q&A": FAQs, interviews, or mailbags.
    "resource_curation",  # The "Tools": Lists of books, libraries, or supplies.
    ] = "concept_explainer"
    depth:Literal["beginner", "intermediate", "expert"]="intermediate"
    tone_profile:str
    constraints:List[str]=Field(default_factory=list)
    tasks:List[Task]

class TopicSuggestion(BaseModel):
    topic: str
    prompt: str = Field(..., description="A detailed writing prompt for the article")
    persona: str = Field(default="informative blogger")
    primary_keyword: str = Field(default="")
    target_words: int = Field(default=1500)
    blog_kind_hint: str = Field(default="concept_explainer")
    rationale: str = Field(..., description="Why this topic fills a gap in existing content")

class TopicSuggestions(BaseModel):
    suggestions: List[TopicSuggestion]


class AgentState(TypedDict, total=False):
    # input
    topic: str
    raw_prompt:str
    persona:str
    example_post:str
    target_words_total:int
    blog_kind_hint:str
    brand_name:str
    brand_context:str

    # ddgs result
    ddgs_queries: List[str]
    ddgs_num_results: int
    ddgs_results: Dict[str, List[SearchResult]]
    ddgs_result_summary:str # from the generative step
    source_registry: List[Source]

    # ripgrep queries
    kb_root:str
    rg_queries: List[str]
    rg_results: Dict[str, List[RipgrepMatch]]
    rg_result_summary: str

    # condenser node (web+KB search)
    condensed_content:str

    # planning agent
    plan: Plan
    current_task_id:int
    completed_task_ids: List[int]
    task_drafts: Dict[int, str]
    
    # stitching agent
    stitched_draft:str

    # styler node
    styled_draft:str

    # seo segment
    primary_keyword:str
    secondary_keywords:List[str]
    seo_keywords:List[str]
    seo_meta_title:str
    seo_meta_description:str
    seo_slug:str

    # wp output
    wp_title:str
    wp_subtitle:str
    wp_body:str
    wp_excerpt:str
    wp_tags:List[str]
    wp_categories:List[str]
    wp_post_id:int
    wp_post_url:str

    # debug
    errors:List[str]
    debug_logs:List[str]
