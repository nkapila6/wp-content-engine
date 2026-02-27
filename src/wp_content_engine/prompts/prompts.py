# Prompts for different steps
# 24/02/2026 02:10PM Nikhil Kapila

from __future__ import annotations

from typing import List

from langchain_core.messages import SystemMessage, HumanMessage

from wp_content_engine.state.state import Plan


def ddgs_summary_prompt(
    topic: str, ddgs_results: dict, target_words_total: int
) -> tuple[SystemMessage, HumanMessage]:
    """
    Generate prompt for summarizing DDGS web search results.

    Args:
        topic: The blog post topic
        ddgs_results: Dict of query -> List[SearchResult]
        target_words_total: Target word count for final article

    Returns:
        Tuple of (SystemMessage, HumanMessage)
    """
    results_str = ""
    for query, results in ddgs_results.items():
        results_str += f"\nQuery: {query}\n"
        for i, result in enumerate(results, 1):
            url = result.get("url", "")
            content = result.get("content", "")[:800]
            results_str += f"  {i}. {url}\n     {content}...\n"

    system_msg = SystemMessage(
        content=(
            "You are a research analyst summarizing web search results for a blog post.\n"
            "Your goal is to create a concise, focused summary (300-500 words) that captures:\n"
            "- Key information and insights relevant to the topic\n"
            "- Main themes and perspectives found across sources\n"
            "- Important facts, statistics, or expert opinions\n"
            "- Any gaps or areas needing more research\n\n"
            "Focus on high-signal information that would help someone write a comprehensive article. "
            "Organize by theme when helpful."
        )
    )

    user_msg = HumanMessage(
        content=f"""Topic: {topic}
Target article length: {target_words_total} words

Web Search Results:
{results_str}

Task:
Provide a focused summary of these web sources. Highlight:
1. Key insights directly relevant to the topic
2. Main themes across sources
3. Important facts or statistics
4. Different perspectives or approaches mentioned

Keep your response between 300-500 words. Be concise but comprehensive."""
    )

    return system_msg, user_msg


def rg_summary_prompt(
    topic: str, rg_results: dict
) -> tuple[SystemMessage, HumanMessage]:
    """
    Generate prompt for summarizing ripgrep KB search results.

    Args:
        topic: The blog post topic
        rg_results: Dict of query -> List[RipgrepMatch]

    Returns:
        Tuple of (SystemMessage, HumanMessage)
    """
    matches_str = ""
    for query, matches in rg_results.items():
        matches_str += f"\nQuery: {query}\n"
        matches_str += f"Found {len(matches)} matches\n"

        for i, match in enumerate(matches[:10], 1):
            file = match.get("file_path", "")
            line = match.get("line", 0)
            text = match.get("line_text", "")[:200]
            matches_str += f'  {i}. {file}:{line}\n     "{text}"\n'

        if len(matches) > 10:
            matches_str += f"  ... and {len(matches) - 10} more matches\n"

    system_msg = SystemMessage(
        content=(
            "You are analyzing knowledge base search results from a local repository.\n"
            "The results are from ripgrep searches over text/markdown files.\n"
            "Your goal is to synthesize these matches into a coherent summary (300-500 words):\n"
            "- Group related matches by theme or file\n"
            "- Identify key concepts, patterns, or documentation sections\n"
            "- Highlight code examples, configuration details, or technical specifics\n"
            "- Note any conflicting information or gaps\n\n"
            "Prioritize matches that are most relevant to the topic. Ignore noise."
        )
    )

    user_msg = HumanMessage(
        content=f"""Topic: {topic}

Knowledge Base Search Results:
{matches_str}

Task:
Provide a synthesis of these KB matches. Organize by:
1. Key themes or topics found
2. Important files or sections referenced
3. Technical details, code, or configuration relevant to topic
4. Any patterns or best practices mentioned

Keep your response between 300-500 words."""
    )

    return system_msg, user_msg


def condenser_prompt(
    topic: str,
    ddgs_summary: str,
    rg_summary: str,
    persona: str,
    example_post: str,
    primary_keyword: str,
    secondary_keywords: list[str],
    seo_keywords: list[str],
    token_limit: int,
    brand_name: str = "",
    brand_context: str = "",
) -> tuple[SystemMessage, HumanMessage]:
    """
    Generate prompt for condensing summaries into budgeted context.

    Args:
        topic: The blog post topic
        ddgs_summary: Summary of web search results
        rg_summary: Summary of KB search results
        persona: Writer persona/description
        example_post: Example post to match style
        primary_keyword: Primary SEO keyword
        secondary_keywords: List of secondary keywords
        seo_keywords: List of SEO keywords
        token_limit: Token limit for output

    Returns:
        Tuple of (SystemMessage, HumanMessage)
    """
    persona_str = persona if persona else "Professional blog writer"
    example_str = example_post[:500] if example_post else "Not provided"

    keywords_str = ""
    if primary_keyword:
        keywords_str += f"Primary: {primary_keyword}\n"
    if secondary_keywords:
        keywords_str += f"Secondary: {', '.join(secondary_keywords)}\n"
    if seo_keywords:
        keywords_str += f"SEO: {', '.join(seo_keywords)}\n"

    system_msg = SystemMessage(
        content=(
            "You are preparing a context document for AI writers planning and drafting a blog post.\n"
            "Your task is to condense research summaries into a single, budgeted context string.\n"
            f"This context will be used by multiple AI writers. Stay within {token_limit} tokens.\n\n"
            "Include:\n"
            "- Core facts and information about the topic\n"
            "- Key insights from web research and knowledge base\n"
            "- Target audience considerations from persona\n"
            "- SEO keywords to naturally incorporate\n"
            "- Tone/style cues from example post\n\n"
            "Organize logically: background → key points → audience → tone → SEO notes.\n"
            "Be comprehensive but efficient. Every word must earn its place."
        )
    )

    brand_str = ""
    if brand_name:
        brand_str += f"\nBrand Name: {brand_name}\n"
    if brand_context:
        preview = brand_context[:2000]
        brand_str += f"\nBrand / Company Context (from knowledge base):\n{preview}\n"

    user_msg = HumanMessage(
        content=f"""Topic: {topic}
{brand_str}
Writer Persona:
{persona_str}

Example Post (first 500 chars):
{example_str}

SEO Keywords:
{keywords_str if keywords_str else "None specified"}

Web Research Summary:
{ddgs_summary}

Knowledge Base Summary:
{rg_summary}

Task:
Create a condensed context document (max {token_limit} tokens) that:
1. Synthesizes the most important information from both summaries
2. Provides enough context for writers to understand the topic deeply
3. Highlights audience needs and expectations from persona
4. Notes SEO keywords to incorporate naturally
5. Captures the voice/style from the example post
6. {"Weaves in brand-specific details about " + brand_name + " where relevant — this content is FOR this brand" if brand_name else "No specific brand context"}

Organize into clear sections. Be concise but thorough. This context will guide both planning and writing."""
    )

    return system_msg, user_msg


def planner_prompt(
    topic: str,
    raw_prompt: str,
    persona: str,
    example_post: str,
    target_words_total: int,
    condensed_content: str,
    primary_keyword: str,
    secondary_keywords: list[str],
    seo_keywords: list[str],
    brand_name: str = "",
    blog_kind_hint: str = "",
) -> tuple[SystemMessage, HumanMessage]:
    """
    Generate prompt for creating a structured Plan with tasks.

    Args:
        topic: The blog post topic
        raw_prompt: Original user prompt
        persona: Writer persona
        example_post: Example post
        target_words_total: Target word count
        condensed_content: Condensed research context
        primary_keyword: Primary SEO keyword
        secondary_keywords: List of secondary keywords
        seo_keywords: List of SEO keywords

    Returns:
        Tuple of (SystemMessage, HumanMessage)
    """
    persona_str = persona if persona else "Professional blog writer"
    example_str = example_post[:400] if example_post else "Not provided"

    keywords_str = ""
    if primary_keyword:
        keywords_str += f"Primary: {primary_keyword}\n"
    if secondary_keywords:
        keywords_str += f"Secondary: {', '.join(secondary_keywords)}\n"
    if seo_keywords:
        keywords_str += f"SEO: {', '.join(seo_keywords)}\n"

    system_msg = SystemMessage(
        content=(
            "You are a blog post architect creating a detailed writing plan.\n"
            "You will output a structured Plan that guides AI writers through drafting each section.\n\n"
            "The Plan must include:\n"
            "- blog_title: Compelling, SEO-friendly title\n"
            "- audience: Who this post is for (be specific)\n"
            "- tone: Overall tone (e.g., 'conversational', 'technical', 'inspiring')\n"
            "- blog_kind: Type of post from: concept_explainer, procedural_guide, analytical_compare, "
            "digest_roundup, structural_deepdive, narrative_log, inquiry_response, resource_curation\n"
            "- depth: beginner, intermediate, or expert\n"
            "- tone_profile: Detailed description of the desired voice/style\n"
            "- constraints: Any limitations or requirements (e.g., avoid jargon, include examples)\n"
            "- tasks: List of 3-7 tasks, each with:\n"
            "  * id: Sequential integer (1, 2, 3...)\n"
            "  * title: Section title\n"
            "  * goal: One sentence describing what reader should understand/do\n"
            "  * bullets: 3-6 specific points to cover\n"
            "  * target_words: Word count (120-550) - sum to target_words_total\n"
            "  * tags: Optional tags (e.g., ['intro', 'technical', 'examples'])\n"
            "  * requires_research, requires_citations, requires_code: Boolean flags\n\n"
            "CRITICAL: The sum of all task.target_words must equal target_words_total.\n"
            "Each task should be independently writable by a different writer."
        )
    )

    brand_instruction = ""
    if brand_name:
        brand_instruction = (
            f"\n10. This article is written FOR {brand_name}. "
            f"Ensure the plan references {brand_name} where appropriate — "
            f"the article should position {brand_name} favourably without being overtly promotional.\n"
        )

    user_msg = HumanMessage(
        content=f"""Topic: {topic}

Original Prompt:
{raw_prompt}

Writer Persona:
{persona_str}

Example Post (first 400 chars):
{example_str}

Target Word Count: {target_words_total} words

SEO Keywords:
{keywords_str if keywords_str else "None specified"}

Research Context:
{condensed_content}

Task:
Create a comprehensive Plan for this blog post.

1. Choose blog_kind that best matches the topic and prompt{f' — STRONGLY PREFER: {blog_kind_hint}' if blog_kind_hint else ''}
2. Define audience (who are they? what do they need?)
3. Set appropriate depth level
4. Create 3-7 tasks that collectively cover the topic
5. Each task must have clear goal, specific bullets, and target words
6. Target words per task: 120-550 (flexible based on importance)
7. Sum of all task.target_words MUST equal {target_words_total}
8. Set flags appropriately (requires_research, requires_citations, requires_code)
9. Include any constraints or style considerations in tone_profile
{brand_instruction}
Return ONLY a valid JSON Plan object."""
    )

    return system_msg, user_msg


def draft_task_prompt(
    plan: Plan,
    current_task_id: int,
    condensed_content: str,
    ddgs_results: dict,
    rg_results: dict,
    primary_keyword: str,
    secondary_keywords: list[str],
) -> tuple[SystemMessage, HumanMessage]:
    """
    Generate prompt for drafting a single task/section.

    Args:
        plan: The full Plan object
        current_task_id: ID of task to draft
        condensed_content: Condensed research context
        ddgs_results: Web search results
        rg_results: KB search results
        primary_keyword: Primary SEO keyword
        secondary_keywords: List of secondary keywords

    Returns:
        Tuple of (SystemMessage, HumanMessage)
    """
    task = next((t for t in plan.tasks if t.id == current_task_id), None)
    if not task:
        raise ValueError(f"Task {current_task_id} not found in plan")

    keywords_str = ""
    if primary_keyword:
        keywords_str += f"Primary: {primary_keyword}\n"
    if secondary_keywords:
        keywords_str += f"Secondary: {', '.join(secondary_keywords)}\n"

    system_msg = SystemMessage(
        content=(
            "You are a blog post section writer. You will write ONE section of a larger article.\n"
            "Your section must:\n"
            "- Fully address the task goal\n"
            "- Cover all bullet points naturally\n"
            "- Stay within target word count\n"
            "- Match the overall tone and audience\n"
            "- Integrate research context where relevant\n"
            "- Incorporate SEO keywords naturally (don't force it)\n\n"
            "Write in a way that flows well with other sections (you don't write the transitions).\n"
            "Use examples, data, or quotes from research to support points.\n"
            "Be specific and actionable. Avoid fluff."
        )
    )

    user_msg = HumanMessage(
        content=f"""Blog Title: {plan.blog_title}

Overall Audience: {plan.audience}
Overall Tone: {plan.tone}
Depth Level: {plan.depth}
Blog Type: {plan.blog_kind}

YOUR TASK:
Task ID: {task.id}
Task Title: {task.title}
Goal: {task.goal}
Target Words: {task.target_words}
Tags: {", ".join(task.tags) if task.tags else "None"}

Bullets to Cover:
{chr(10).join(f"- {b}" for b in task.bullets)}

Requirements:
- Requires research citations: {task.requires_citations}
- Requires code examples: {task.requires_code}
- Requires additional research: {task.requires_research}

Context for this article:
{condensed_content}

SEO Keywords:
{keywords_str if keywords_str else "None specified"}

Task:
Write this section ({task.target_words} words) covering all the bullets above.
Make it engaging, informative, and aligned with the overall article goals.
If citations are required, use [Source X] notation referencing the research context.
If code examples are needed, include clear, well-commented code."""
    )

    return system_msg, user_msg


def task_revision_prompt(
    task_draft: str,
    plan: Plan,
    current_task_id: int,
    ddgs_results: dict,
    rg_results: dict,
) -> tuple[SystemMessage, HumanMessage]:
    """
    Generate prompt for fact-checking and revising a task draft.

    Args:
        task_draft: Current draft of the task
        plan: The full Plan object
        current_task_id: ID of task being revised
        ddgs_results: Web search results
        rg_results: KB search results

    Returns:
        Tuple of (SystemMessage, HumanMessage)
    """
    task = next((t for t in plan.tasks if t.id == current_task_id), None)
    if not task:
        raise ValueError(f"Task {current_task_id} not found in plan")

    sources_str = ""

    web_sources = []
    for query, results in ddgs_results.items():
        for i, result in enumerate(results[:3], 1):
            url = result.get("url", "")
            content = result.get("content", "")[:300]
            web_sources.append(f"Web [{len(web_sources) + 1}]: {url}\n{content}...")

    kb_sources = []
    for query, matches in rg_results.items():
        for i, match in enumerate(matches[:3], 1):
            file = match.get("file_path", "")
            text = match.get("line_text", "")[:200]
            kb_sources.append(f'KB [{len(kb_sources) + 1}]: {file}\n"{text}"')

    if web_sources:
        sources_str += "\n=== Web Sources ===\n" + "\n\n".join(web_sources)
    if kb_sources:
        sources_str += "\n\n=== Knowledge Base Sources ===\n" + "\n\n".join(kb_sources)

    system_msg = SystemMessage(
        content=(
            "You are a fact-checker and editor reviewing a blog post section.\n"
            "Your role is to:\n"
            "- Verify key factual claims against the provided sources\n"
            "- Add citations where appropriate using [Source X] notation\n"
            "- Fix any obvious errors or inconsistencies\n"
            "- Improve clarity and flow without changing the overall message\n"
            "- Ensure the draft stays within the target word count\n\n"
            "Don't rewrite the entire section unless necessary. Focus on accuracy and quality."
        )
    )

    user_msg = HumanMessage(
        content=f"""Task: {task.title}
Goal: {task.goal}
Target Words: {task.target_words}

CURRENT DRAFT:
{task_draft}

AVAILABLE SOURCES:
{sources_str}

Task:
Review and improve this draft:
1. Verify key facts against sources
2. Add citations where facts are sourced
3. Fix any errors or inconsistencies
4. Improve clarity
5. Stay close to target word count ({task.target_words})

Return the improved draft."""
    )

    return system_msg, user_msg


def stitcher_prompt(
    plan: Plan,
    task_drafts: dict[int, str],
    persona: str,
    example_post: str,
    target_words_total: int,
) -> tuple[SystemMessage, HumanMessage]:
    """
    Generate prompt for stitching task drafts into coherent article.

    Args:
        plan: The full Plan object
        task_drafts: Dict of task_id -> draft content
        persona: Writer persona
        example_post: Example post
        target_words_total: Target word count

    Returns:
        Tuple of (SystemMessage, HumanMessage)
    """
    persona_str = persona if persona else "Professional blog writer"
    example_str = example_post[:300] if example_post else "Not provided"

    drafts_str = ""
    for task in plan.tasks:
        draft = task_drafts.get(task.id, "")
        drafts_str += f"\n{'=' * 60}\n"
        drafts_str += f"TASK {task.id}: {task.title}\n"
        drafts_str += f"{'=' * 60}\n"
        drafts_str += f"{draft}\n"

    system_msg = SystemMessage(
        content=(
            "You are assembling a complete blog post from individually written sections.\n"
            "Your job is to:\n"
            "- Combine all sections into one coherent article\n"
            "- Write smooth transitions between sections\n"
            "- Ensure the article flows logically from start to finish\n"
            "- Add an engaging introduction that hooks the reader\n"
            "- Write a compelling conclusion that reinforces key points\n"
            "- Match the persona and style from the example\n"
            "- Target the total word count closely\n\n"
            "Don't rewrite the sections themselves. Focus on integration and flow."
        )
    )

    user_msg = HumanMessage(
        content=f"""Blog Title: {plan.blog_title}

Writer Persona:
{persona_str}

Example Post Style (first 300 chars):
{example_str}

Target Total Words: {target_words_total}
Blog Type: {plan.blog_kind}
Tone: {plan.tone}

INDIVIDUAL SECTIONS:
{drafts_str}

Task:
Assemble this into a complete blog post:

1. Write an engaging introduction (100-200 words) that:
   - Hooks the reader immediately
   - States what they'll learn
   - Sets the right tone

2. Combine all sections in order, adding smooth transitions

3. Write a compelling conclusion (100-150 words) that:
   - Summarizes key takeaways
   - Leaves the reader satisfied with a strong closing thought
   - Do NOT add a generic call-to-action like "contact us", "reach out", "ready to explore", etc.

4. Review the full article and ensure it flows perfectly

Return the complete article with introduction, sections with transitions, and conclusion."""
    )

    return system_msg, user_msg


def styler_prompt(
    stitched_draft: str,
    persona: str,
    example_post: str,
    tone_profile: str,
    primary_keyword: str,
    secondary_keywords: list[str],
) -> tuple[SystemMessage, HumanMessage]:
    """
    Generate prompt for styling article to match persona/example.

    Args:
        stitched_draft: The stitched article draft
        persona: Writer persona description
        example_post: Example post to match style
        tone_profile: Detailed tone profile from Plan
        primary_keyword: Primary SEO keyword
        secondary_keywords: List of secondary keywords

    Returns:
        Tuple of (SystemMessage, HumanMessage)
    """
    persona_str = persona if persona else "Professional blog writer"
    tone_str = tone_profile if tone_profile else "Professional yet engaging"

    keywords_str = ""
    if primary_keyword:
        keywords_str += f"Primary: {primary_keyword}\n"
    if secondary_keywords:
        keywords_str += f"Secondary: {', '.join(secondary_keywords)}\n"

    system_msg = SystemMessage(
        content=(
            "You are a style editor finalizing a blog post's voice and tone.\n"
            "Your task is to:\n"
            "- Match the exact voice and style from the example post\n"
            "- Apply the tone_profile precisely\n"
            "- Ensure SEO keywords appear naturally (don't overdo it)\n"
            "- Polish sentences for flow and readability\n"
            "- Fix any awkward phrasing or inconsistencies\n"
            "- Maintain the core content and structure\n\n"
            "The goal is to make it sound like it was written by the persona, not a machine.\n"
            "IMPORTANT: Do NOT append a keyword list, keyword block, or any meta-information at the end of the article. "
            "The article body must end naturally with the conclusion paragraph."
        )
    )

    user_msg = HumanMessage(
        content=f"""TARGET STYLE:
Persona: {persona_str}

Tone Profile:
{tone_str}

Example Post (analyze this style):
{example_post}

CURRENT DRAFT:
{stitched_draft}

SEO Keywords:
{keywords_str if keywords_str else "None specified"}

Task:
Rewrite this draft to perfectly match the persona and example style:

1. Analyze the example post's:
   - Sentence structure (short/long/mixed)
   - Word choice (formal/casual/technical)
   - Voice (first/third person, active/passive)
   - Use of humor, metaphors, analogies
   - Paragraph structure
   - Opening/closing patterns

2. Apply those exact patterns to the draft

3. Incorporate SEO keywords naturally where they fit

4. Polish for flow and readability

IMPORTANT: Do NOT append a keyword list, keyword block, or any meta-information at the end of the article.
Do NOT add a generic call-to-action like "contact us", "reach out", "ready to explore", "get in touch", etc.

Return the fully styled article."""
    )

    return system_msg, user_msg


def seo_prompt(
    styled_draft: str,
    topic: str,
    primary_keyword: str,
    secondary_keywords: list[str],
    seo_keywords: list[str],
    plan: Plan,
) -> str:
    """
    Generate prompt for generating SEO metadata.

    Args:
        styled_draft: The styled final article
        topic: The blog post topic
        primary_keyword: Current primary keyword (may be refined)
        secondary_keywords: Current secondary keywords
        seo_keywords: Current SEO keywords
        plan: The full Plan object

    Returns:
        User message as a string
    """
    draft_preview = styled_draft[:2000] if len(styled_draft) > 2000 else styled_draft

    existing_keywords = ""
    if primary_keyword:
        existing_keywords += f"Primary: {primary_keyword}\n"
    if secondary_keywords:
        existing_keywords += f"Secondary: {', '.join(secondary_keywords)}\n"
    if seo_keywords:
        existing_keywords += f"SEO: {', '.join(seo_keywords)}\n"

    user_msg = f"""Topic: {topic}

Blog Title: {plan.blog_title}
Target Audience: {plan.audience}
Blog Type: {plan.blog_kind}

Existing Keywords (refine these if better options exist):
{existing_keywords if existing_keywords else "None specified"}

ARTICLE (first 2000 chars):
{draft_preview}

Task:
Generate comprehensive SEO metadata for this article:

1. Analyze the article content for natural keyword usage
2. Refine the primary keyword if a better, more specific option exists
3. Select secondary keywords that complement the primary
4. Generate long-tail SEO keywords (phrases people actually search)
5. Write an irresistible meta title (50-60 chars) with primary keyword
6. Write a compelling meta description (150-160 chars) that drives clicks
7. Create an SEO-friendly URL slug"""

    return user_msg
