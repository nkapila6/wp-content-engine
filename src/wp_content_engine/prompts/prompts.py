# Prompts for different steps
# 24/02/2026 02:10PM Nikhil Kapila

from __future__ import annotations

from typing import List

from langchain_core.messages import SystemMessage, HumanMessage

from wp_content_engine.state.state import Plan, TopicSuggestion


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
    persona_str = persona if persona else "A thoughtful writer who shares personal insights and speaks directly to the reader"
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
            "You are preparing a context brief for writers planning and drafting a blog post.\n"
            "Condense research summaries into a single, budgeted context string.\n"
            f"This context will guide multiple writers. Stay within {token_limit} tokens.\n\n"
            "Include:\n"
            "- Core facts and insights about the topic\n"
            "- Interesting angles, surprising findings, or strong opinions from sources\n"
            "- Concrete stories, examples, or data points that a writer could reference\n"
            "- Target audience considerations from persona\n"
            "- SEO keywords to naturally incorporate\n"
            "- VOICE NOTES: a short paragraph reminding writers of the desired voice/persona\n"
            "  and any tone cues from the example post. Writers should sound like a real person,\n"
            "  not an information dispenser.\n\n"
            "Organize: background → interesting angles → concrete details → audience → voice notes → SEO.\n"
            "Prioritize material that gives writers something vivid to say, not just facts to recite."
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
Create a condensed context brief (max {token_limit} tokens) that:
1. Synthesizes the most important information from both summaries
2. Highlights the most interesting, surprising, or opinion-worthy angles
3. Includes 2-3 concrete details (stats, stories, examples) writers can weave in
4. Notes audience needs and expectations from persona
5. Includes a VOICE NOTES section reminding writers to use a personal, human tone
   (first-person or warm third-person — not encyclopedic)
6. Notes SEO keywords to incorporate naturally
7. {"Weaves in brand-specific details about " + brand_name + " where relevant — this content is FOR this brand" if brand_name else "No specific brand context"}

Organize into clear sections. Prioritize material that helps writers sound like real people
sharing what they know, not textbooks summarizing information."""
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
    persona_str = persona if persona else "A thoughtful writer who shares personal insights and speaks directly to the reader"
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
            "You will output a structured Plan that guides writers through drafting each section.\n\n"
            "IMPORTANT — VOICE DIRECTION:\n"
            "The final article must read like a person sharing their thoughts and experiences,\n"
            "NOT like an informational note or textbook entry. Default to a warm, first-person or\n"
            "relatable third-person narrative unless the persona explicitly demands otherwise.\n"
            "Your tone_profile should describe a specific human voice (e.g., 'a curious educator\n"
            "who mixes research with personal classroom stories') — never just 'professional' or 'formal'.\n\n"
            "The Plan must include:\n"
            "- blog_title: Compelling, SEO-friendly title (avoid generic 'Ultimate Guide' patterns)\n"
            "- audience: Who this post is for (be specific)\n"
            "- tone: Overall tone (e.g., 'conversational', 'reflective', 'witty-but-informed')\n"
            "- blog_kind: Type of post from: concept_explainer, procedural_guide, analytical_compare, "
            "digest_roundup, structural_deepdive, narrative_log, inquiry_response, resource_curation\n"
            "- depth: beginner, intermediate, or expert\n"
            "- tone_profile: A RICH description of the desired voice — describe the writer as a character.\n"
            "  Include: their perspective, how they open paragraphs, whether they use 'I' or 'we',\n"
            "  how they handle complexity (analogies? stories? humour?), and what they would NEVER do.\n"
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
            "You are a skilled writer crafting ONE section of a longer blog post.\n"
            "Write like a real person sharing their perspective — not like a textbook or encyclopedia.\n\n"
            "VOICE RULES:\n"
            "- Use first-person ('I', 'we', 'my') or a warm, relatable third-person voice\n"
            "- Share opinions, reflections, or light anecdotes — don't just state facts\n"
            "- Vary sentence length: mix short punchy lines with longer flowing ones\n"
            "- Use conversational connectors ('honestly', 'here's the thing', 'what surprised me')\n"
            "- Pose occasional rhetorical questions to pull the reader in\n"
            "- Prefer concrete, vivid language over abstract generalities\n"
            "- Avoid listicle-style bullet dumps — weave points into flowing paragraphs\n\n"
            "CONTENT RULES:\n"
            "- Fully address the task goal and cover all bullet points\n"
            "- Stay within target word count\n"
            "- Match the overall tone and audience\n"
            "- Integrate research context where relevant (cite specifics, not vague nods)\n"
            "- Incorporate SEO keywords naturally — never force them\n\n"
            "Think of this as writing a section of a magazine column, not an essay assignment."
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
Write it like you're explaining this to a friend over coffee — knowledgeable but never stuffy.
Bring in your own perspective or a relatable scenario where it fits.
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
    persona_str = persona if persona else "A thoughtful writer who shares personal insights and speaks directly to the reader"
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
            "You are weaving individually written sections into one seamless blog post.\n"
            "The sections were drafted in a personal, conversational voice — your job is to\n"
            "PRESERVE that voice while connecting the pieces.\n\n"
            "Your priorities:\n"
            "- Write transitions that feel like natural shifts in a conversation, not academic segues\n"
            "- Add an introduction that hooks with a story, question, or bold statement — not a summary\n"
            "- Write a conclusion that lands with a personal reflection or memorable thought\n"
            "- Keep the first-person or warm third-person voice consistent throughout\n"
            "- If a section sounds too formal or robotic, soften it during integration\n"
            "- Target the total word count closely\n\n"
            "Do NOT flatten personality into generic prose. The article should read like one person\n"
            "wrote it in a single sitting, sharing what they genuinely think and know."
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
Weave this into a complete blog post that reads like one person wrote it:

1. Write an introduction (100-200 words) that:
   - Opens with a hook: a short story, a surprising fact, a question, or a bold opinion
   - Avoids the pattern 'In this article, we will explore...' — that's a dead giveaway
   - Sets a personal, conversational tone from the first sentence

2. Combine all sections in order, adding transitions that feel like natural thought progression
   (e.g. 'Which brings me to something I keep coming back to...' not 'Moving on to the next topic...')

3. Write a conclusion (100-150 words) that:
   - Offers a personal reflection, a forward-looking thought, or a memorable takeaway
   - Feels like the writer wrapping up a conversation, not filing a report
   - Do NOT add a generic call-to-action like "contact us", "reach out", "ready to explore", etc.

4. Read the full article back — does it sound like a person talking? If any part sounds robotic, fix it.

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
    persona_str = persona if persona else "A thoughtful writer who shares personal insights and speaks directly to the reader"
    tone_str = tone_profile if tone_profile else "Warm, conversational, and opinionated — like a knowledgeable friend sharing what they've learned"

    keywords_str = ""
    if primary_keyword:
        keywords_str += f"Primary: {primary_keyword}\n"
    if secondary_keywords:
        keywords_str += f"Secondary: {', '.join(secondary_keywords)}\n"

    system_msg = SystemMessage(
        content=(
            "You are a voice editor doing the final pass on a blog post.\n"
            "Your single obsession: make this sound like a HUMAN wrote it.\n\n"
            "HUMANIZATION CHECKLIST (apply every time):\n"
            "- Replace any 'In today's world...', 'It is important to note...',\n"
            "  'This article explores...' phrasing with natural alternatives\n"
            "- Ensure the article uses 'I', 'we', 'you' — a real person is talking\n"
            "- Break up any paragraph longer than 4-5 sentences\n"
            "- Add at least 2-3 rhetorical questions across the article\n"
            "- Vary rhythm: follow a long sentence with a short punchy one\n"
            "- Replace generic adjectives ('important', 'significant', 'various')\n"
            "  with specific, vivid ones\n"
            "- If there's an example post, match its voice and quirks closely\n"
            "- If there's no example post, default to a warm, opinionated column-style voice\n"
            "- Apply the tone_profile from the plan as your north star\n"
            "- Incorporate SEO keywords naturally — never force them\n\n"
            "PRESERVE the core content and structure. Your job is voice, not rewriting.\n\n"
            "IMPORTANT: Do NOT append a keyword list, keyword block, or any meta-information\n"
            "at the end. The article must end naturally with the conclusion."
        )
    )

    example_section = ""
    if example_post:
        example_section = (
            f"REFERENCE VOICE (match this style closely):\n{example_post}\n\n"
            "Analyze this example's sentence structure, word choice, use of 'I'/'we',\n"
            "humour, metaphors, paragraph length, and opening/closing patterns.\n"
            "Then apply those exact patterns to the draft below.\n\n"
        )
    else:
        example_section = (
            "NO EXAMPLE POST PROVIDED.\n"
            "Default to a warm, opinionated voice — like a knowledgeable friend writing a column.\n"
            "Use first person naturally. Share brief reflections or reactions to the material.\n"
            "Imagine someone who genuinely cares about the topic writing on their personal blog.\n\n"
        )

    user_msg = HumanMessage(
        content=f"""TARGET VOICE:
Persona: {persona_str}

Tone Profile:
{tone_str}

{example_section}CURRENT DRAFT:
{stitched_draft}

SEO Keywords:
{keywords_str if keywords_str else "None specified"}

Task:
Polish this draft so it sounds unmistakably human:

1. Read the draft aloud in your head — flag anything that sounds like 'AI wrote this'
2. Replace stiff transitions, generic openers, and passive constructions
3. Ensure the persona's voice is consistent from first paragraph to last
4. Incorporate SEO keywords where they fit naturally
5. Keep paragraph lengths varied and readable

IMPORTANT: Do NOT append a keyword list, keyword block, or any meta-information at the end.
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


def autopilot_topic_prompt(
    brand_name: str,
    brand_context: str,
    existing_posts: List[dict],
    batch_size: int = 5,
) -> tuple[SystemMessage, HumanMessage]:
    """
    Prompt the LLM to generate new, non-duplicate blog topic suggestions
    by analysing the existing WordPress catalog and brand context.
    """
    catalog_str = ""
    if existing_posts:
        for i, post in enumerate(existing_posts, 1):
            title = post.get("title", "Untitled")
            excerpt = post.get("excerpt", "")[:200].strip()
            catalog_str += f"  {i}. {title}\n"
            if excerpt:
                catalog_str += f"     {excerpt}\n"
    else:
        catalog_str = "  (no posts yet)\n"

    blog_kinds = (
        "concept_explainer, procedural_guide, analytical_compare, "
        "digest_roundup, structural_deepdive, narrative_log, "
        "inquiry_response, resource_curation"
    )

    system_msg = SystemMessage(
        content=(
            f"You are a content strategist for {brand_name or 'a brand'}.\n"
            "Your job is to analyse the existing blog catalog, identify content "
            "gaps, and propose fresh topics that complement — never duplicate — "
            "what has already been published.\n\n"
            "Each suggestion must include a detailed writing prompt a writer "
            "could execute without further briefing.\n"
            "Vary the blog_kind_hint across suggestions to keep the blog diverse.\n"
            f"Valid blog_kind values: {blog_kinds}"
        )
    )

    brand_section = ""
    if brand_context:
        preview = brand_context[:3000]
        brand_section = f"\nBrand / Company Context:\n{preview}\n"

    user_msg = HumanMessage(
        content=f"""Brand: {brand_name or '(not specified)'}
{brand_section}
Existing Blog Catalog:
{catalog_str}
Task:
Suggest exactly {batch_size} new blog topics that:
1. Do NOT duplicate or substantially overlap any existing post above
2. Fill gaps in the catalog's coverage
3. Are relevant to the brand's audience and domain
4. Span a variety of blog_kind types
5. Each include a self-contained writing prompt (2-4 sentences)

Return ONLY a JSON object matching the TopicSuggestions schema."""
    )

    return system_msg, user_msg
