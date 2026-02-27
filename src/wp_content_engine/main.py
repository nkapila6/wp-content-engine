# main entry point
# 25/02/2026 02:10PM Nikhil Kapila

"""
Main entry point for WP Content Engine graph.

Usage:
    uv run python -m wp_content_engine.main
    or
    uv run python src/wp_content_engine/main.py [OPTIONS]

Options:
    --show-graph          Display ASCII graph visualization before execution
    --quiet               Minimal output, save intermediates
    --save-intermediates  Save each stage output to separate files
    --summary-only         Show only summary, not node-by-node details
"""

import sys
import os
import json
from typing import Literal
from pathlib import Path

import markdown
from dotenv import load_dotenv
from tqdm import tqdm

from langgraph.graph import StateGraph, END

from wp_content_engine.state.state import AgentState
from wp_content_engine.nodes.query_nodes import enhance_queries_node
from wp_content_engine.nodes.ddgs_nodes import ddgs_search_node, ddgs_summary_node
from wp_content_engine.nodes.rg_nodes import exec_ripgrep_queries_node, rg_summary_node
from wp_content_engine.nodes.nodes import (
    condenser_node,
    planner_node,
    draft_task_node,
    task_revision_node,
    advance_task_node,
    stitcher_node,
    styler_node,
    seo_node,
    wp_format_node,
)
from wp_content_engine.nodes.wp_nodes import wp_push_node

load_dotenv()


# Parse command line flags
FLAGS = {
    "show_graph": "--show-graph" in sys.argv,
    "quiet": "--quiet" in sys.argv,
    "save_intermediates": "--save-intermediates" in sys.argv,
    "summary_only": "--summary-only" in sys.argv,
}


def save_intermediate(stage: str, data: dict, output_dir: str = "outputs"):
    """
    Save intermediate output to a JSON file for easier inspection.

    Args:
        stage: Name of the stage/node
        data: Data to save
        output_dir: Directory to save to (default: outputs/)
    """
    if not FLAGS["save_intermediates"]:
        return

    Path(output_dir).mkdir(exist_ok=True)

    suffix = ""
    if "current_task_id" in data:
        suffix = f" - Task {data['current_task_id']}"
    elif stage in ("Draft Task", "Task Revision") and "task_drafts" in data:
        task_ids = list(data["task_drafts"].keys())
        if task_ids:
            suffix = f" - Task {task_ids[0]}"

    filename = f"{output_dir}/{stage}{suffix}.json"

    serializable_data = {}
    for key, value in data.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool, list)):
            serializable_data[key] = value
        elif isinstance(value, dict):
            serializable_data[key] = value
        elif hasattr(value, "model_dump"):
            serializable_data[key] = value.model_dump()
        else:
            serializable_data[key] = str(value)

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(serializable_data, f, indent=2, ensure_ascii=False, default=str)


def display_stage_output(node_name: str, output: dict):
    """
    Display node output in a clean, readable format.

    Args:
        node_name: Name of the node
        output: Output from the node
    """
    if FLAGS["quiet"] or FLAGS["summary_only"]:
        return

    print(f"\n{'=' * 60}")
    print(f" {node_name}")
    print(f"{'=' * 60}")

    # Show key outputs only
    key_outputs = {
        "ddgs_queries": "Web queries",
        "rg_queries": "Ripgrep queries",
        "ddgs_results": "Web results",
        "rg_results": "KB results",
        "ddgs_result_summary": "Web summary",
        "rg_result_summary": "KB summary",
        "condensed_content": "Condensed context",
        "plan": "Plan",
        "current_task_id": "Current task",
        "completed_task_ids": "Completed tasks",
        "task_drafts": "Task drafts",
        "stitched_draft": "Stitched article",
        "styled_draft": "Styled article",
        "seo_meta_title": "SEO title",
        "seo_slug": "SEO slug",
        "primary_keyword": "Primary keyword",
        "secondary_keywords": "Secondary keywords",
        "wp_title": "WP title",
        "wp_subtitle": "WP subtitle",
        "wp_excerpt": "WP excerpt",
        "wp_tags": "WP tags",
        "wp_categories": "WP categories",
        "wp_post_url": "WP post URL",
    }

    for key, label in key_outputs.items():
        if key in output and output[key]:
            value = output[key]

            # Format based on type
            if isinstance(value, list):
                if len(value) <= 5:
                    print(f"\n{label}:")
                    for i, item in enumerate(value, 1):
                        print(f"  {i}. {str(item)[:100]}")
                else:
                    print(f"\n{label}: {len(value)} items")
            elif isinstance(value, dict):
                print(f"\n{label}: {len(value)} items")
                for k, v in list(value.items())[:3]:
                    preview = str(v)[:100] + "..." if len(str(v)) > 100 else str(v)
                    print(f"  {k}: {preview}")
                if len(value) > 3:
                    print(f"  ... and {len(value) - 3} more")
            elif isinstance(value, str):
                if len(value) <= 300:
                    print(f"\n{label}:\n{value}")
                else:
                    print(f"\n{label} ({len(value)} chars):")
                    print(f"{value[:300]}...")
            elif hasattr(value, "model_fields"):  # Pydantic model
                print(f"\n{label}: Pydantic model")
                for field in list(value.model_fields.keys())[:5]:
                    print(f"  - {field}")
            else:
                print(f"\n{label}: {value}")


# Wrapper functions for clean output
def wrap_node(node_func, node_name: str):
    """
    Wrap a node function to display output and save intermediates.

    Args:
        node_func: The original node function
        node_name: Name for display

    Returns:
        Wrapped function
    """

    def wrapped(state, config=None):
        # Call original node
        result = node_func(state, config)

        # Display output
        if result:
            display_stage_output(node_name, result)

        # Save intermediate
        save_intermediate(node_name, result)

        return result

    return wrapped


def check_tasks_remaining(state: AgentState) -> Literal["continue", "done"]:
    """
    Conditional edge function to check if more tasks need to be drafted.

    Returns:
        "continue" if more tasks remain, "done" if all tasks complete
    """
    current_task_id = state.get("current_task_id")
    if current_task_id is None:
        return "done"
    return "continue"


def get_user_input() -> AgentState:
    """
    Get blog post details from user via CLI prompts.

    Returns:
        AgentState with user-provided values
    """
    print("\n" + "=" * 60)
    print(" BLOG POST CONFIGURATION")
    print("=" * 60)

    topic = input("\n Blog topic: ").strip()
    if not topic:
        print(" Topic is required!")
        sys.exit(1)

    prompt = input(" Main prompt/description: ").strip()
    if not prompt:
        print(" Prompt is required!")
        sys.exit(1)

    persona = (
        input("\n Persona (e.g., 'Technical educator', 'Python developer'): ").strip()
        or "A thoughtful writer who shares personal insights and speaks directly to the reader"
    )

    example = input("\n Example post (paste or press Enter to skip): ").strip()

    words_input = input("\n Target word count (default 1500): ").strip()
    target_words = int(words_input) if words_input else 1500

    print("\n" + "-" * 60)
    print("SEO CONFIGURATION (optional)")
    print("-" * 60)

    use_seo = input("Add SEO keywords? (y/N, default N): ").strip().lower() == "y"

    primary_keyword = ""
    secondary_keywords = []
    seo_keywords = []

    if use_seo:
        primary_keyword = input("\n Primary keyword: ").strip()

        secondary_input = input(" Secondary keywords (comma-separated): ").strip()
        secondary_keywords = (
            [k.strip() for k in secondary_input.split(",")] if secondary_input else []
        )

        seo_input = input(" Additional SEO keywords (comma-separated): ").strip()
        seo_keywords = [k.strip() for k in seo_input.split(",")] if seo_input else []

    kb_root = os.getenv("KB_ROOT", "")
    if kb_root:
        kb_path = Path(kb_root).expanduser().resolve()
        print(f"\n Knowledge base root: {kb_path}")
    else:
        kb_path = None

    brand_name = os.getenv("BRAND_NAME", "")
    brand_context = ""
    if kb_path and kb_path.exists():
        kb_texts = []
        for f in sorted(kb_path.glob("*.md")):
            try:
                kb_texts.append(f.read_text(encoding="utf-8"))
            except Exception:
                pass
        if kb_texts:
            brand_context = "\n\n---\n\n".join(kb_texts)
            if not brand_name:
                brand_name = input(
                    "\n Brand name (from KB, e.g. 'The Scholars School'): "
                ).strip()

    return {
        "topic": topic,
        "raw_prompt": prompt,
        "persona": persona,
        "example_post": example,
        "target_words_total": target_words,
        "primary_keyword": primary_keyword,
        "secondary_keywords": secondary_keywords,
        "seo_keywords": seo_keywords,
        "kb_root": str(kb_path) if kb_path else "",
        "brand_name": brand_name,
        "brand_context": brand_context,
        "ddgs_queries": [],
        "rg_queries": [],
        "ddgs_num_results": 5,
        "errors": [],
        "debug_logs": [],
    }


def build_graph() -> StateGraph:
    """
    Build complete LangGraph for blog post generation.

    Returns:
        Compiled StateGraph ready for execution
    """
    graph = StateGraph(AgentState)

    # Wrap nodes for clean output
    if FLAGS["save_intermediates"] or not FLAGS["quiet"]:
        enhance_queries = wrap_node(enhance_queries_node, "Enhance Queries")
        ddgs_search = wrap_node(ddgs_search_node, "DDGS Search")
        exec_ripgrep = wrap_node(exec_ripgrep_queries_node, "Ripgrep Search")
        ddgs_summary = wrap_node(ddgs_summary_node, "DDGS Summary")
        rg_summary = wrap_node(rg_summary_node, "Ripgrep Summary")
        condenser = wrap_node(condenser_node, "Condenser")
        planner = wrap_node(planner_node, "Planner")
        draft_task = wrap_node(draft_task_node, "Draft Task")
        task_revision = wrap_node(task_revision_node, "Task Revision")
        advance_task = wrap_node(advance_task_node, "Advance Task")
        stitcher = wrap_node(stitcher_node, "Stitcher")
        styler = wrap_node(styler_node, "Styler")
        seo = wrap_node(seo_node, "SEO")
        wp_format = wrap_node(wp_format_node, "WP Format")
        wp_push = wrap_node(wp_push_node, "WP Push")
    else:
        enhance_queries = enhance_queries_node
        ddgs_search = ddgs_search_node
        exec_ripgrep = exec_ripgrep_queries_node
        ddgs_summary = ddgs_summary_node
        rg_summary = rg_summary_node
        condenser = condenser_node
        planner = planner_node
        draft_task = draft_task_node
        task_revision = task_revision_node
        advance_task = advance_task_node
        stitcher = stitcher_node
        styler = styler_node
        seo = seo_node
        wp_format = wp_format_node
        wp_push = wp_push_node

    # Research phase
    graph.add_node("enhance_queries", enhance_queries)
    graph.add_node("ddgs_search", ddgs_search)
    graph.add_node("exec_ripgrep", exec_ripgrep)

    # Summary phase
    graph.add_node("ddgs_summary", ddgs_summary)
    graph.add_node("rg_summary", rg_summary)

    # Planning phase
    graph.add_node("condenser", condenser)
    graph.add_node("planner", planner)

    # Writing phase (task loop)
    graph.add_node("draft_task", draft_task)
    graph.add_node("task_revision", task_revision)
    graph.add_node("advance_task", advance_task)

    # Finalization phase
    graph.add_node("stitcher", stitcher)
    graph.add_node("styler", styler)
    graph.add_node("seo", seo)
    graph.add_node("wp_format", wp_format)
    graph.add_node("wp_push", wp_push)

    # Define edges

    # Entry point
    graph.set_entry_point("enhance_queries")

    # Research phase (parallel execution of DDGS and ripgrep)
    graph.add_edge("enhance_queries", "ddgs_search")
    graph.add_edge("enhance_queries", "exec_ripgrep")

    # Summary phase
    graph.add_edge("ddgs_search", "ddgs_summary")
    graph.add_edge("exec_ripgrep", "rg_summary")

    # Condense both summaries
    graph.add_edge("ddgs_summary", "condenser")
    graph.add_edge("rg_summary", "condenser")

    # Planning
    graph.add_edge("condenser", "planner")

    # Writing loop
    graph.add_edge("planner", "draft_task")
    graph.add_edge("draft_task", "task_revision")
    graph.add_edge("task_revision", "advance_task")

    # Conditional edge: continue writing or finalize
    graph.add_conditional_edges(
        "advance_task",
        check_tasks_remaining,
        {"continue": "draft_task", "done": "stitcher"},
    )

    # Finalization phase
    graph.add_edge("stitcher", "styler")
    graph.add_edge("styler", "seo")
    graph.add_edge("seo", "wp_format")
    graph.add_edge("wp_format", "wp_push")
    graph.add_edge("wp_push", END)

    return graph


def print_node_progress(node_name: str, total_tasks: int, current_step: int):
    """Print progress bar for node execution."""
    progress = tqdm(
        desc=f"Node: {node_name}",
        total=1,
        ncols=80,
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {elapsed}",
        leave=False,
    )
    progress.update(1)
    progress.close()


def save_to_file(state: AgentState, slug: str):
    """
    Save the final blog post as both WP-ready JSON and a markdown preview.

    Args:
        state: The final AgentState with all results
        slug: SEO slug used for filenames

    Returns:
        Tuple of (json_path, md_path)
    """
    seo_title = state.get("seo_meta_title", "")
    seo_desc = state.get("seo_meta_description", "")
    primary_kw = state.get("primary_keyword", "")
    secondary_kws = state.get("secondary_keywords", [])

    body_md = state.get("wp_body", "") or state.get("styled_draft", "")
    body_html = markdown.markdown(body_md, extensions=["extra", "sane_lists"]) if body_md else ""

    wp_payload = {
        "title": state.get("wp_title", "") or seo_title,
        "subtitle": state.get("wp_subtitle", ""),
        "body": body_html,
        "excerpt": state.get("wp_excerpt", "") or seo_desc,
        "slug": state.get("seo_slug", slug),
        "tags": state.get("wp_tags", []),
        "categories": state.get("wp_categories", []),
        "seo": {
            "meta_title": seo_title,
            "meta_description": seo_desc,
            "primary_keyword": primary_kw,
            "secondary_keywords": secondary_kws,
            "seo_keywords": state.get("seo_keywords", []),
            "og_title": seo_title,
            "og_description": seo_desc,
            "twitter_title": seo_title,
            "twitter_description": seo_desc,
        },
    }

    json_path = f"{slug}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(wp_payload, f, indent=2, ensure_ascii=False)

    md_path = f"{slug}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# {wp_payload['title']}\n\n")
        if wp_payload["subtitle"]:
            f.write(f"*{wp_payload['subtitle']}*\n\n")
        f.write("---\n\n")
        f.write(wp_payload["body"] or "*No content generated*")

    return json_path, md_path


def display_full_state(state: AgentState):
    """
    Display the full state including all fields and errors.

    Args:
        state: The AgentState to display
    """
    print("\n" + "=" * 60)
    print(" FULL STATE")
    print("=" * 60)

    for key, value in state.items():
        if value is None or value == [] or value == "":
            continue

        print(f"\n {key}:")

        if isinstance(value, dict) and len(value) > 5:
            print(f"  (dict with {len(value)} items)")
            for k, v in list(value.items())[:3]:
                v_preview = str(v)[:100] + "..." if len(str(v)) > 100 else str(v)
                print(f"    {k}: {v_preview}")
            print(f"  ... and {len(value) - 3} more items")
        elif isinstance(value, list) and len(value) > 5:
            print(f"  (list with {len(value)} items)")
            for i, item in enumerate(value[:3]):
                print(f"    [{i}]: {str(item)[:100]}...")
            print(f"  ... and {len(value) - 3} more items")
        elif isinstance(value, str) and len(value) > 200:
            print(f"  {value[:200]}...")
        else:
            print(f"  {value}")

    # Highlight errors if any
    errors = state.get("errors", [])
    if errors:
        print("\n" + "!" * 60)
        print(" ERRORS")
        print("!" * 60)
        for error in errors:
            print(f"\n• {error}")


def display_results(state: AgentState):
    """
    Display the final blog post and SEO metadata.

    Args:
        state: The final AgentState with results
    """
    print("\n" + "=" * 60)
    print(" EXECUTION COMPLETE")
    print("=" * 60)

    # Display SEO metadata
    print("\n SEO METADATA")
    print("-" * 60)
    print(f"Title: {state.get('seo_meta_title', '')}")
    print(f"Slug: {state.get('seo_slug', '')}")
    print(f"Primary Keyword: {state.get('primary_keyword', '')}")

    secondary = state.get("secondary_keywords", [])
    if secondary:
        print(f"Secondary Keywords: {', '.join(secondary)}")

    seo_kw = state.get("seo_keywords", [])
    if seo_kw:
        print(f"SEO Keywords: {', '.join(seo_kw)}")

    print(f"Meta Description: {state.get('seo_meta_description', '')}")

    # Display final article (truncated for console)
    print("\n" + "=" * 60)
    print(" FINAL ARTICLE")
    print("=" * 60)

    styled_draft = state.get("styled_draft", "")
    if styled_draft:
        preview_length = 1000
        if len(styled_draft) > preview_length:
            print(
                f"{styled_draft[:preview_length]}\n\n... ({len(styled_draft) - preview_length} more characters)"
            )
        else:
            print(styled_draft)
    else:
        print(" No article content generated")

    wp_title = state.get("wp_title", "")
    if wp_title:
        print("\n" + "=" * 60)
        print(" WP OUTPUT")
        print("=" * 60)
        print(f"Title: {wp_title}")
        print(f"Subtitle: {state.get('wp_subtitle', '')}")
        print(f"Excerpt: {state.get('wp_excerpt', '')}")
        wp_url = state.get("wp_post_url", "")
        if wp_url:
            print(f"Published: {wp_url}")


def main():
    """
    Main execution function for WP Content Engine.
    """
    # Check for CLI flags
    show_graph = "--show-graph" in sys.argv

    # Build graph
    graph = build_graph()

    # Show graph structure if requested
    if FLAGS["show_graph"]:
        print("\n" + "=" * 60)
        print(" GRAPH STRUCTURE")
        print("=" * 60)
        graph.print_ascii()
        print("=" * 60)
        print()

    # Get user input
    initial_state = get_user_input()

    # Display configuration summary
    print("\n" + "=" * 60)
    print("  CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"Topic: {initial_state['topic']}")
    print(f"Target Words: {initial_state['target_words_total']}")
    print(f"Persona: {initial_state['persona']}")
    print(f"Primary Keyword: {initial_state['primary_keyword'] or 'Not specified'}")
    print("=" * 60)

    # Execute graph
    if FLAGS["quiet"]:
        print("\n" + "=" * 60)
        print(" EXECUTING (Quiet Mode)")
        print("=" * 60)
        print()
    elif FLAGS["summary_only"]:
        print("\n" + "=" * 60)
        print(" EXECUTING (Summary Only)")
        print("=" * 60)
        print()
    else:
        print("\n" + "=" * 60)
        print(" EXECUTING (Detailed Mode)")
        print("=" * 60)
        print()

    if FLAGS["save_intermediates"]:
        print(" Intermediate outputs will be saved to: outputs/")
        print()

    # Compile graph (use debug mode only if not quiet)
    if FLAGS["quiet"]:
        app = graph.compile(debug=False)
    else:
        app = graph.compile(debug=True)

    try:
        # Execute graph
        result = app.invoke(initial_state)

        # Check for errors
        errors = result.get("errors", [])
        if errors:
            print("\n" + "!" * 60)
            print(" ERRORS ENCOUNTERED")
            print("!" * 60)
            for error in errors:
                print(f"\n• {error}")

            # Display full state
            display_full_state(result)

            raise Exception("Graph execution failed with errors")

        # Display results
        display_results(result)

        # Save to file
        slug = result.get("seo_slug", "blog_post") or "blog_post"
        json_path, md_path = save_to_file(result, slug)

        print(f"\n WP JSON saved to: {json_path}")
        print(f" Markdown preview saved to: {md_path}")

        print("\n" + "=" * 60)
        print(" SUCCESS!")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\n" + "!" * 60)
        print("  Execution interrupted by user")
        print("!" * 60)
        sys.exit(1)
    except Exception as e:
        print("\n\n" + "!" * 60)
        print(f" ERROR: {e}")
        print("!" * 60)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
