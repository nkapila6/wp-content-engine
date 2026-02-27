# Disclaimer: Written by AI with small modifications from my end.
# 27/02/2026 02:35PM Nikhil Kapila

#!/usr/bin/env python3
"""
Autopilot runner for WP Content Engine.

Fetches existing WordPress posts, uses an LLM to identify content gaps,
generates new topics autonomously, and runs the full pipeline for each.

Usage:
    uv run python -m wp_content_engine.autopilot [OPTIONS]

Options:
    --max-posts N          Stop after N posts (default: unlimited)
    --batch-size N         Topics to generate per LLM call (default: 5)
    --cooldown N           Seconds between pipeline runs (default: 10)
    --dry-run              Generate topics and print them; skip pipeline
    --quiet                Minimal output
    --save-intermediates   Save each stage output per run
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import markdown
from dotenv import load_dotenv

from wp_content_engine.llm import llm
from wp_content_engine.main import build_graph, FLAGS
from wp_content_engine.batch import load_brand_context
from wp_content_engine.nodes.wp_nodes import fetch_existing_posts
from wp_content_engine.prompts.prompts import autopilot_topic_prompt
from wp_content_engine.state.state import TopicSuggestions

load_dotenv()

COMPLETED_LOG = "autopilot_completed.jsonl"
OUTPUT_DIR = "autopilot_outputs"


# ── CLI helpers ──────────────────────────────────────────────────────────

def _parse_int_flag(flag: str, default: int) -> int:
    """Extract --flag N from sys.argv, return *default* when absent."""
    for i, arg in enumerate(sys.argv):
        if arg == flag and i + 1 < len(sys.argv):
            try:
                return int(sys.argv[i + 1])
            except ValueError:
                pass
        if arg.startswith(f"{flag}="):
            try:
                return int(arg.split("=", 1)[1])
            except ValueError:
                pass
    return default


def _validate_wp_credentials():
    wp_url = os.getenv("WP_URL", "")
    wp_user = os.getenv("WP_USER", "")
    wp_password = os.getenv("WP_APP_PASSWORD", "")
    if not wp_url or not wp_user or not wp_password:
        print(
            "\n  ERROR: Autopilot requires WordPress credentials.\n"
            "  Set WP_URL, WP_USER, and WP_APP_PASSWORD in your .env file."
        )
        sys.exit(1)


# ── Topic generation ─────────────────────────────────────────────────────

def generate_topics(
    brand_name: str,
    brand_context: str,
    existing_posts: list[dict],
    batch_size: int,
) -> list[dict]:
    """Ask the LLM for *batch_size* new topic suggestions."""
    sys_msg, human_msg = autopilot_topic_prompt(
        brand_name=brand_name,
        brand_context=brand_context,
        existing_posts=existing_posts,
        batch_size=batch_size,
    )
    structured_llm = llm.with_structured_output(TopicSuggestions)
    result: TopicSuggestions = structured_llm.invoke([sys_msg, human_msg])
    return [s.model_dump() for s in result.suggestions]


# ── State builder ────────────────────────────────────────────────────────

def suggestion_to_initial_state(
    suggestion: dict,
    brand_name: str,
    brand_context: str,
    kb_root: str,
    existing_titles: list[str],
) -> dict:
    """Convert a TopicSuggestion dict into an AgentState initial dict."""
    raw_prompt = suggestion.get("prompt", "")
    if existing_titles:
        diversity_note = (
            "Previously generated titles (avoid overlap): "
            + ", ".join(existing_titles[-30:])
        )
        raw_prompt += f"\n\nDIVERSITY NOTE: {diversity_note}"

    return {
        "topic": suggestion.get("topic", ""),
        "raw_prompt": raw_prompt,
        "persona": suggestion.get("persona", "informative blogger"),
        "example_post": "",
        "target_words_total": suggestion.get("target_words", 1500),
        "blog_kind_hint": suggestion.get("blog_kind_hint", ""),
        "primary_keyword": suggestion.get("primary_keyword", ""),
        "secondary_keywords": [],
        "seo_keywords": [],
        "kb_root": kb_root,
        "brand_name": brand_name,
        "brand_context": brand_context,
        "ddgs_queries": [],
        "rg_queries": [],
        "ddgs_num_results": 5,
        "errors": [],
        "debug_logs": [],
    }


# ── Logging helpers ──────────────────────────────────────────────────────

def _log_completion(entry: dict):
    with open(COMPLETED_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _save_outputs(result: dict, slug: str):
    run_dir = Path(OUTPUT_DIR) / slug
    run_dir.mkdir(parents=True, exist_ok=True)

    body_md = result.get("wp_body", "")
    body_html = (
        markdown.markdown(body_md, extensions=["extra", "sane_lists"])
        if body_md
        else ""
    )

    wp_payload = {
        "title": result.get("wp_title", ""),
        "subtitle": result.get("wp_subtitle", ""),
        "body": body_html,
        "excerpt": result.get("wp_excerpt", ""),
        "slug": slug,
        "tags": result.get("wp_tags", []),
        "categories": result.get("wp_categories", []),
        "seo": {
            "meta_title": result.get("seo_meta_title", ""),
            "meta_description": result.get("seo_meta_description", ""),
            "primary_keyword": result.get("primary_keyword", ""),
            "secondary_keywords": result.get("secondary_keywords", []),
            "seo_keywords": result.get("seo_keywords", []),
        },
    }

    with open(run_dir / "wp_post.json", "w", encoding="utf-8") as f:
        json.dump(wp_payload, f, indent=2, ensure_ascii=False)

    with open(run_dir / "article.md", "w", encoding="utf-8") as f:
        f.write(f"# {wp_payload['title']}\n\n")
        f.write(wp_payload["body"] or "*No content*")


# ── Main loop ────────────────────────────────────────────────────────────

def main():
    max_posts = _parse_int_flag("--max-posts", default=0)  # 0 = unlimited
    batch_size = _parse_int_flag("--batch-size", default=5)
    cooldown = _parse_int_flag("--cooldown", default=10)
    dry_run = "--dry-run" in sys.argv

    FLAGS["save_intermediates"] = "--save-intermediates" in sys.argv
    FLAGS["quiet"] = "--quiet" in sys.argv

    if not dry_run:
        _validate_wp_credentials()

    kb_root = os.getenv("KB_ROOT", "")
    brand_context, resolved_kb = load_brand_context(kb_root)
    brand_name = os.getenv("BRAND_NAME", "")

    if not brand_name and brand_context:
        brand_name = input("  Brand name (e.g. 'The Scholars School'): ").strip()

    # Fetch current WordPress catalog
    print("\n  Fetching existing WordPress posts...")
    existing_posts = fetch_existing_posts()
    print(f"  Found {len(existing_posts)} existing posts.")

    existing_titles = [p["title"] for p in existing_posts]

    app = None
    if not dry_run:
        graph = build_graph()
        app = graph.compile(debug=not FLAGS["quiet"])
        Path(OUTPUT_DIR).mkdir(exist_ok=True)

    posts_created = 0
    results_summary: list[dict] = []

    try:
        while True:
            remaining = (max_posts - posts_created) if max_posts else batch_size
            request_size = min(batch_size, remaining) if max_posts else batch_size

            if request_size <= 0:
                break

            print(f"\n{'=' * 60}")
            print(f"  Generating {request_size} topic suggestions...")
            print(f"{'=' * 60}")

            candidates = generate_topics(
                brand_name=brand_name,
                brand_context=brand_context,
                existing_posts=existing_posts,
                batch_size=request_size,
            )

            for idx, suggestion in enumerate(candidates, 1):
                topic = suggestion.get("topic", "Untitled")
                rationale = suggestion.get("rationale", "")

                print(f"\n  [{posts_created + 1}] {topic}")
                if rationale:
                    print(f"      Rationale: {rationale}")

                if dry_run:
                    kind = suggestion.get("blog_kind_hint", "")
                    words = suggestion.get("target_words", "")
                    print(f"      Kind: {kind}  |  Words: {words}")
                    print(f"      Prompt: {suggestion.get('prompt', '')[:120]}...")
                    posts_created += 1
                    if max_posts and posts_created >= max_posts:
                        break
                    continue

                initial_state = suggestion_to_initial_state(
                    suggestion,
                    brand_name=brand_name,
                    brand_context=brand_context,
                    kb_root=resolved_kb,
                    existing_titles=existing_titles,
                )

                try:
                    result = app.invoke(initial_state)

                    errors = result.get("errors", [])
                    if errors:
                        print(f"      WARNINGS: {errors}")

                    slug = result.get("seo_slug", f"autopilot-{posts_created + 1}") or f"autopilot-{posts_created + 1}"
                    title = result.get("wp_title", "") or result.get("seo_meta_title", "")
                    wp_url = result.get("wp_post_url", "")

                    _save_outputs(result, slug)

                    existing_posts.append({
                        "id": result.get("wp_post_id", ""),
                        "title": title,
                        "slug": slug,
                        "excerpt": result.get("wp_excerpt", ""),
                    })
                    existing_titles.append(title)

                    _log_completion({
                        "topic": topic,
                        "slug": slug,
                        "title": title,
                        "wp_post_url": wp_url,
                        "status": "success",
                        "errors": errors,
                        "timestamp": datetime.now().isoformat(),
                    })

                    results_summary.append({"title": title, "slug": slug, "url": wp_url})
                    posts_created += 1
                    print(f"      SUCCESS: {slug}")
                    if wp_url:
                        print(f"      URL: {wp_url}")

                except Exception as exc:
                    print(f"      FAILED: {exc}")
                    _log_completion({
                        "topic": topic,
                        "status": "failed",
                        "error": str(exc),
                        "timestamp": datetime.now().isoformat(),
                    })

                if max_posts and posts_created >= max_posts:
                    break

                if cooldown and idx < len(candidates):
                    print(f"\n      Cooling down {cooldown}s...")
                    time.sleep(cooldown)

            if max_posts and posts_created >= max_posts:
                break

    except KeyboardInterrupt:
        print("\n\n  Interrupted by user.")

    # Summary
    print(f"\n{'=' * 60}")
    print("  AUTOPILOT COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Posts created: {posts_created}")
    if results_summary:
        for r in results_summary:
            url_part = f"  ->  {r['url']}" if r.get("url") else ""
            print(f"    - {r['title']}{url_part}")
    if not dry_run:
        print(f"  Output dir:   {OUTPUT_DIR}/")
        print(f"  Log:          {COMPLETED_LOG}")


if __name__ == "__main__":
    main()
