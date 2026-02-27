# Disclaimer: Written by AI with small modifications from my end.
# 27/02/2026 10:00AM Nikhil Kapila

#!/usr/bin/env python3
"""
Batch runner for WP Content Engine.

Reads a CSV/JSON manifest of blog topics and runs the pipeline for each,
tracking previously generated content to ensure diversity.

Usage:
    uv run python -m wp_content_engine.batch manifest.csv [OPTIONS]

Options:
    --save-intermediates   Save each stage output per run
    --quiet                Minimal output
    --dry-run              Validate manifest without running pipeline
    --resume               Skip topics already in the completed log
    --output-dir DIR       Output directory (default: batch_outputs/)

CSV format:
    topic,prompt,persona,primary_keyword,secondary_keywords,target_words
    "Good education","Why education matters...",...,...,1500

JSON format:
    [{"topic": "...", "prompt": "...", ...}, ...]
"""

import csv
import json
import os
import sys
import hashlib
from datetime import datetime
from pathlib import Path

import markdown
from dotenv import load_dotenv

from wp_content_engine.main import build_graph, FLAGS

load_dotenv()


COMPLETED_LOG = "batch_completed.jsonl"
TITLE_CACHE = "batch_titles.json"
STATUS_CSV = "batch_status.csv"

STATUS_CSV_FIELDS = [
    "row", "topic", "status", "slug", "title", "wp_post_url",
    "tags", "categories", "errors", "timestamp",
]


def load_manifest(path: str) -> list[dict]:
    """Load topics from CSV or JSON manifest."""
    p = Path(path)
    if p.suffix == ".csv":
        with open(p, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)
    elif p.suffix == ".json":
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported manifest format: {p.suffix}. Use .csv or .json")


def load_completed(log_path: str) -> set[str]:
    """Load set of already-completed topic hashes from the log."""
    completed = set()
    p = Path(log_path)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
                completed.add(entry.get("topic_hash", ""))
            except json.JSONDecodeError:
                continue
    return completed


def load_title_cache(cache_path: str) -> list[str]:
    """Load previously generated titles to avoid duplicates."""
    p = Path(cache_path)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
    return []


def save_title_cache(titles: list[str], cache_path: str):
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(titles, f, indent=2, ensure_ascii=False)


def topic_hash(row: dict) -> str:
    """Deterministic hash for a manifest row to track completion."""
    key = f"{row.get('topic', '')}|{row.get('prompt', '')}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def row_to_initial_state(row: dict, brand_name: str, brand_context: str, kb_root: str) -> dict:
    """Convert a manifest row to an AgentState initial dict."""
    secondary = row.get("secondary_keywords", "")
    if isinstance(secondary, str):
        secondary = [k.strip() for k in secondary.split(",") if k.strip()]

    seo_kw = row.get("seo_keywords", "")
    if isinstance(seo_kw, str):
        seo_kw = [k.strip() for k in seo_kw.split(",") if k.strip()]

    target_words = int(row.get("target_words", 1500))

    return {
        "topic": row.get("topic", ""),
        "raw_prompt": row.get("prompt", ""),
        "persona": row.get("persona", "A thoughtful writer who shares personal insights and speaks directly to the reader"),
        "example_post": row.get("example_post", ""),
        "target_words_total": target_words,
        "blog_kind_hint": row.get("blog_kind", ""),
        "primary_keyword": row.get("primary_keyword", ""),
        "secondary_keywords": secondary,
        "seo_keywords": seo_kw,
        "kb_root": kb_root,
        "brand_name": brand_name,
        "brand_context": brand_context,
        "ddgs_queries": [],
        "rg_queries": [],
        "ddgs_num_results": 5,
        "errors": [],
        "debug_logs": [],
    }


def write_status_csv(statuses: list[dict], output_dir: str):
    """Write the batch_status.csv tracking every topic's outcome."""
    csv_path = Path(output_dir) / STATUS_CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=STATUS_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(statuses)


def load_brand_context(kb_root: str) -> tuple[str, str]:
    """Load brand context from KB markdown files."""
    kb_path = Path(kb_root).expanduser().resolve()
    if not kb_path.exists():
        return "", ""

    texts = []
    for f in sorted(kb_path.glob("*.md")):
        try:
            texts.append(f.read_text(encoding="utf-8"))
        except Exception:
            pass

    brand_context = "\n\n---\n\n".join(texts) if texts else ""
    return brand_context, str(kb_path)


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run python -m wp_content_engine.batch <manifest.csv|manifest.json> [OPTIONS]")
        sys.exit(1)

    manifest_path = sys.argv[1]
    output_dir = "batch_outputs"
    dry_run = "--dry-run" in sys.argv
    resume = "--resume" in sys.argv

    FLAGS["save_intermediates"] = "--save-intermediates" in sys.argv
    FLAGS["quiet"] = "--quiet" in sys.argv

    for arg in sys.argv[2:]:
        if arg.startswith("--output-dir="):
            output_dir = arg.split("=", 1)[1]

    Path(output_dir).mkdir(exist_ok=True)

    rows = load_manifest(manifest_path)
    print(f"\n Loaded {len(rows)} topics from {manifest_path}")

    if dry_run:
        print("\n DRY RUN — validating manifest:")
        for i, row in enumerate(rows, 1):
            topic = row.get("topic", "")
            prompt = row.get("prompt", "")
            if not topic or not prompt:
                print(f"  [{i}] INVALID — missing topic or prompt: {row}")
            else:
                print(f"  [{i}] OK — {topic[:60]}")
        sys.exit(0)

    completed = load_completed(COMPLETED_LOG) if resume else set()
    prev_titles = load_title_cache(TITLE_CACHE)

    kb_root = os.getenv("KB_ROOT", "")
    brand_context, resolved_kb = load_brand_context(kb_root)
    brand_name = os.getenv("BRAND_NAME", "")

    if not brand_name and brand_context:
        brand_name = input(" Brand name (e.g. 'The Scholars School'): ").strip()

    graph = build_graph()
    app = graph.compile(debug=not FLAGS["quiet"])

    success_count = 0
    fail_count = 0
    skipped_count = 0

    statuses: list[dict] = []

    for i, row in enumerate(rows, 1):
        t_hash = topic_hash(row)
        topic = row.get("topic", "Unknown")

        if resume and t_hash in completed:
            print(f"\n [{i}/{len(rows)}] SKIP (already done): {topic[:60]}")
            skipped_count += 1
            statuses.append({
                "row": i, "topic": topic, "status": "skipped",
                "slug": "", "title": "", "wp_post_url": "",
                "tags": "", "categories": "", "errors": "",
                "timestamp": datetime.now().isoformat(),
            })
            write_status_csv(statuses, output_dir)
            continue

        print(f"\n {'=' * 60}")
        print(f" [{i}/{len(rows)}] {topic[:60]}")
        print(f" {'=' * 60}")

        initial_state = row_to_initial_state(row, brand_name, brand_context, resolved_kb)

        if prev_titles:
            diversity_note = (
                f"Previously generated titles (avoid overlap): "
                f"{', '.join(prev_titles[-20:])}"
            )
            initial_state["raw_prompt"] += f"\n\nDIVERSITY NOTE: {diversity_note}"

        try:
            result = app.invoke(initial_state)

            errors = result.get("errors", [])
            if errors:
                print(f"  WARNINGS: {errors}")

            slug = result.get("seo_slug", f"post-{i}") or f"post-{i}"
            run_dir = Path(output_dir) / slug
            run_dir.mkdir(exist_ok=True)

            wp_tags = result.get("wp_tags", [])
            wp_categories = result.get("wp_categories", [])

            body_md = result.get("wp_body", "")
            body_html = markdown.markdown(body_md, extensions=["extra", "sane_lists"]) if body_md else ""

            wp_payload = {
                "title": result.get("wp_title", ""),
                "subtitle": result.get("wp_subtitle", ""),
                "body": body_html,
                "excerpt": result.get("wp_excerpt", ""),
                "slug": slug,
                "tags": wp_tags,
                "categories": wp_categories,
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

            generated_title = result.get("wp_title", "") or result.get("seo_meta_title", "")
            wp_post_url = result.get("wp_post_url", "")
            if generated_title:
                prev_titles.append(generated_title)
                save_title_cache(prev_titles, TITLE_CACHE)

            log_entry = {
                "topic_hash": t_hash,
                "topic": topic,
                "slug": slug,
                "title": generated_title,
                "timestamp": datetime.now().isoformat(),
                "status": "success",
                "errors": errors,
            }
            with open(COMPLETED_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

            success_count += 1
            print(f"  SUCCESS: {slug}")

            statuses.append({
                "row": i,
                "topic": topic,
                "status": "done",
                "slug": slug,
                "title": generated_title,
                "wp_post_url": wp_post_url,
                "tags": "; ".join(wp_tags),
                "categories": "; ".join(wp_categories),
                "errors": "; ".join(errors) if errors else "",
                "timestamp": datetime.now().isoformat(),
            })

        except Exception as e:
            fail_count += 1
            print(f"  FAILED: {e}")

            log_entry = {
                "topic_hash": t_hash,
                "topic": topic,
                "timestamp": datetime.now().isoformat(),
                "status": "failed",
                "error": str(e),
            }
            with open(COMPLETED_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

            statuses.append({
                "row": i,
                "topic": topic,
                "status": "failed",
                "slug": "",
                "title": "",
                "wp_post_url": "",
                "tags": "",
                "categories": "",
                "errors": str(e),
                "timestamp": datetime.now().isoformat(),
            })

        write_status_csv(statuses, output_dir)

    # Append any manifest rows that were never reached (e.g. early abort)
    processed_rows = {s["row"] for s in statuses}
    for i, row in enumerate(rows, 1):
        if i not in processed_rows:
            statuses.append({
                "row": i,
                "topic": row.get("topic", "Unknown"),
                "status": "pending",
                "slug": "", "title": "", "wp_post_url": "",
                "tags": "", "categories": "", "errors": "",
                "timestamp": "",
            })
    write_status_csv(statuses, output_dir)

    print(f"\n {'=' * 60}")
    print(" BATCH COMPLETE")
    print(f" {'=' * 60}")
    print(f"  Success: {success_count}")
    print(f"  Failed:  {fail_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Total:   {len(rows)}")
    print(f"  Output:  {output_dir}/")
    print(f"  Status:  {output_dir}/{STATUS_CSV}")


if __name__ == "__main__":
    main()
