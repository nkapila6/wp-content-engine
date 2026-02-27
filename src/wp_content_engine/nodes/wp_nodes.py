# WordPress REST API push node
# 26/02/2026 Nikhil Kapila

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import markdown
import requests

from wp_content_engine.state.state import AgentState


def _md_to_html(text: str) -> str:
    """Convert markdown body to WordPress-friendly HTML."""
    if not text:
        return text
    return markdown.markdown(text, extensions=["extra", "sane_lists"])


def _resolve_tags(
    tag_names: List[str],
    wp_url: str,
    auth: Tuple[str, str],
) -> List[int]:
    """Create-or-find WP tags by name and return their IDs."""
    tag_ids = []
    tags_endpoint = f"{wp_url}/wp-json/wp/v2/tags"

    for name in tag_names:
        name = name.strip()
        if not name:
            continue

        resp = requests.get(
            tags_endpoint,
            params={"search": name, "per_page": 1},
            auth=auth,
            timeout=10,
        )
        if resp.ok and resp.json():
            for existing in resp.json():
                if existing["name"].lower() == name.lower():
                    tag_ids.append(existing["id"])
                    break
            else:
                create = requests.post(
                    tags_endpoint,
                    json={"name": name},
                    auth=auth,
                    timeout=10,
                )
                if create.ok:
                    tag_ids.append(create.json()["id"])
        else:
            create = requests.post(
                tags_endpoint,
                json={"name": name},
                auth=auth,
                timeout=10,
            )
            if create.ok:
                tag_ids.append(create.json()["id"])

    return tag_ids


def _resolve_category(
    category_name: str,
    wp_url: str,
    auth: Tuple[str, str],
) -> List[int]:
    """Create-or-find a WP category by name and return its ID as a list."""
    if not category_name:
        return []

    cats_endpoint = f"{wp_url}/wp-json/wp/v2/categories"

    resp = requests.get(
        cats_endpoint,
        params={"search": category_name, "per_page": 5},
        auth=auth,
        timeout=10,
    )
    if resp.ok:
        for cat in resp.json():
            if cat["name"].lower() == category_name.lower():
                return [cat["id"]]

    create = requests.post(
        cats_endpoint,
        json={"name": category_name},
        auth=auth,
        timeout=10,
    )
    if create.ok:
        return [create.json()["id"]]

    return []


def fetch_existing_posts() -> List[Dict[str, str]]:
    """
    Paginate through the WP REST API and return every post's title, slug,
    and excerpt.  Works for both 'publish' and 'draft' statuses so
    autopilot can avoid duplicating content in either state.

    Returns an empty list when WP credentials are not configured.
    """
    wp_url = os.getenv("WP_URL", "").rstrip("/")
    wp_user = os.getenv("WP_USER", "")
    wp_password = os.getenv("WP_APP_PASSWORD", "")

    if not wp_url or not wp_user or not wp_password:
        return []

    auth = (wp_user, wp_password)
    endpoint = f"{wp_url}/wp-json/wp/v2/posts"
    posts: List[Dict[str, str]] = []
    page = 1

    while True:
        resp = requests.get(
            endpoint,
            params={
                "per_page": 100,
                "page": page,
                "status": "publish,draft",
                "_fields": "id,title,slug,excerpt",
            },
            auth=auth,
            timeout=15,
        )
        if not resp.ok or not resp.json():
            break

        for item in resp.json():
            posts.append({
                "id": item["id"],
                "title": item.get("title", {}).get("rendered", ""),
                "slug": item.get("slug", ""),
                "excerpt": item.get("excerpt", {}).get("rendered", ""),
            })

        total_pages = int(resp.headers.get("X-WP-TotalPages", 1))
        if page >= total_pages:
            break
        page += 1

    return posts


def wp_push_node(state: AgentState, *_args, **_kwargs) -> Dict[str, object]:
    """
    Node: push the final article to WordPress via REST API.

    Uses only native WP fields: title, content, excerpt, slug, tags, categories.
    Keywords are pushed as WP Tags. Skips silently when WP_URL is not configured.
    """
    wp_url = os.getenv("WP_URL", "").rstrip("/")
    wp_user = os.getenv("WP_USER", "")
    wp_password = os.getenv("WP_APP_PASSWORD", "")
    wp_status = os.getenv("WP_POST_STATUS", "draft")
    wp_category = os.getenv("WP_DEFAULT_CATEGORY", "")

    if not wp_url or not wp_user or not wp_password:
        return {}

    auth = (wp_user, wp_password)

    title = state.get("wp_title", "") or state.get("seo_meta_title", "")
    body_md = state.get("wp_body", "") or state.get("styled_draft", "")
    body = _md_to_html(body_md)
    excerpt = state.get("wp_excerpt", "") or state.get("seo_meta_description", "")
    slug = state.get("seo_slug", "")

    if not body:
        errors = list(state.get("errors", []))
        errors.append("wp_push_node: no body content to publish")
        return {"errors": errors}

    wp_tag_names = state.get("wp_tags", [])
    wp_cat_names = state.get("wp_categories", [])

    tag_ids = _resolve_tags(wp_tag_names, wp_url, auth)

    cat_ids = []
    for cat_name in wp_cat_names:
        cat_ids.extend(_resolve_category(cat_name, wp_url, auth))
    if not cat_ids and wp_category:
        cat_ids = _resolve_category(wp_category, wp_url, auth)

    endpoint = f"{wp_url}/wp-json/wp/v2/posts"

    payload = {
        "title": title,
        "content": body,
        "excerpt": excerpt,
        "slug": slug,
        "status": wp_status,
        "tags": tag_ids,
    }
    if cat_ids:
        payload["categories"] = cat_ids

    try:
        resp = requests.post(
            endpoint,
            json=payload,
            auth=auth,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        post_id = data.get("id")
        post_url = data.get("link", "")

        print(f"  WP post created: ID={post_id} URL={post_url} status={wp_status}")
        print(f"  Tags: {len(tag_ids)} assigned, Category: {cat_ids or 'default'}")

        return {
            "wp_post_id": post_id,
            "wp_post_url": post_url,
        }
    except requests.RequestException as e:
        errors = list(state.get("errors", []))
        errors.append(f"wp_push_node error: {e}")
        return {"errors": errors}
