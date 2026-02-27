#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# From: https://github.com/nkapila6/mcp-local-rag/blob/main/src/mcp_local_rag/utils/fetch.py
"""
Created on 2025-06-04 22:55:29 Wednesday

@author: Nikhil Kapila
"""

import re
import requests
import time

from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup, Tag


BOILERPLATE_TAGS = {"nav", "header", "footer", "aside", "form", "script", "style", "noscript"}
BOILERPLATE_ROLES = {"navigation", "banner", "contentinfo", "complementary", "form"}
BOILERPLATE_CLASSES = re.compile(
    r"nav|menu|sidebar|footer|header|cookie|banner|popup|modal|advert|social|share|comment",
    re.I,
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def _extract_main_content(soup: BeautifulSoup) -> str:
    """Extract main article content, stripping nav/header/footer boilerplate."""
    try:
        for tag in list(soup.find_all(BOILERPLATE_TAGS)):
            tag.decompose()

        for tag in list(soup.find_all(True, attrs={"role": True})):
            if isinstance(tag, Tag) and tag.get("role") in BOILERPLATE_ROLES:
                tag.decompose()

        for tag in list(soup.find_all(True, class_=True)):
            if isinstance(tag, Tag):
                classes = " ".join(tag.get("class", []))
                if BOILERPLATE_CLASSES.search(classes):
                    tag.decompose()
    except Exception:
        pass

    main = soup.find("main") or soup.find("article") or soup.find("body") or soup
    text = main.get_text(separator=" ", strip=True)
    text = re.sub(r"\s{3,}", "  ", text)
    return text


def fetch_content(url: str, timeout: int = 8) -> Optional[str]:
    """Fetch content from a URL with timeout, stripping boilerplate."""
    try:
        start_time = time.time()
        response = requests.get(url, headers=HEADERS, timeout=timeout)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        content = _extract_main_content(soup)
        print(f"Fetched {url} in {time.time() - start_time:.2f}s")
        return content[:5000] if content else None
    except Exception as e:
        print(f"Error fetching {url}: {type(e).__name__} - {str(e)}")
        return None

def fetch_all_content(results: List[Dict], include_urls:bool=True) -> List[str]:
    """Fetch content from all URLs using a thread pool."""
    urls = [site['href'] for site in results if site.get('href')]
    
    # parallelize requests
    with ThreadPoolExecutor(max_workers=5) as executor:
        # submit fetch tasks to executor
        future_to_url = {executor.submit(fetch_content, url): url for url in urls}
        
        content_list = []
        for future, url in future_to_url.items():
            try:
                content = future.result()
                if content:
                    result = {
                        "type": "text",
                        "text": content,
                    }
                    if include_urls:
                        result["url"] = url
                    content_list.append(result)
            except Exception as e:
                print(f"Request failed with exception: {e}")
        
    return content_list
