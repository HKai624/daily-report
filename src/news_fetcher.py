"""AI 新闻抓取模块：Hacker News + ArXiv，免费无需 API Key"""

import logging
import requests
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

AI_KEYWORDS = [
    "AI", "LLM", "GPT", "Claude", "Gemini", "machine learning",
    "deep learning", "transformer", "OpenAI", "Anthropic", "DeepMind",
    "neural network", "language model", "diffusion", "RLHF", "RAG",
    "agent", "copilot", "chatgpt", "llama", "mistral", "groq",
]


def _is_ai_related(title: str) -> bool:
    lower = title.lower()
    return any(kw.lower() in lower for kw in AI_KEYWORDS)


def fetch_hacker_news_ai(max_items: int = 5) -> list[dict]:
    """从 Hacker News (Algolia API) 抓取 AI 相关热门文章"""
    try:
        url = "https://hn.algolia.com/api/v1/search_by_date"
        params = {
            "query": "AI OR LLM OR GPT OR machine learning OR OpenAI",
            "tags": "story",
            "hitsPerPage": max_items * 3,
        }
        resp = requests.get(url, params=params, timeout=15)
        data = resp.json()
        hits = data.get("hits", [])

        results = []
        for hit in hits:
            title = hit.get("title", "") or hit.get("story_title", "")
            link = hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID')}"
            if title and _is_ai_related(title) and link:
                results.append({
                    "title": title.strip(),
                    "link": link,
                    "published": hit.get("created_at", ""),
                    "source": "Hacker News",
                })
            if len(results) >= max_items:
                break

        logger.info(f"Hacker News: 获取 {len(results)} 条 AI 相关文章")
        return results
    except Exception as e:
        logger.error(f"Hacker News 抓取失败: {e}")
        return []


def fetch_arxiv_ai(max_items: int = 5) -> list[dict]:
    """从 ArXiv cs.AI 分类抓取最新论文"""
    try:
        url = "http://export.arxiv.org/api/query"
        params = {
            "search_query": "cat:cs.AI",
            "sortBy": "submittedDate",
            "start": 0,
            "max_results": max_items,
        }
        resp = requests.get(url, params=params, timeout=20)
        root = ET.fromstring(resp.text)

        ns = {"a": "http://www.w3.org/2005/Atom"}
        results = []

        for entry in root.findall("a:entry", ns):
            title_el = entry.find("a:title", ns)
            link_el = entry.find("a:id", ns)
            published_el = entry.find("a:published", ns)

            title_text = title_el.text.strip().replace("\n", " ") if title_el is not None and title_el.text else ""
            link_text = link_el.text.strip() if link_el is not None and link_el.text else ""
            published_text = published_el.text.strip() if published_el is not None and published_el.text else ""

            if title_text and link_text:
                results.append({
                    "title": title_text,
                    "link": link_text,
                    "published": published_text,
                    "source": "ArXiv cs.AI",
                })

        logger.info(f"ArXiv: 获取 {len(results)} 篇论文")
        return results
    except Exception as e:
        logger.error(f"ArXiv 抓取失败: {e}")
        return []


def fetch_all_news(max_total: int = 5) -> list[dict]:
    """从全部新闻源抓取并合并，按时间排序，取前 max_total 条"""
    all_news = []

    for fetcher in [fetch_hacker_news_ai, fetch_arxiv_ai]:
        try:
            items = fetcher(max_items=3)
            all_news.extend(items)
        except Exception:
            continue

    all_news.sort(key=lambda x: x.get("published", ""), reverse=True)
    return all_news[:max_total]
