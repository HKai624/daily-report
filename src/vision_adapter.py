import json
import logging
import time
from typing import Optional

import requests

from src.config import ZHIPU_API_KEY, VISION_MODEL, VISION_ENDPOINT, API_RETRY_TIMES

thought_logger = logging.getLogger("agent_thought")


def analyze_image(image_url: str, prompt: str = "") -> str:
    """调用智谱 GLM-4.6V API，分析图片内容。

    Args:
        image_url: 图片的公网可访问 URL（不支持本地路径和 Base64）。
        prompt: 可选的额外分析指令（拼接到默认 prompt 后面）。

    Returns:
        模型返回的图片分析结果字符串。
    """
    if not prompt:
        prompt = "请详细描述这张图片的内容，包括：1) 图片类型（截图/照片/图表/文档等）；2) 图片中的所有文字信息；3) 图片传达的主要信息或意图。"
    headers = {
        "Authorization": f"Bearer {ZHIPU_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": VISION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }

    last_error = None
    for attempt in range(API_RETRY_TIMES):
        try:
            resp = requests.post(
                VISION_ENDPOINT,
                headers=headers,
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            thought_logger.info(
                f"智谱视觉分析成功: {len(content)} chars, "
                f"usage={data.get('usage', {})}"
            )
            return content
        except requests.exceptions.Timeout as e:
            last_error = e
            thought_logger.warning(
                f"智谱 API 超时 (attempt {attempt + 1}/{API_RETRY_TIMES})"
            )
        except requests.exceptions.RequestException as e:
            last_error = e
            thought_logger.warning(
                f"智谱 API 请求失败 (attempt {attempt + 1}/{API_RETRY_TIMES}): {e}"
            )
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            last_error = e
            thought_logger.error(f"智谱 API 响应解析失败: {e}")
            break

    raise last_error or Exception("智谱 API 所有重试均失败")


def handle_message_with_image(user_message: str, image_url: str) -> str:
    """接收用户文本和图片 URL，先调用智谱提取图片内容，再构造增强消息。

    注意：此函数只负责构造增强后的消息文本，不调用 DeepSeek。
    实际推理由后续的 LangGraph 管线完成。

    Args:
        user_message: 用户的原始文本消息。
        image_url: 图片的公网可访问 URL。

    Returns:
        拼接了图片内容的增强消息文本。
    """
    try:
        image_description = analyze_image(image_url)
    except Exception as e:
        thought_logger.error(f"图片分析失败，使用降级提示: {e}")
        return (
            f"[系统提示] 用户发送了一张图片，但图片分析服务暂时不可用。\n"
            f"用户说：'{user_message}'。\n"
            f"请告知用户图片识别暂时不可用，并请用户直接描述图片内容。"
        )

    thought_logger.info(
        f"图片内容已提取: {image_description[:100]}..."
    )

    enhanced_message = (
        f"[系统提示] 用户发送了一张图片，图片分析结果如下：\n"
        f"---\n{image_description}\n---\n"
        f"用户对图片的说明：'{user_message}'。\n"
        f"请结合图片分析结果，回答用户的问题或执行用户请求的操作。"
    )
    return enhanced_message
