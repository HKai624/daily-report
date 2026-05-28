import asyncio
import logging
import threading
from typing import Callable, Awaitable, Any

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from src.config import API_RETRY_TIMES, FALLBACK_MODELS, TOOL_CALL_TIMEOUT, MODEL_NAME

thought_logger = logging.getLogger("agent_thought")


@retry(
    stop=stop_after_attempt(API_RETRY_TIMES),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((asyncio.TimeoutError, Exception)),
    reraise=True,
)
async def execute_tool_with_retry(
    tool_func: Callable[..., Awaitable[Any]],
    tool_name: str,
    tool_args: dict,
    timeout: int = TOOL_CALL_TIMEOUT,
) -> Any:
    try:
        result = await asyncio.wait_for(
            tool_func(**tool_args), timeout=timeout
        )
        thought_logger.info(f"工具 {tool_name} 调用成功: {str(result)[:200]}...")
        return result
    except asyncio.TimeoutError:
        thought_logger.warning(f"工具 {tool_name} 调用超时 ({timeout}s)")
        raise
    except Exception as e:
        thought_logger.error(f"工具 {tool_name} 调用失败: {str(e)}")
        raise


def execute_tool_with_retry_sync(
    tool_func: Callable[..., Any],
    tool_name: str,
    tool_args: dict,
    timeout: int = TOOL_CALL_TIMEOUT,
    max_retries: int = API_RETRY_TIMES,
) -> Any:
    """同步版工具重试包装器：超时 + 指数退避重试"""
    last_error = None
    for attempt in range(max_retries):
        result_container = []
        error_container = []

        def target():
            try:
                result_container.append(tool_func(**tool_args))
            except Exception as e:
                error_container.append(e)

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            last_error = TimeoutError(f"工具 {tool_name} 超时 ({timeout}s)")
            thought_logger.warning(
                f"工具 {tool_name} 第{attempt + 1}次尝试超时 ({timeout}s)"
            )
            continue

        if error_container:
            last_error = error_container[0]
            thought_logger.warning(
                f"工具 {tool_name} 第{attempt + 1}次尝试失败: {last_error}"
            )
            if attempt < max_retries - 1:
                import time
                wait = min(2 ** attempt, 10)
                thought_logger.info(f"工具 {tool_name} {wait}s 后重试...")
                time.sleep(wait)
            continue

        thought_logger.info(f"工具 {tool_name} 调用成功: {str(result_container[0])[:200]}")
        return result_container[0]

    raise last_error or Exception(f"工具 {tool_name} 所有重试均失败")


async def call_llm_with_fallback(
    prompt: str,
    llm_call: Callable[..., Awaitable[Any]],
    primary_model: str = MODEL_NAME,
    fallback_models: list = None,
) -> Any:
    """模型 fallback 链：主模型失败后依次尝试备用模型"""
    if fallback_models is None:
        fallback_models = FALLBACK_MODELS

    models_to_try = [primary_model] + [
        m for m in fallback_models if m != primary_model
    ]

    last_error = None
    for model in models_to_try:
        try:
            result = await llm_call(prompt, model=model)
            thought_logger.info(f"模型 {model} 调用成功")
            return result
        except Exception as e:
            thought_logger.warning(f"模型 {model} 调用失败: {str(e)}，尝试备用模型")
            last_error = e
            continue

    raise Exception(f"所有模型调用均失败，最后错误: {last_error}")
