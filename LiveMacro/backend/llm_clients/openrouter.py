"""
OpenRouter Claude with native web search.

Configuration:
- Plugin: {"id": "web", "engine": "native"}
- Web search options: {"search_context_size": ...}
- No reasoning (reasoning disabled)
"""

from openai import OpenAI

from config import OPENROUTER_API_KEY, get_logger

logger = get_logger(__name__)

# Web search configuration (exactly same as test)
# lower websearch effort -- testing
# still not working -- stopped
WEB_SEARCH_PLUGIN = {"id": "web", "engine": "native", "max_results": 5}
WEB_SEARCH_OPTIONS = {"search_context_size": "low"}

# Model configuration
MODEL_ID = "anthropic/claude-sonnet-4.5"
DISPLAY_NAME = "Claude Sonnet 4.5 (OpenRouter + native web search)"


def generate(system_msg, user_msg):
    """
    Call OpenRouter Claude with native web search.

    Args:
        system_msg: System prompt
        user_msg: User prompt

    Returns:
        Tuple of (text_response, citations)
    """
    logger.info(f"Calling OpenRouter LLM model={DISPLAY_NAME} ({MODEL_ID}) ...")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )

    extra_body = {
        "plugins": [WEB_SEARCH_PLUGIN],
        "web_search_options": WEB_SEARCH_OPTIONS,
    }

    resp = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        extra_body=extra_body,
    )

    text = resp.choices[0].message.content
    citations = None
    return str(text), citations
