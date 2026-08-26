"""
Claude API with web search tool.

Configuration:
- Tool type: web_search_20250305
- Tool name: web_search
- Citations are always enabled (cannot disable)

Notes from Anthropic docs:
- Web search is SERVER-SIDE (API executes searches automatically)
- No tool loop needed - different from regular tool use
- Response includes server_tool_use blocks (not tool_use)
"""

import anthropic
from anthropic import APIError, APIConnectionError, RateLimitError

from config import ANTHROPIC_API_KEY, get_logger

logger = get_logger(__name__)

# Model configuration
MODEL_ID = "claude-sonnet-4-5-20250929"
DISPLAY_NAME = "Claude Sonnet 4.5 (Claude API + web search tool)"
MAX_TOKENS = 4096


def generate(system_msg, user_msg):
    """
    Call Claude API with web search tool.

    Args:
        system_msg: System prompt
        user_msg: User prompt

    Returns:
        Tuple of (text_response, None)
        
    Raises:
        APIError: For API-related errors
        Exception: For other unexpected errors
    """
    logger.info(f"Calling Claude API model={DISPLAY_NAME} ({MODEL_ID}) ...")

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        # Web search tool definition. user_location is intentionally
        # generic; consumers can refine to any U.S. region they prefer.
        web_search_tool = {
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": 15,
            "user_location": {
                "type": "approximate",
                "country": "US",
                "timezone": "America/New_York",
            },
        }

        messages = [{"role": "user", "content": user_msg}]
        
        response = client.messages.create(
            model=MODEL_ID,
            max_tokens=MAX_TOKENS,
            system=system_msg,
            messages=messages,
            tools=[web_search_tool],
        )

        # Handle pause_turn for long-running searches
        # COMMENTED OUT: This causes additional API calls when searches take longer
        # if response.stop_reason == "pause_turn":
        #     logger.info("Long-running search paused, continuing turn...")
        #     messages.append({"role": "assistant", "content": response.content})
        #
        #     response = client.messages.create(
        #         model=MODEL_ID,
        #         max_tokens=MAX_TOKENS,
        #         system=system_msg,
        #         messages=messages,
        #         tools=[web_search_tool],
        #     )

        # Check for other stop reasons
        if response.stop_reason == "max_tokens":
            logger.warning("Response truncated due to max_tokens limit")
        elif response.stop_reason not in ["end_turn", "max_tokens", "pause_turn"]:
            logger.warning(f"Unexpected stop_reason: {response.stop_reason}")

        # Check for web search errors in response blocks
        search_errors = []
        for block in response.content:
            if block.type == "web_search_tool_result":
                # Check if this is an error result
                if hasattr(block, 'content') and isinstance(block.content, dict):
                    if block.content.get('type') == 'web_search_tool_result_error':
                        error_code = block.content.get('error_code')
                        search_errors.append(error_code)
                        logger.error(f"Web search error: {error_code}")

        if search_errors:
            logger.warning(f"Web search encountered {len(search_errors)} error(s): {search_errors}")

        # Extract text content only
        text_output = []
        for block in response.content:
            if block.type == "text":
                text_output.append(block.text)

        text = "".join(text_output)
        
        # Log search usage
        if hasattr(response, 'usage'):
            logger.info(f"Token usage - Input: {response.usage.input_tokens}, "
                       f"Output: {response.usage.output_tokens}")
            
            if hasattr(response.usage, 'server_tool_use'):
                server_usage = response.usage.server_tool_use
                if isinstance(server_usage, dict):
                    searches = server_usage.get('web_search_requests', 0)
                else:
                    searches = getattr(server_usage, 'web_search_requests', 0)
                logger.info(f"Web searches performed: {searches}")

        return text, None

    except RateLimitError as e:
        logger.error(f"Rate limit exceeded: {e}")
        raise
    
    except APIConnectionError as e:
        logger.error(f"API connection error: {e}")
        raise
    
    except APIError as e:
        logger.error(f"API error: {e}")
        raise
    
    except Exception as e:
        logger.error(f"Unexpected error in generate(): {e}")
        raise