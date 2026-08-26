from typing import Callable

def get_client(model_name):
    if model_name == "gpt-5-search-api":
        from .gpt_5_search_api import generate
        return generate

    elif model_name == "claude-sonnet-4.5-openrouter":
        # OpenRouter Claude with native web search
        from .openrouter import generate
        return generate

    elif model_name == "claude-sonnet-4.5-api":
        # Claude API with web search tool
        from .claude_sonnet_4_5_api import generate
        return generate

    elif model_name == "claude-code-agent":
        # Claude Code Agent SDK with web search + financial-analysis plugin
        from .claude_code_agent import generate
        return generate

    else:
        raise ValueError(f"Unknown model: {model_name}")
