"""
Claude Code Agent for macroeconomic forecasting.

Uses the Claude Agent SDK through the signed-in Claude Code flow and
explicitly strips API/provider auth from the spawned CLI process so this
path does not consume Anthropic API credits.
"""

import asyncio
import os
from collections import deque
from contextlib import contextmanager, suppress
from pathlib import Path
from threading import Lock

from config import get_logger

from claude_agent_sdk import query, ClaudeAgentOptions, HookMatcher
from claude_agent_sdk.types import ToolUseBlock, ToolResultBlock

logger = get_logger(__name__)

# Pin Sonnet 4.6 explicitly so wakeups do not drift with the moving `sonnet` alias.
MODEL_ID = "claude-sonnet-4-6"
EFFORT_LEVEL = "high"
DISPLAY_NAME = (
    "Claude Code Agent "
    "(claude-sonnet-4-6, effort=high via signed-in Claude Code SDK + financial-analysis plugin)"
)

_LIVEMACRO_ROOT = Path(__file__).resolve().parents[3]
_FINANCIAL_ANALYSIS_PLUGIN_DIR = (
    _LIVEMACRO_ROOT / "financial-services-plugins" / "financial-analysis"
)
_FINANCIAL_ANALYSIS_PLUGIN_MANIFEST = (
    _FINANCIAL_ANALYSIS_PLUGIN_DIR / ".claude-plugin" / "plugin.json"
)
_FINANCIAL_ANALYSIS_MCP_CONFIG = _FINANCIAL_ANALYSIS_PLUGIN_DIR / ".mcp.json"
ALLOWED_TOOLS = ["WebSearch", "WebFetch", "ToolSearch"]
DISALLOWED_TOOLS = [
    "Agent",
    "Bash",
    "Edit",
    "Glob",
    "Grep",
    "LS",
    "MultiEdit",
    "NotebookEdit",
    "NotebookRead",
    "Read",
    "Skill",
    "Task",
    "Write",
]
DEFAULT_AGENT_TIMEOUT_SECS = 1800.0
_AUTH_ENV_LOCK = Lock()
_CLI_AUTH_ENV_VARS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_API_URL",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_REGION",
    "AWS_DEFAULT_REGION",
    "GOOGLE_APPLICATION_CREDENTIALS",
)


@contextmanager
def _signed_in_sdk_only_env():
    """
    Remove provider/API auth from the current process while the SDK launches the
    Claude CLI subprocess. The CLI then falls back to the machine's signed-in
    Claude Code session.
    """

    removed_env = {}
    stripped_names = []

    with _AUTH_ENV_LOCK:
        for name in _CLI_AUTH_ENV_VARS:
            value = os.environ.pop(name, None)
            if value is not None:
                removed_env[name] = value
                stripped_names.append(name)

        if stripped_names:
            logger.warning(
                "Stripped Claude API/provider auth env for SDK agent run: %s",
                ", ".join(sorted(stripped_names)),
            )

        try:
            yield
        finally:
            os.environ.update(removed_env)


def _build_stderr_collector(stderr_lines):
    def _collect(line):
        clean = line.strip()
        if clean:
            stderr_lines.append(clean)

    return _collect


def _is_allowed_tool_name(tool_name):
    return tool_name in ALLOWED_TOOLS or tool_name.startswith("mcp__")


def _tool_denial_reason(tool_name):
    return (
        f"Tool '{tool_name}' is not permitted for macro forecasting runs. "
        f"Only {', '.join(ALLOWED_TOOLS)} and configured MCP tools may be used."
    )


async def _enforce_allowed_tools_pre_use(hook_input, _tool_use_id, _context):
    tool_name = (
        hook_input.get("tool_name", "")
        if isinstance(hook_input, dict)
        else getattr(hook_input, "tool_name", "")
    )

    if _is_allowed_tool_name(tool_name):
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "allow",
            },
            "suppressOutput": True,
        }

    reason = _tool_denial_reason(tool_name)
    logger.error("Blocking forbidden tool before execution: %s", tool_name)
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        },
        "reason": reason,
    }


def _build_agent_options(stderr_lines):
    option_kwargs = {
        "model": MODEL_ID,
        "effort": EFFORT_LEVEL,
        "tools": ALLOWED_TOOLS,
        "allowed_tools": ALLOWED_TOOLS,
        "disallowed_tools": DISALLOWED_TOOLS,
        "hooks": {
            "PreToolUse": [
                HookMatcher(
                    matcher=".*",
                    hooks=[_enforce_allowed_tools_pre_use],
                    timeout=5.0,
                )
            ]
        },
        "permission_mode": "bypassPermissions",
        "cwd": str(_LIVEMACRO_ROOT),
        "stderr": _build_stderr_collector(stderr_lines),
    }

    if _FINANCIAL_ANALYSIS_MCP_CONFIG.is_file():
        option_kwargs["mcp_servers"] = _FINANCIAL_ANALYSIS_MCP_CONFIG
        logger.info(
            "Financial analysis MCP servers loaded from %s",
            _FINANCIAL_ANALYSIS_MCP_CONFIG,
        )
    else:
        logger.warning(
            "Financial analysis MCP config not found: %s",
            _FINANCIAL_ANALYSIS_MCP_CONFIG,
        )

    if _FINANCIAL_ANALYSIS_PLUGIN_MANIFEST.is_file():
        option_kwargs["plugins"] = [
            {"type": "local", "path": str(_FINANCIAL_ANALYSIS_PLUGIN_DIR)}
        ]
        logger.info(
            "Financial analysis local plugin enabled from %s",
            _FINANCIAL_ANALYSIS_PLUGIN_DIR,
        )
    else:
        logger.warning(
            "Financial analysis plugin manifest not found: %s",
            _FINANCIAL_ANALYSIS_PLUGIN_MANIFEST,
        )

    return ClaudeAgentOptions(**option_kwargs)


def _get_agent_timeout_secs():
    raw_value = os.getenv("CLAUDE_CODE_AGENT_TIMEOUT_SECS", str(DEFAULT_AGENT_TIMEOUT_SECS))
    try:
        timeout_secs = float(raw_value)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid CLAUDE_CODE_AGENT_TIMEOUT_SECS=%r; using default %.1fs",
            raw_value,
            DEFAULT_AGENT_TIMEOUT_SECS,
        )
        timeout_secs = DEFAULT_AGENT_TIMEOUT_SECS
    return max(timeout_secs, 1.0)


async def _run_agent(system_msg, user_msg):
    """
    Run the Claude Agent SDK query (query + ClaudeAgentOptions + async for).
    """

    # Build the agent prompt: system instructions + user task
    prompt = (
        f"{system_msg}\n\n"
        f"{user_msg}\n\n"
        "AGENT INSTRUCTIONS:\n"
        "You are an autonomous macro forecasting agent. Follow these steps:\n"
        "1. Use only the tools allowed in this run. Use WebSearch and WebFetch to find "
        "the LATEST data from official or otherwise reliable public sources.\n"
        "2. If a source returns an error or repeated timeouts, do not keep retrying "
        "the same source. Switch promptly to another official or reliable public "
        "source.\n"
        "3. The financial-analysis plugin and its MCP tools are enabled and can be used "
        "if they are useful for relevant data, context, or cross-checks.\n"
        "4. Do not request, call, or rely on Bash, file-edit, or local filesystem "
        "tools. If a source is inaccessible with the allowed tools, continue with the "
        "best accessible source instead of attempting other tools.\n"
        "5. After gathering all data, produce your output in the EXACT format specified — "
        "a single line of space-separated key=value pairs. No extra text.\n"
        "6. The LAST line of your response MUST be the key=value output line."
    )

    result_text = ""
    stderr_lines = deque(maxlen=40)

    # Track tool usage for summary
    tools_used = []        # list of (tool_name, input_summary)
    mcp_tools_used = []    # list of (server, function, input_summary)

    options = _build_agent_options(stderr_lines)
    timeout_secs = _get_agent_timeout_secs()

    async def _consume_query_stream():
        stream = query(
            prompt=prompt,
            options=options,
        )
        try:
            async for message in stream:
                if hasattr(message, "result"):
                    nonlocal_result = message.result
                    logger.info("Agent completed with result (len=%d)", len(nonlocal_result))
                    nonlocal_state["result_text"] = nonlocal_result
                elif hasattr(message, "content"):
                    # Inspect each content block for tool usage
                    blocks = message.content if isinstance(message.content, list) else [message.content]
                    for block in blocks:
                        if isinstance(block, ToolUseBlock):
                            tool_name = block.name
                            tool_input = str(block.input)[:150]

                            if not _is_allowed_tool_name(tool_name):
                                logger.error(
                                    "Forbidden tool detected in agent stream: %s input=%s",
                                    tool_name,
                                    tool_input,
                                )
                                raise PermissionError(_tool_denial_reason(tool_name))

                            if tool_name.startswith("mcp__"):
                                parts = tool_name.split("__", 2)
                                server = parts[1] if len(parts) > 1 else "unknown"
                                func = parts[2] if len(parts) > 2 else "unknown"
                                mcp_tools_used.append((server, func, tool_input))
                                logger.info(
                                    "Agent MCP tool call: server=%s func=%s input=%s",
                                    server,
                                    func,
                                    tool_input,
                                )
                            else:
                                tools_used.append((tool_name, tool_input))
                                logger.info("Agent tool call: %s input=%s", tool_name, tool_input)

                        elif isinstance(block, ToolResultBlock):
                            if block.is_error:
                                logger.warning(
                                    "Agent tool error (tool_use_id=%s): %s",
                                    block.tool_use_id,
                                    str(block.content)[:200],
                                )
        finally:
            with suppress(Exception):
                await stream.aclose()

    try:
        nonlocal_state = {"result_text": result_text}
        with _signed_in_sdk_only_env():
            await asyncio.wait_for(_consume_query_stream(), timeout=timeout_secs)
        result_text = nonlocal_state["result_text"]
    except asyncio.TimeoutError as exc:
        result_text = nonlocal_state["result_text"]
        stderr_summary = " | ".join(stderr_lines) if stderr_lines else ""
        if result_text:
            logger.warning(
                "Claude Code Agent timed out after emitting a result; discarding it so forecasting can retry. timeout=%.1fs",
                timeout_secs,
            )
        else:
            logger.error("Claude Code Agent timed out after %.1fs", timeout_secs)
        if stderr_summary:
            logger.error("Claude CLI stderr before timeout: %s", stderr_summary)
            raise TimeoutError(
                f"Claude Code Agent timed out after {timeout_secs:.1f}s\n"
                f"Claude CLI stderr: {stderr_summary}"
            ) from exc
        raise TimeoutError(
            f"Claude Code Agent timed out after {timeout_secs:.1f}s"
        ) from exc
    except Exception as exc:
        result_text = nonlocal_state["result_text"]
        stderr_summary = " | ".join(stderr_lines) if stderr_lines else ""
        if result_text:
            logger.warning(
                "Claude SDK raised after returning a result; using returned result. error=%s",
                exc,
            )
            if stderr_summary:
                logger.warning("Claude CLI stderr before exit: %s", stderr_summary)
        else:
            if stderr_summary:
                logger.error("Claude CLI stderr before failure: %s", stderr_summary)
                raise RuntimeError(f"{exc}\nClaude CLI stderr: {stderr_summary}") from exc
            raise

    # Summary: did the agent use MCP financial-analysis plugin?
    if mcp_tools_used:
        servers_hit = sorted(set(s for s, _, _ in mcp_tools_used))
        logger.info("FINANCIAL PLUGIN USED: %d MCP calls across servers: %s",
                     len(mcp_tools_used), ", ".join(servers_hit))
        for server, func, inp in mcp_tools_used:
            logger.info("  MCP detail: %s.%s(%s)", server, func, inp[:80])
    else:
        logger.info("FINANCIAL PLUGIN NOT USED: 0 MCP tool calls in this run")

    # Log all tools summary
    tool_counts = {}
    for name, _ in tools_used:
        tool_counts[name] = tool_counts.get(name, 0) + 1
    if tool_counts:
        summary = ", ".join(f"{k}={v}" for k, v in sorted(tool_counts.items()))
        logger.info("Agent tool usage summary: %s", summary)

    return result_text


def generate(system_msg, user_msg):
    """
    Call Claude Code Agent for macro forecasting.

    Interface matches other LLM clients:
        generate(system_msg, user_msg) -> (text, citations)
    """
    logger.info("Calling Claude Code Agent model=%s (%s) ...", DISPLAY_NAME, MODEL_ID)

    result = asyncio.run(_run_agent(system_msg, user_msg))

    if not result:
        raise RuntimeError("Claude Code Agent returned empty result")

    logger.info("Claude Code Agent response length: %d chars", len(result))
    return result, None
