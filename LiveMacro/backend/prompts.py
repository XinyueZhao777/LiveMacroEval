def build_system_msg():
    return (
        "You are a real-time, web-searching economist forecasting macro variables. "
        "Use web search to gather the latest available facts, official releases, trackers, expert consensus, "
        "policy moves, market moves, and news, etc., including revisions. Identify information shocks and update your prediction. "
        "Think silently and return ONLY the requested output format — a single line of space-separated key=value pairs. No extra commentary."
    )


def _format_scale_block(variables):
    lines = []
    for v in variables:
        scale_hint = v.get("scale_hint")
        if scale_hint:
            lines.append(
                f'- {v["key"]}: {scale_hint} (unit: {v["unit_hint"]})'
            )
        else:
            print(f"Warning: No scale_hint provided for variable {v['key']}")
    return "\n".join(lines) if lines else "- (no scale hints provided)"


def _build_example_line(variables):
    example_parts = ["target_month=YYYY-MM", "release_month=YYYY-MM"]
    for v in variables:
        example_parts.append(f'{v["key"]}=VALUE_HERE')
    return " ".join(example_parts)


def build_user_prompt(target_month, release_month, variables):
    """
    target_month: datetime.date for the month the data measures (e.g., CPI for 2025-11)
    release_month: datetime.date for the month it’s officially released (e.g., 2025-12)
    variables: list of dicts with keys: key, title, unit_hint
    """
    month_str_long = target_month.strftime("%B %Y")
    target_month_iso = target_month.strftime("%Y-%m")
    release_month_iso = release_month.strftime("%Y-%m")

    targets_block = "\n".join(
        [f'- "{v["key"]}": {v["title"]} (units: {v["unit_hint"]})' for v in variables]
    )

    sources_block = "\n".join(
        [
            f'- {v["key"]}: {", ".join(v["official_sources"])}'
            for v in variables
        ]
    )
    scale_block = _format_scale_block(variables)
    ini_example = _build_example_line(variables)

    return f"""Goal: Nowcast macro variables for {month_str_long}.
- Target month = {target_month_iso} (month the data measures)
- Release month = {release_month_iso} (month of official publication)
- Use the full, up-to-date set of official and reputable information available as of the system time when this response is generated (including releases, revisions, trackers, consensus, policy/market moves, and relevant news) to produce a live nowcast.
- Forecast all and only the variables defined in Variables & units.
- Geography: United States (U.S.). Use U.S. official releases and reputable consensus trackers for the U.S.

Comparability & whole-month / whole-quarter logic:
- All variables are MONTHLY and measure conditions for the ENTIRE target month, EXCEPT GDP variables.
- GDP variables (real_gdp_level, real_gdp_qoq, real_gdp_yoy) are QUARTERLY and the following rules apply ONLY IF any GDP variable is in Variables & units:
    - If no GDP variables are listed in Variables & units, do NOT include GDP in any form.
    - For GDP, target_month identifies the QUARTER to be forecast, not a monthly value. Each GDP variable refers to the calendar quarter that CONTAINS the target_month (for example, target_month = 2025-12 → 2025Q4).
    - For GDP, release_month is the calendar month in which that quarter’s GDP data are first officially published (for example, 2025Q4 GDP → release_month = 2026-01).
- When updating forecasts, assess how new information affects the full target month. Take into account how much of the month has already elapsed and what remains. Information may continue to arrive after the month has ended (e.g., official statistical releases).

Variables & units: 
You must generate numeric forecasts for ALL variables listed below, every time. No additional variables are permitted, and none may be omitted.
Each variable’s unit is authoritatively defined here and must be strictly followed. If a source reports a variable in a different unit or scale, convert it to the unit specified below before output.
{targets_block}

Official sources:
When provided, the following official sources describe definitions and historical ranges. Use them for grounding and calibration, not verbatim copying.
{sources_block}

Scale guardrails: 
Each variable lists an order-of-magnitude hint in its requested unit. Treat these as acceptable scale guardrails and only go outside them when compelling evidence points to an extreme shock.
{scale_block}

Calibration rules:
- For each variable, use the order-of-magnitude guardrails to sanity-check your draws within the stated unit. Only deviate if strong evidence supports the shift.
- Negative values are explicitly allowed where economically meaningful.
- Never emit placeholder tokens such as \"missing\", \"nan\", \"-999\", \"9999\", or zero values unless zero is a defensible forecast.
- If uncertainty exists, resolve it by choosing your best numeric estimate in the stated units.
- Failure to meet these constraints is considered an incorrect response.

Web search:
- You must pull the most CURRENT information (official/statistical releases, expert consensus, policy decisions, market moves, relevant news, etc).
- If data have been revised, use the most recently published revision.
- Prefer official sources when available. If no official source is available for a variable, rely on web search and standard, reputable research sources.

Output format (STRICT):
- Output exactly one line of space-separated key=value pairs.
- The first two pairs must be target_month=YYYY-MM and release_month=YYYY-MM, in that order, followed by the variables defined in Variables & units.
- Each key must appear exactly once and must EXACTLY match the keys defined in Variables & units.
- Strictly follow each variable’s unit in Variables & units.
- For percent variables, report values as percentages rather than decimals (e.g., 5.2 represents 5.2%).
- Formatting: values must be pure numbers only (no % sign), rounded to at most 2 decimal places. Avoid scientific notation. Never use commas. 

Example (structure only — DO NOT COPY THEM VERBATIM):
{ini_example}

Rules:
- No extra text, no explanation.
- Even if uncertain, provide best estimates (never NA/missing/-999 placeholders).
- Think silently; output ONLY the line.
""".strip()