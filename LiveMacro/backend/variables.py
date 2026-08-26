from typing import List, Dict

VARIABLES = {"core_macroeconomic_conditions": [
        # ============================================================
        # 1. OUTPUT / PRODUCTION (4 VARIABLES)
        # ============================================================
        # 1.1 Real GDP
        {
            "key": "real_gdp_level",
            "title": "Real Gross Domestic Product (level, billions of chained 2017 dollars, SAAR)",
            "unit_hint": "billion_chained_2017_usd_saar",
            "official_sources": [
                "https://fred.stlouisfed.org/series/GDPC1",
            ],
        },
        {
            "key": "real_gdp_qoq",
            "title": "Real Gross Domestic Product (QoQ %, SAAR)",
            "unit_hint": "percent",
            "official_sources": [
                "https://fred.stlouisfed.org/series/A191RL1Q225SBEA",
            ],
        },
        {
            "key": "real_gdp_yoy",
            "title": "Real Gross Domestic Product (YoY %, i.e., Percent Change from Quarter One Year Ago)",
            "unit_hint": "percent",
            "official_sources": ["https://fred.stlouisfed.org/series/A191RO1Q156NBEA"],
        },
        # 1.2 Industrial Production
        {
            "key": "industrial_production_index",
            "title": "Industrial Production Index (level, 2017=100, SA)",
            "unit_hint": "index_2017eq100",
            "official_sources": [
                "https://www.federalreserve.gov/releases/g17/current/",
            ],
        },
        {
            "key": "industrial_production_mom",
            "title": "Industrial Production Index (MoM %, SA)",
            "unit_hint": "percent",
            "official_sources": [
                "https://www.federalreserve.gov/releases/g17/current/",
            ],
        },
        {
            "key": "industrial_production_yoy",
            "title": "Industrial Production Index (YoY %, SA, i.e., Percent Change from Month One Year Ago)",
            "unit_hint": "percent",
            "official_sources": [
                "https://www.federalreserve.gov/releases/g17/current/",
            ],
        },
        # 1.3 Durable Goods Orders
        {
            "key": "durable_goods_level",
            "title": "Manufacturers' New Orders: Durable Goods (level, SA, billions of USD)",
            "unit_hint": "billion_usd",
            "official_sources": [
                "https://fred.stlouisfed.org/series/DGORDER",
            ],
        },
        {
            "key": "durable_goods_mom",
            "title": "Manufacturers' New Orders: Durable Goods (MoM %, SA)",
            "unit_hint": "percent",
            "official_sources": [
                "https://fred.stlouisfed.org/graph/?g=1OQEM",
            ],
        },
        {
            "key": "durable_goods_yoy",
            "title": "Manufacturers' New Orders: Durable Goods (YoY %, i.e., Percent Change from Year Ago)",
            "unit_hint": "percent",
            "official_sources": ["https://fred.stlouisfed.org/graph/?g=1OQEO"],
        },
        # 1.4 ISM Manufacturing PMI
        {
            "key": "ism_manufacturing_index",
            "title": "ISM Manufacturing PMI (index level, SA)",
            "unit_hint": "index",
            "official_sources": [
                "https://www.ismworld.org/supply-management-news-and-reports/reports/ism-report-on-business/",
            ],
        },
        {
            "key": "ism_manufacturing_mom",
            "title": "ISM Manufacturing PMI (MoM Percentage Point Change, index points)",
            "unit_hint": "index_points",
            "official_sources": [
                "https://www.ismworld.org/supply-management-news-and-reports/reports/ism-report-on-business/",
            ],
        },
        # ============================================================
        # 2. INFLATION (3 VARIABLES)
        # ============================================================
        # 2.1 CPI
        {
            "key": "cpi_index",
            "title": "CPI: All Items (index level, 1982-84=100, SA)",
            "unit_hint": "index_1982_84eq100",
            "official_sources": [
                "https://fred.stlouisfed.org/series/CPIAUCSL",
            ],
        },
        {
            "key": "cpi_mom",
            "title": "CPI: All Items (MoM %, SA)",
            "unit_hint": "percent",
            "official_sources": [
                "https://fred.stlouisfed.org/graph/?g=1OlYu",
            ],
        },
        {
            "key": "cpi_yoy",
            "title": "CPI: All Items (YoY %, i.e., Percent Change from Year Ago)",
            "unit_hint": "percent",
            "official_sources": [
                "https://fred.stlouisfed.org/graph/?g=1NqEP",
            ],
        },
        # 2.2 PPI
        {
            "key": "ppi_index",
            "title": "Producer Price Index: Final Demand (index level, 1982=100, NSA)",
            "unit_hint": "index_1982eq100",
            "official_sources": [
                "https://fred.stlouisfed.org/series/PPIACO",
            ],
        },
        {
            "key": "ppi_mom",
            "title": "Producer Price Index: Final Demand (MoM %, NSA)",
            "unit_hint": "percent",
            "official_sources": [
                "https://fred.stlouisfed.org/graph/?g=1OQFK",
            ],
        },
        {
            "key": "ppi_yoy",
            "title": "Producer Price Index: Final Demand (YoY %, i.e., Percent Change from Year Ago, NSA)",
            "unit_hint": "percent",
            "official_sources": [
                "https://fred.stlouisfed.org/graph/?g=1OQFQ",
            ],
        },
        # 2.3 PCE Price Index (BEA)
        {
            "key": "pce_price_index_level",
            "title": "Personal Consumption Expenditures Price Index (level, 2017=100, SA)",
            "unit_hint": "index_2017eq100",
            "official_sources": [
                "https://fred.stlouisfed.org/series/PCEPI",
            ],
        },
        {
            "key": "pce_price_index_mom",
            "title": "Personal Consumption Expenditures Price Index (MoM %, SA)",
            "unit_hint": "percent",
            "official_sources": [
                "https://fred.stlouisfed.org/graph/?g=1OQFU",
            ],
        },
        {
            "key": "pce_price_index_yoy",
            "title": "Personal Consumption Expenditures Price Index (YoY %, i.e., Percent Change from Month One Year Ago)",
            "unit_hint": "percent",
            "official_sources": ["https://fred.stlouisfed.org/graph/?g=1OQFW"],
        },
        # ============================================================
        # 4. LABOR MARKET (2 VARIABLES)
        # ============================================================
        # 4.1 Nonfarm Payrolls
        {
            "key": "nonfarm_payrolls_level",
            "title": "Total Nonfarm Payrolls (level, SA, thousands of persons)",
            "unit_hint": "thousand_persons",
            "official_sources": [
                "https://fred.stlouisfed.org/series/PAYEMS",
            ],
        },
        {
            "key": "nonfarm_payrolls_change",
            "title": "Total Nonfarm Payrolls (change from prior month, SA, thousands of persons)",
            "unit_hint": "thousand_persons",
            "official_sources": [
                "https://fred.stlouisfed.org/graph/?g=1O1Ix",
            ],
        },
        # 4.2 Unemployment Rate
        {
            "key": "unemployment_rate",
            "title": "Unemployment Rate (SA, %)",
            "unit_hint": "percent",
            "official_sources": [
                "https://fred.stlouisfed.org/series/UNRATE",
            ],
        },
        {
            "key": "unemployment_rate_mom",
            "title": "Unemployment Rate (change from prior month, SA, percentage points)",
            "unit_hint": "percentage_points",
            "official_sources": [
                "https://fred.stlouisfed.org/graph/?g=1OQH7",
            ],
        },
    ],
    "demand_sectoral_activity": [
        # ============================================================
        # 3. CONSUMPTION (3 VARIABLES)
        # ============================================================
        # 3.1 Retail Sales
        {
            "key": "retail_sales_level",
            "title": "Advance Monthly Retail and Food Services Sales (level, SA, billions of USD)",
            "unit_hint": "billion_usd",
            "official_sources": [
                "https://fred.stlouisfed.org/series/RSAFS",
            ],
        },
        {
            "key": "retail_sales_mom",
            "title": "Advance Monthly Retail and Food Services Sales (MoM %, SA)",
            "unit_hint": "percent",
            "official_sources": [
                "https://fred.stlouisfed.org/series/MARTSMPCSM44X72USS",
            ],
        },
        {
            "key": "retail_sales_yoy",
            "title": "Advance Monthly Retail and Food Services Sales (YoY %, SA, i.e., Percent Change from Month One Year Ago)",
            "unit_hint": "percent",
            "official_sources": [
                "https://fred.stlouisfed.org/graph/?g=1OQId",
            ],
        },
        # 3.2 Real PCE
        {
            "key": "real_pce_level",
            "title": "Real Personal Consumption Expenditures (level, billions of chained 2017 dollars, SAAR)",
            "unit_hint": "billion_chained_2017_usd_saar",
            "official_sources": [
                "https://fred.stlouisfed.org/series/PCEC96",
            ],
        },
        {
            "key": "real_pce_mom",
            "title": "Real Personal Consumption Expenditures (MoM %, SA)",
            "unit_hint": "percent",
            "official_sources": [
                "https://fred.stlouisfed.org/graph/?g=1OQIy",
            ],
        },
        {
            "key": "real_pce_yoy",
            "title": "Real Personal Consumption Expenditures (YoY %, i.e., Percent Change from Month One Year Ago)",
            "unit_hint": "percent",
            "official_sources": ["https://fred.stlouisfed.org/graph/?g=1OQJ4"],
        },
        # 3.3 Real Disposable Personal Income
        {
            "key": "real_dpi_level",
            "title": "Real Disposable Personal Income (level, billions of chained 2017 dollars, SAAR)",
            "unit_hint": "billion_chained_2017_usd_saar",
            "official_sources": [
                "https://fred.stlouisfed.org/series/DSPIC96",
            ],
        },
        {
            "key": "real_dpi_mom",
            "title": "Real Disposable Personal Income (MoM %, SA)",
            "unit_hint": "percent",
            "official_sources": [
                "https://fred.stlouisfed.org/graph/?g=1OQJn",
            ],
        },
        {
            "key": "real_dpi_yoy",
            "title": "Real Disposable Personal Income (YoY %, i.e., Percent Change from Month One Year Ago)",
            "unit_hint": "percent",
            "official_sources": ["https://fred.stlouisfed.org/graph/?g=1OQJF"],
        },
        # ============================================================
        # 5. HOUSING (4 VARIABLES)
        # ============================================================
        # 5.1 Housing Starts
        {
            "key": "housing_starts_level",
            "title": "Housing Starts (level, SAAR, thousands of units)", # changed to thousands to reduce unit error
            "unit_hint": "thousand_units_saar",
            "official_sources": [
                "https://fred.stlouisfed.org/series/HOUST",
            ],
        },
        {
            "key": "housing_starts_mom",
            "title": "Housing Starts (MoM %)",
            "unit_hint": "percent",
            "official_sources": [
                "https://fred.stlouisfed.org/graph/?g=1OQJV",
            ],
        },
        {
            "key": "housing_starts_yoy",
            "title": "Housing Starts (YoY %, i.e., Percent Change from Month One Year Ago)",
            "unit_hint": "percent",
            "official_sources": [
                "https://fred.stlouisfed.org/graph/?g=1OQK2",
            ],
        },
        # 5.2 Building Permits
        {
            "key": "building_permits_level",
            "title": "Building Permits (level, SAAR, thousands of units)", # changed to thousands to reduce unit error
            "unit_hint": "thousand_units_saar",
            "official_sources": [
                "https://fred.stlouisfed.org/series/PERMIT",
            ],
        },
        {
            "key": "building_permits_mom",
            "title": "Building Permits (MoM %)",
            "unit_hint": "percent",
            "official_sources": [
                "https://fred.stlouisfed.org/graph/?g=1OQKe",
            ],
        },
        {
            "key": "building_permits_yoy",
            "title": "Building Permits (YoY %, i.e., Percent Change from Month One Year Ago)",
            "unit_hint": "percent",
            "official_sources": [
                "https://fred.stlouisfed.org/graph/?g=1OQKh",
            ],
        },
        # 5.3 Existing Home Sales
        {
            "key": "existing_home_sales_level",
            "title": "Existing Home Sales (level, SAAR, thousands of units)", # changed to thousands to reduce unit error
            "unit_hint": "thousand_units_saar",
            "official_sources": [
                "https://fred.stlouisfed.org/series/EXHOSLUSM495S",
            ],
        },
        {
            "key": "existing_home_sales_mom",
            "title": "Existing Home Sales (MoM %)",
            "unit_hint": "percent",
            "official_sources": [
                "https://fred.stlouisfed.org/graph/?g=1OQKq",
            ],
        },
        {
            "key": "existing_home_sales_yoy",
            "title": "Existing Home Sales (YoY %)",
            "unit_hint": "percent",
            "official_sources": [
                "https://www.nar.realtor/research-and-statistics",
            ],
        },
        # 5.4 New Home Sales
        {
            "key": "new_home_sales_level",
            "title": "New Home Sales (level, SAAR, thousands of units)", # changed to thousands to reduce unit error
            "unit_hint": "thousand_units_saar",
            "official_sources": [
                "https://fred.stlouisfed.org/series/HSN1F",
            ],
        },
        {
            "key": "new_home_sales_mom",
            "title": "New Home Sales (MoM %)",
            "unit_hint": "percent",
            "official_sources": [
                "https://fred.stlouisfed.org/graph/?g=1OQLW",
            ],
        },
        {
            "key": "new_home_sales_yoy",
            "title": "New Home Sales (YoY %, i.e., Percent Change from Month One Year Ago)",
            "unit_hint": "percent",
            "official_sources": [
                "https://fred.stlouisfed.org/graph/?g=1OQLX",
            ],
        },
        # ============================================================
        # 6. SERVICES / BUSINESS ACTIVITY (1 VARIABLE)
        # ============================================================
        # 6.1 ISM Services PMI
        {
            "key": "ism_services_index",
            "title": "ISM Services PMI (index level, SA)",
            "unit_hint": "index",
            "official_sources": [
                "https://www.ismworld.org/supply-management-news-and-reports/reports/ism-report-on-business/",
            ],
        },
        {
            "key": "ism_services_mom",
            "title": "ISM Services PMI (MoM Percentage Point Change, index points)",
            "unit_hint": "index_points",
            "official_sources": [
                "https://www.ismworld.org/supply-management-news-and-reports/reports/ism-report-on-business/",
            ],
        },
    ],
}

# Instead of giving recent values, we now just give order-of-magnitude in the variable’s *native unit*.
# Typical order-of-magnitude in the variable’s *native unit*.
# Use these as digit-count priors for model predictions (with historical extremes in mind).
DIGIT_MAGNITUDE_HINTS = {
    # =========================
    # 1. OUTPUT / PRODUCTION
    # =========================
    # Real GDP level (billions of chained 2017$, SAAR): historically ~5,000–25,000
    "real_gdp_level": "5-digit (≈ O(10^4–10^5))",

    # Growth rates are in percent; digits vary with sign/decimals and occasional double-digits in crises
    "real_gdp_qoq": "percent",
    "real_gdp_yoy": "percent",

    # Industrial production index (2017=100): historically ~50–115-ish
    "industrial_production_index": "3-digit (≈ O(10^2–10^3))",
    "industrial_production_mom": "percent",
    "industrial_production_yoy": "percent",

    # Durable goods orders (billions of USD): typically ~200–350, can swing but stays 3-digit in billions
    "durable_goods_level": "3-digit (≈ O(10^2–10^3))",
    "durable_goods_mom": "percent",
    "durable_goods_yoy": "percent",

    # ISM PMI (index level): typically ~35–65
    "ism_manufacturing_index": "2-digit (≈ O(10^1–10^2))",
    # MoM change in index points: typically within about -5 to +5 (rarely larger)
    "ism_manufacturing_mom": "1-digit (≈ O(10^0–10^1))",

    # =========================
    # 2. INFLATION
    # =========================
    # CPI index (1982-84=100): historically ~20–330+
    "cpi_index": "3-digit (≈ O(10^2–10^3))",
    "cpi_mom": "percent",
    "cpi_yoy": "percent",

    # PPI final demand (1982=100): historically ~50–260+
    "ppi_index": "3-digit (≈ O(10^2–10^3))",
    "ppi_mom": "percent",
    "ppi_yoy": "percent",

    # PCE price index (2017=100): historically ~60–130+
    "pce_price_index_level": "3-digit (≈ O(10^2–10^3))",
    "pce_price_index_mom": "percent",
    "pce_price_index_yoy": "percent",

    # =========================
    # 4. LABOR MARKET
    # =========================
    # Payrolls level (thousands of persons): historically ~60,000–160,000
    "nonfarm_payrolls_level": "6-digit (≈ O(10^5–10^6))",
    # Payroll change (thousands): usually 2–3 digits, but crisis months can be ~10,000–20,000+
    "nonfarm_payrolls_change": "2–3-digit typical; allow up to 5-digit in crises (≈ O(10^2–10^5))",

    "unemployment_rate": "percent",
    # Monthly change is in percentage points; usually tenths, can spike to a few p.p. in extreme episodes
    "unemployment_rate_mom": "percentage_points",

    # =========================
    # 3. CONSUMPTION
    # =========================
    # Retail sales (billions of USD): historically ~100–800+
    "retail_sales_level": "3-digit (≈ O(10^2–10^3))",
    "retail_sales_mom": "percent",
    "retail_sales_yoy": "percent",

    # Real PCE (billions chained 2017$, SAAR): historically ~7,000–17,000+
    "real_pce_level": "5-digit (≈ O(10^4–10^5))",
    "real_pce_mom": "percent",
    "real_pce_yoy": "percent",

    # Real DPI (billions chained 2017$, SAAR): historically ~8,000–19,000+
    "real_dpi_level": "5-digit (≈ O(10^4–10^5))",
    "real_dpi_mom": "percent",
    "real_dpi_yoy": "percent",

    # =========================
    # 5. HOUSING (levels are thousands of units, SAAR)
    # =========================
    # Housing starts (thousands): historically ~400–2,500+
    "housing_starts_level": "4-digit (≈ O(10^3–10^4))",
    "housing_starts_mom": "percent",
    "housing_starts_yoy": "percent",

    # Building permits (thousands): historically ~500–2,500+
    "building_permits_level": "3-4-digit (≈ O(10^3–10^4))",
    "building_permits_mom": "percent",
    "building_permits_yoy": "percent",

    # Existing home sales (thousands): historically ~3,000–7,000+
    "existing_home_sales_level": "4-digit (≈ O(10^3–10^4))",
    "existing_home_sales_mom": "percent",
    "existing_home_sales_yoy": "percent",

    # New home sales (thousands): historically ~300–1,400+
    "new_home_sales_level": "3-4-digit (≈ O(10^3–10^4))",
    "new_home_sales_mom": "percent",
    "new_home_sales_yoy": "percent",

    # =========================
    # 6. SERVICES / BUSINESS ACTIVITY
    # =========================
    "ism_services_index": "2-digit (≈ O(10^1–10^2))",
    "ism_services_mom": "1-digit (≈ O(10^0–10^1))",
}


def _attach_digit_magnitude_hints():
    """Attach scale guardrail hints to each variable definition."""
    for group_vars in VARIABLES.values():
        for var in group_vars:
            hint = DIGIT_MAGNITUDE_HINTS.get(var["key"], "")
            var["scale_hint"] = hint


_attach_digit_magnitude_hints()

def get_variables(name):
    try:
        return VARIABLES[name]
    except KeyError:
        raise KeyError(f"Unknown variable group: {name}")
