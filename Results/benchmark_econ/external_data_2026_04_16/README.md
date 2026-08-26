Fetched on 2026-04-16.

Files:

- source_manifest.csv and source_manifest.json: source links, status, and notes
- raw/: direct source files downloaded as-is
- processed/: normalized CSVs for benchmark use

Status summary:

- real_dpi: fetched exactly from FRED series DSPIC96 as a direct CSV fallback for BEA data
- real_pce: fetched exactly from FRED series PCEC96 as a direct CSV fallback for BEA data
- new_home_sales: fetched exactly from the official Census workbook and parsed from the monthly U.S. SAAR column
- existing_home_sales: parsed 100 monthly observations from the manually expanded Investing.com calendar HTML, covering 2017-12 through 2026-03
- ism_manufacturing: parsed 99 monthly observations from the manually expanded Investing.com calendar HTML, covering 2018-01 through 2026-03
- ism_services: parsed 99 monthly observations from the manually expanded Investing.com calendar HTML, covering 2018-01 through 2026-03

Notes:

- The Investing.com files came from locally saved HTML after manually clicking Show More; the parser reads the embedded __NEXT_DATA__ payload rather than the short visible table.
- existing_home_sales had one payload row with a missing reference_period; it was inferred as release-month-minus-one after checking adjacent rows in sequence.
- real_dpi and real_pce: if you want official-programmatic pulls instead of the FRED fallback, register for a BEA API key and switch the source to BEA.
