# Example Financial Documents

Place 2-3 real, publicly available financial PDFs here to enable the
one-click "Try Example" feature.

## Recommended sources (all public SEC filings)

| Filename | Source | Why it's good for demos |
|---|---|---|
| tesla_10q_q3_2023.pdf | SEC EDGAR | Dense tables, revenue charts |
| apple_earnings_q4_2023.pdf | Apple IR | Clean layout, labelled charts |
| microsoft_10k_fy2023.pdf | SEC EDGAR | Complex segment tables |

## Download

```bash
# SEC EDGAR full-text search:
# https://efts.sec.gov/LATEST/search-index?q=%22Tesla%22&forms=10-Q
```

## Wiring into app.py

```python
gr.Examples(
    examples=[["examples/tesla_10q_q3_2023.pdf", "What was Q3 revenue?"]],
    inputs=[pdf_input, question_box],
    label="Try a real filing",
)
```
