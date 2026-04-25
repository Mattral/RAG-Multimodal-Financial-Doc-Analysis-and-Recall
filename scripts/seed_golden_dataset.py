#!/usr/bin/env python3
"""Seed the golden evaluation dataset with additional samples.

Usage:
    python scripts/seed_golden_dataset.py --output evals/golden_datasets/financial_qa.jsonl

Generates high-quality financial QA samples for pipeline evaluation.
Extend this script to add domain-specific samples for your use case.
"""
import argparse
import json
from pathlib import Path

SAMPLES = [
    {"question": "What was Tesla's total revenue in Q3 2023?",
     "ground_truth": "Tesla's total revenue in Q3 2023 was $23.35 billion, a 9% year-over-year increase.",
     "source_documents": ["tesla_10q_q3_2023.pdf"], "expected_page": 4,
     "expected_numeric_values": ["23.35", "9%"], "tags": ["revenue", "quarterly", "yoy"]},
    {"question": "What was the gross profit margin for Tesla in Q3 2023?",
     "ground_truth": "Tesla's gross profit margin in Q3 2023 was 17.9%, down from 25.1% in Q3 2022.",
     "source_documents": ["tesla_10q_q3_2023.pdf"], "expected_page": 5,
     "expected_numeric_values": ["17.9%", "25.1%"], "tags": ["margin", "yoy"]},
    {"question": "How many vehicles did Tesla deliver in 2023?",
     "ground_truth": "Tesla delivered approximately 1.81 million vehicles in full-year 2023, a 38% increase.",
     "source_documents": ["tesla_annual_2023.pdf"], "expected_page": 2,
     "expected_numeric_values": ["1.81 million", "38%"], "tags": ["deliveries", "annual"]},
    {"question": "What were the key risk factors related to competition in Tesla's 10-K?",
     "ground_truth": "Tesla cites competition from legacy automakers transitioning to EVs, new EV-native companies, and pricing pressure as key risks.",
     "source_documents": ["tesla_10k_2023.pdf"], "expected_page": 15,
     "expected_numeric_values": [], "tags": ["risk_factors", "competition"]},
    {"question": "What was Apple's revenue for fiscal year 2023?",
     "ground_truth": "Apple's revenue for fiscal year 2023 was $383.3 billion, a 3% decline from $394.3 billion in FY2022.",
     "source_documents": ["apple_10k_fy2023.pdf"], "expected_page": 22,
     "expected_numeric_values": ["383.3 billion", "394.3 billion", "3%"], "tags": ["revenue", "annual"]},
    {"question": "What was Microsoft's cloud revenue growth in FY2023?",
     "ground_truth": "Microsoft's Intelligent Cloud segment grew 19% to $87.9 billion; Azure and cloud services grew 29%.",
     "source_documents": ["msft_10k_fy2023.pdf"], "expected_page": 33,
     "expected_numeric_values": ["19%", "87.9 billion", "29%"], "tags": ["cloud", "growth"]},
    {"question": "What were Tesla's free cash flows in Q3 2023?",
     "ground_truth": "Free cash flow was $4.4 billion in Q3 2023: $5.1B operating cash minus $0.7B capex.",
     "source_documents": ["tesla_10q_q3_2023.pdf"], "expected_page": 8,
     "expected_numeric_values": ["4.4 billion", "5.1 billion", "0.7 billion"], "tags": ["cash_flow", "fcf"]},
    {"question": "What was Tesla's energy generation and storage revenue in Q3 2023?",
     "ground_truth": "Tesla's energy generation and storage segment revenue was $1.56 billion in Q3 2023.",
     "source_documents": ["tesla_10q_q3_2023.pdf"], "expected_page": 6,
     "expected_numeric_values": ["1.56 billion"], "tags": ["segment", "energy"]},
]

def main():
    parser = argparse.ArgumentParser(description="Seed golden QA dataset")
    parser.add_argument("--output", default="evals/golden_datasets/financial_qa.jsonl")
    parser.add_argument("--append", action="store_true", help="Append to existing file")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"

    with open(output_path, mode) as f:
        for sample in SAMPLES:
            f.write(json.dumps(sample) + "\n")

    action = "appended" if args.append else "written"
    print(f"✅ {len(SAMPLES)} samples {action} to {output_path}")
    print(f"   Run: rag-financial evaluate --dataset {output_path}")

if __name__ == "__main__":
    main()
