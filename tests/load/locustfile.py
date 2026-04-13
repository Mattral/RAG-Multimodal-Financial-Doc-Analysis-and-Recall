"""Locust load test for the RAG Financial API.

Usage:
    locust -f tests/load/locustfile.py --host=http://localhost:8000

Targets:
  - /api/v1/query  (primary — simulate 100 concurrent analysts)
  - /healthz       (baseline)
"""
from __future__ import annotations
import random
from locust import HttpUser, task, between, events

SAMPLE_QUERIES = [
    "What was the total revenue in Q3 2024?",
    "What is the year-over-year EBITDA growth?",
    "How did gross margins change compared to last year?",
    "What are the key risk factors mentioned in the 10-K?",
    "What is the debt-to-equity ratio?",
    "Summarize the cash flow from operations.",
    "What guidance was provided for the next quarter?",
    "How many employees does the company have?",
    "What were the main drivers of revenue growth?",
    "Describe the company's capital allocation strategy.",
]


class FinancialAnalystUser(HttpUser):
    """Simulates a financial analyst querying the RAG system."""

    wait_time = between(1, 3)

    def on_start(self):
        self.headers = {"X-API-Key": "dev-key", "X-Tenant-ID": f"load_test_{random.randint(1, 10)}"}

    @task(8)
    def query_document(self):
        """Main query task — 80% of traffic."""
        query = random.choice(SAMPLE_QUERIES)
        with self.client.post(
            "/api/v1/query",
            json={"query": query, "top_k": 5},
            headers=self.headers,
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") != "success":
                    resp.failure(f"Query returned non-success: {data.get('error')}")
            elif resp.status_code == 503:
                resp.failure("Pipeline not ready")
            else:
                resp.failure(f"HTTP {resp.status_code}")

    @task(2)
    def health_check(self):
        """Health probe — 20% of traffic."""
        self.client.get("/healthz", headers=self.headers)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Print SLO summary on test completion."""
    stats = environment.stats
    total = stats.total
    print(f"\n{'='*60}")
    print("LOAD TEST SUMMARY")
    print(f"Requests: {total.num_requests}")
    print(f"Failures: {total.num_failures} ({total.fail_ratio:.1%})")
    print(f"Median latency: {total.median_response_time}ms")
    print(f"P95 latency: {total.get_response_time_percentile(0.95)}ms")
    print(f"P99 latency: {total.get_response_time_percentile(0.99)}ms")
    print(f"RPS: {total.current_rps:.1f}")
    # SLO check
    p99 = total.get_response_time_percentile(0.99)
    slo_passed = p99 < 8000 and total.fail_ratio < 0.01
    print(f"SLO (p99<8s, <1% errors): {'✅ PASSED' if slo_passed else '❌ FAILED'}")
    print('='*60)
