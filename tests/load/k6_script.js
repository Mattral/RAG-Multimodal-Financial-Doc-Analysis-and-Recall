/**
 * k6 load test for RAG Financial API
 *
 * Usage:
 *   k6 run tests/load/k6_script.js
 *   k6 run --vus 50 --duration 60s tests/load/k6_script.js
 *
 * SLOs tested:
 *   - p(99) query latency < 8000ms
 *   - Error rate < 1%
 *   - p(95) query latency < 5000ms
 */
import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

const errorRate = new Rate('errors');
const queryLatency = new Trend('query_latency_ms');

export const options = {
  stages: [
    { duration: '30s', target: 10 },   // ramp up
    { duration: '60s', target: 50 },   // sustained load
    { duration: '30s', target: 100 },  // peak
    { duration: '30s', target: 0 },    // ramp down
  ],
  thresholds: {
    'http_req_duration{name:query}': ['p(99)<8000', 'p(95)<5000'],
    'errors': ['rate<0.01'],
    'http_req_failed': ['rate<0.01'],
  },
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';
const API_KEY = __ENV.API_KEY || 'dev-key';

const SAMPLE_QUERIES = [
  'What was total revenue in Q3 2023?',
  'How did gross margins change year-over-year?',
  'What are the key risk factors related to competition?',
  'What was EBITDA in the most recent quarter?',
  'Describe the revenue trend from the charts',
  'What guidance was provided for next quarter?',
  'How many employees does the company have?',
  'What were the main drivers of revenue growth?',
];

export default function () {
  const query = SAMPLE_QUERIES[Math.floor(Math.random() * SAMPLE_QUERIES.length)];
  const tenantId = `load_test_${Math.floor(Math.random() * 5)}`;

  // ── Health check (10% of traffic) ─────────────────────────────────────────
  if (Math.random() < 0.1) {
    const healthRes = http.get(`${BASE_URL}/healthz`);
    check(healthRes, { 'health OK': (r) => r.status === 200 });
    sleep(0.5);
    return;
  }

  // ── Query (90% of traffic) ─────────────────────────────────────────────────
  const payload = JSON.stringify({ query, top_k: 5, tenant_id: tenantId });
  const params = {
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key': API_KEY,
      'X-Tenant-ID': tenantId,
    },
    tags: { name: 'query' },
  };

  const start = Date.now();
  const res = http.post(`${BASE_URL}/api/v1/query`, payload, params);
  const duration = Date.now() - start;

  queryLatency.add(duration);
  errorRate.add(res.status >= 400);

  check(res, {
    'status 200': (r) => r.status === 200,
    'has answer': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.status === 'success' && body.answer !== null;
      } catch { return false; }
    },
    'latency < 8s': () => duration < 8000,
  });

  sleep(Math.random() * 2 + 0.5);  // 0.5–2.5s think time
}

export function handleSummary(data) {
  const p99 = data.metrics.http_req_duration.values['p(99)'];
  const p95 = data.metrics.http_req_duration.values['p(95)'];
  const errorPct = (data.metrics.http_req_failed.values.rate * 100).toFixed(2);
  const sloPass = p99 < 8000 && data.metrics.http_req_failed.values.rate < 0.01;

  console.log('\n' + '='.repeat(60));
  console.log('LOAD TEST SUMMARY');
  console.log('='.repeat(60));
  console.log(`Total requests: ${data.metrics.http_reqs.values.count}`);
  console.log(`Error rate:     ${errorPct}%`);
  console.log(`p50 latency:    ${data.metrics.http_req_duration.values['p(50)'].toFixed(0)}ms`);
  console.log(`p95 latency:    ${p95.toFixed(0)}ms`);
  console.log(`p99 latency:    ${p99.toFixed(0)}ms`);
  console.log(`SLO (p99<8s, <1% errors): ${sloPass ? '✅ PASSED' : '❌ FAILED'}`);
  console.log('='.repeat(60));

  return { stdout: '' };
}
