import assert from "node:assert/strict";
import test from "node:test";

const workerUrl = new URL("../dist/server/index.js", import.meta.url);
workerUrl.searchParams.set("api-test", `${process.pid}-${Date.now()}`);
const { default: worker } = await import(workerUrl.href);
const environment = {
  ASSETS: {
    fetch: async () => new Response("Not found", { status: 404 }),
  },
};
const context = {
  waitUntil() {},
  passThroughOnException() {},
};

function request(body, method = "POST") {
  return worker.fetch(
    new Request("http://localhost/api/predict", {
      method,
      headers: { "Content-Type": "application/json" },
      body: method === "POST" ? body : undefined,
    }),
    environment,
    context,
  );
}

const validPayload = {
  major_category: "STEM",
  year_of_study: "Senior",
  pre_semester_gpa: 3.2,
  weekly_genai_hours: 8,
  primary_use_case: "Direct_Answer_Generation",
  prompt_engineering_skill: "Intermediate",
  tool_diversity: 3,
  paid_subscription: true,
  traditional_study_hours: 8,
  perceived_ai_dependency: 5,
  institutional_policy: "Allowed_With_Citation",
  anxiety_level_during_exams: 7,
};
const nonHighPayload = {
  major_category: "STEM",
  year_of_study: "Freshman",
  pre_semester_gpa: 3.934,
  weekly_genai_hours: 0.23,
  primary_use_case: "Copywriting/Drafting",
  prompt_engineering_skill: "Advanced",
  tool_diversity: 3,
  paid_subscription: false,
  traditional_study_hours: 17.12,
  perceived_ai_dependency: 1,
  institutional_policy: "Actively_Encouraged",
  anxiety_level_during_exams: 4,
};
const highPayload = {
  major_category: "STEM",
  year_of_study: "Graduate",
  pre_semester_gpa: 2.39,
  weekly_genai_hours: 40,
  primary_use_case: "Direct_Answer_Generation",
  prompt_engineering_skill: "Advanced",
  tool_diversity: 3,
  paid_subscription: false,
  traditional_study_hours: 11.46,
  perceived_ai_dependency: 8,
  institutional_policy: "Strict_Ban",
  anxiety_level_during_exams: 10,
};

test("GET exposes the audited selective model metrics", async () => {
  const response = await request(undefined, "GET");
  assert.equal(response.status, 200);
  const result = await response.json();
  assert.equal(result.prediction_ready, true);
  assert.match(result.model_version, /^asym-selective-logistic-/);
  assert.ok(result.metrics.selective_accuracy >= 0.85);
  assert.equal(result.metrics.selective_coverage, 0.7124);
  assert.equal(result.metrics.abstention_rate, 0.28759999999999997);
  assert.equal(result.decision.selectedOn, "validation only");
  assert.equal(result.decision.nonHighProbabilityThreshold, 0.2);
  assert.equal(result.decision.highProbabilityThreshold, 0.515);
});

test("POST returns a real high-risk probability and review-aware decision", async () => {
  const response = await request(JSON.stringify(validPayload));
  assert.equal(response.status, 200);
  const result = await response.json();
  assert.ok(["NonHigh", "Review", "High"].includes(result.predicted_class));
  assert.ok(["auto", "review"].includes(result.decision_status));
  assert.ok(result.high_risk_probability >= 0);
  assert.ok(result.high_risk_probability <= 1);
  assert.ok(Math.abs(
    result.probabilities.NonHigh + result.probabilities.High - 1,
  ) < 1e-12);
  assert.match(result.summary, /高风险概率/);
  assert.match(result.disclaimer, /不构成医学或心理诊断/);
});

test("POST can emit all three operational outcomes", async () => {
  const [nonHighResponse, reviewResponse, highResponse] = await Promise.all([
    request(JSON.stringify(nonHighPayload)),
    request(JSON.stringify(validPayload)),
    request(JSON.stringify(highPayload)),
  ]);
  assert.deepEqual(
    [nonHighResponse.status, reviewResponse.status, highResponse.status],
    [200, 200, 200],
  );
  const [nonHigh, review, high] = await Promise.all([
    nonHighResponse.json(),
    reviewResponse.json(),
    highResponse.json(),
  ]);
  assert.equal(nonHigh.predicted_class, "NonHigh");
  assert.equal(nonHigh.decision_status, "auto");
  assert.equal(review.predicted_class, "Review");
  assert.equal(review.decision_status, "review");
  assert.equal(high.predicted_class, "High");
  assert.equal(high.decision_status, "auto");
});

test("rejects invalid, incomplete, out-of-range, and oversized requests", async () => {
  const invalidJson = await request("{");
  assert.equal(invalidJson.status, 400);

  const incomplete = await request(JSON.stringify({ major_category: "STEM" }));
  assert.equal(incomplete.status, 422);
  assert.ok((await incomplete.json()).missing_fields.length > 1);

  const outOfRange = await request(JSON.stringify({
    ...validPayload,
    pre_semester_gpa: 5,
  }));
  assert.equal(outOfRange.status, 422);
  assert.deepEqual((await outOfRange.json()).invalid_fields, ["pre_semester_gpa"]);

  const oversized = await request(JSON.stringify({
    ...validPayload,
    padding: "x".repeat(17000),
  }));
  assert.equal(oversized.status, 413);
});
