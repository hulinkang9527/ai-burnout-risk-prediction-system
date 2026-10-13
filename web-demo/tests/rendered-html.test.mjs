import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function render(path = "/", init = {}) {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}-${path}`);
  const { default: worker } = await import(workerUrl.href);
  const headers = new Headers(init.headers);
  if (!headers.has("accept")) {
    headers.set("accept", path.startsWith("/api/") ? "application/json" : "text/html");
  }
  return worker.fetch(
    new Request(`http://localhost${path}`, {
      ...init,
      headers,
    }),
    {
      ASSETS: {
        fetch: async () => new Response("Not found", { status: 404 }),
      },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );
}

const validPredictionPayload = {
  major_category: "STEM",
  year_of_study: "Senior",
  pre_semester_gpa: 3.2,
  weekly_genai_hours: 8,
  primary_use_case: "Ideation",
  prompt_engineering_skill: "Intermediate",
  tool_diversity: 3,
  paid_subscription: false,
  traditional_study_hours: 12,
  perceived_ai_dependency: 5,
  institutional_policy: "Allowed_With_Citation",
  anxiety_level_during_exams: 6,
};

async function postPrediction(payload) {
  return render("/api/predict", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  });
}

test("server-renders the finished Chinese research site", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /基于生成式AI使用与感知依赖特征的学业倦怠风险预测系统设计与实现/);
  assert.match(html, /二手数据可视化看板/);
  assert.match(html, /模型表现与可解释性/);
  assert.match(html, /学业倦怠风险评估/);
  assert.match(html, /86\.09%/);
  assert.match(html, /71\.24%/);
  assert.match(html, /52\.34%/);
  assert.match(html, /人工复核/);
  assert.match(html, /20\.0%–51\.5%/);
  assert.doesNotMatch(html, /Your site is taking shape|Building your site/);
});

test("page, edge API, contract, and exported metrics share one canonical model", async () => {
  const [page, route, layout, compactRaw, summaryRaw, contractRaw] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/api/predict/route.ts", import.meta.url), "utf8"),
    readFile(new URL("../app/layout.tsx", import.meta.url), "utf8"),
    readFile(new URL("../public/compact_model.json", import.meta.url), "utf8"),
    readFile(new URL("../public/model_summary.json", import.meta.url), "utf8"),
    readFile(new URL("../public/project_contract.json", import.meta.url), "utf8"),
  ]);
  const compact = JSON.parse(compactRaw);
  const summary = JSON.parse(summaryRaw);
  const contract = JSON.parse(contractRaw);

  assert.equal(compact.modelVersion, summary.modelVersion);
  assert.equal(compact.metrics.selective_accuracy, summary.metrics.selective_accuracy);
  assert.equal(compact.metrics.selective_coverage, summary.metrics.selective_coverage);
  assert.equal(compact.metrics.roc_auc, summary.metrics.roc_auc);
  assert.ok(compact.metrics.selective_accuracy >= 0.85);
  assert.equal(compact.metrics.selective_coverage, 0.7124);
  assert.equal(compact.metrics.abstention_rate, 0.28759999999999997);
  assert.ok(compact.metrics.overall_accuracy < 0.85);
  assert.equal(compact.decision.selectedOn, "validation only");
  assert.equal(compact.decision.nonHighProbabilityThreshold, 0.2);
  assert.equal(compact.decision.highProbabilityThreshold, 0.515);
  assert.ok(compact.metrics.validation_wilson_lower_95 >= 0.85);
  assert.ok(compact.metrics.selective_wilson_lower_95 >= 0.85);
  assert.equal(compact.linear.coefficients.length, 37);
  assert.equal(contract.data.sha256, "4d911088c4b12d60a450a9acae6b606f4119ebbb48679518e427a4fc00778472");
  assert.equal(contract.primaryTask.modelVersion, "random-forest-e959d64abc4a");
  assert.equal(contract.auxiliaryTask.modelVersion, compact.modelVersion);
  assert.equal(contract.auxiliaryTask.derivedFeatures.length, 8);
  assert.equal(contract.inputContract.fields.length, 12);
  assert.equal(contract.canonicalStatuses.review_low_confidence, "0.200 < p(High) < 0.515");

  assert.match(page, /model_summary\.json/);
  assert.match(page, /project_contract\.json/);
  assert.match(page, /需人工复核/);
  assert.match(route, /compact_model\.json/);
  assert.match(route, /nonHighProbabilityThreshold/);
  assert.match(route, /highProbabilityThreshold/);
  assert.match(layout, /86\.09%/);
  assert.doesNotMatch(layout, /og-model-v3\.png/);
  assert.doesNotMatch(route, /RandomForest|trees|真实随机森林/);
});

test("edge prediction API accepts the exact 12-field contract", async () => {
  const response = await postPrediction(validPredictionPayload);
  assert.equal(response.status, 200);
  assert.equal(response.headers.get("access-control-allow-origin"), null);
  assert.equal(response.headers.get("cache-control"), "no-store");

  const body = await response.json();
  assert.match(body.predicted_class, /^(NonHigh|Review|High)$/);
  assert.match(body.canonical_status, /^(auto_non_high|review_low_confidence|auto_high)$/);
  assert.equal(typeof body.high_risk_probability, "number");
});

test("edge prediction API rejects an unknown categorical value", async () => {
  const response = await postPrediction({
    ...validPredictionPayload,
    major_category: "Engineering",
  });
  assert.equal(response.status, 422);
  const body = await response.json();
  assert.equal(body.error_code, "INVALID_CATEGORICAL_FIELDS");
  assert.deepEqual(body.invalid_fields, ["major_category"]);
});

test("edge prediction API rejects a string boolean", async () => {
  const response = await postPrediction({
    ...validPredictionPayload,
    paid_subscription: "false",
  });
  assert.equal(response.status, 422);
  const body = await response.json();
  assert.equal(body.error_code, "INVALID_BOOLEAN_FIELD");
  assert.deepEqual(body.invalid_fields, ["paid_subscription"]);
});

test("edge prediction API rejects a numeric string", async () => {
  const response = await postPrediction({
    ...validPredictionPayload,
    weekly_genai_hours: "8",
  });
  assert.equal(response.status, 422);
  const body = await response.json();
  assert.equal(body.error_code, "INVALID_NUMERIC_FIELDS");
  assert.deepEqual(body.invalid_fields, ["weekly_genai_hours"]);
});

test("edge prediction API rejects fractional ordinal scale values", async () => {
  const response = await postPrediction({
    ...validPredictionPayload,
    perceived_ai_dependency: 5.5,
    anxiety_level_during_exams: 6.5,
  });
  assert.equal(response.status, 422);
  const body = await response.json();
  assert.equal(body.error_code, "INVALID_INTEGER_FIELD");
  assert.deepEqual(body.invalid_fields, [
    "perceived_ai_dependency",
    "anxiety_level_during_exams",
  ]);
});

test("edge prediction API rejects extra fields", async () => {
  const response = await postPrediction({
    ...validPredictionPayload,
    post_semester_gpa: 3.4,
  });
  assert.equal(response.status, 422);
  const body = await response.json();
  assert.equal(body.error_code, "UNEXPECTED_FIELDS");
  assert.deepEqual(body.unexpected_fields, ["post_semester_gpa"]);
});

test("edge prediction API rejects a JSON array", async () => {
  const response = await postPrediction([validPredictionPayload]);
  assert.equal(response.status, 400);
  const body = await response.json();
  assert.equal(body.error_code, "INVALID_JSON_OBJECT");
});
