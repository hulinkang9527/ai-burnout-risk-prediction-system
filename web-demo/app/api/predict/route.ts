import compactModel from "../../../public/compact_model.json";

export const runtime = "edge";

const MAX_BODY_BYTES = 16 * 1024;
const JSON_HEADERS = {
  "Cache-Control": "no-store",
  "Content-Type": "application/json; charset=utf-8",
  "X-Content-Type-Options": "nosniff",
};
const REQUIRED_FIELDS = [
  "major_category",
  "year_of_study",
  "pre_semester_gpa",
  "weekly_genai_hours",
  "primary_use_case",
  "prompt_engineering_skill",
  "tool_diversity",
  "paid_subscription",
  "traditional_study_hours",
  "perceived_ai_dependency",
  "institutional_policy",
  "anxiety_level_during_exams",
] as const;
const REQUIRED_FIELD_SET = new Set<string>(REQUIRED_FIELDS);
const NUMERIC_LIMITS: Record<string, [number, number]> = {
  pre_semester_gpa: [0, 4],
  weekly_genai_hours: [0, 40],
  tool_diversity: [1, 5],
  traditional_study_hours: [0, 40],
  perceived_ai_dependency: [1, 10],
  anxiety_level_during_exams: [1, 10],
};
const INTEGER_FIELDS = [
  "tool_diversity",
  "perceived_ai_dependency",
  "anxiety_level_during_exams",
] as const;
const CATEGORICAL_OPTIONS: Record<string, readonly string[]> = {
  major_category: ["Arts", "Business", "Humanities", "Medical", "STEM"],
  year_of_study: ["Freshman", "Graduate", "Junior", "Senior", "Sophomore"],
  primary_use_case: [
    "Copywriting/Drafting",
    "Debugging/Troubleshooting",
    "Direct_Answer_Generation",
    "Ideation",
    "Summarizing_Reading",
  ],
  prompt_engineering_skill: ["Advanced", "Beginner", "Intermediate"],
  institutional_policy: [
    "Actively_Encouraged",
    "Allowed_With_Citation",
    "Strict_Ban",
  ],
};

type DisplayClass = "NonHigh" | "Review" | "High";
type CompactMetrics = {
  overall_accuracy: number;
  overall_macro_f1: number;
  roc_auc: number;
  selective_accuracy: number;
  selective_macro_f1: number;
  selective_balanced_accuracy: number;
  selective_coverage: number;
  abstention_rate: number;
  accepted_n: number;
  selective_wilson_lower_95: number;
  selective_wilson_upper_95: number;
  high_precision: number;
  high_recall_within_accepted: number;
  auto_high_recall_all_high: number;
  validation_selective_accuracy: number;
  validation_wilson_lower_95: number;
  legacy_three_class_accuracy: number;
  legacy_three_class_macro_f1: number;
};
type CompactModel = {
  modelVersion: string;
  trainedRows: number;
  validationRows: number;
  holdoutRows: number;
  task: {
    positiveClass: string;
    negativeClass: string;
    reviewClass: string;
    type: string;
  };
  decision: {
    nonHighProbabilityThreshold: number;
    highProbabilityThreshold: number;
    reviewLowerProbabilityExclusive: number;
    reviewUpperProbabilityExclusive: number;
    rule: string;
    selectedOn: string;
    criterion: string;
  };
  metrics: CompactMetrics;
  numeric: {
    features: string[];
    median: number[];
    mean: number[];
    scale: number[];
  };
  categorical: {
    features: string[];
    fallback: Array<string | boolean>;
    categories: Array<Array<string | boolean>>;
  };
  linear: {
    coefficients: number[];
    intercept: number;
    link: string;
  };
  safety: { disclaimer: string };
};

const model = compactModel as CompactModel;

function requestKey(feature: string) {
  const explicit: Record<string, string> = {
    Major_Category: "major_category",
    Year_of_Study: "year_of_study",
    Pre_Semester_GPA: "pre_semester_gpa",
    Weekly_GenAI_Hours: "weekly_genai_hours",
    Primary_Use_Case: "primary_use_case",
    Prompt_Engineering_Skill: "prompt_engineering_skill",
    Tool_Diversity: "tool_diversity",
    Paid_Subscription: "paid_subscription",
    Traditional_Study_Hours: "traditional_study_hours",
    Perceived_AI_Dependency: "perceived_ai_dependency",
    Institutional_Policy: "institutional_policy",
    Anxiety_Level_During_Exams: "anxiety_level_during_exams",
  };
  return explicit[feature] || feature;
}

function numericFeatures(payload: Record<string, unknown>) {
  const ai = Number(payload.weekly_genai_hours);
  const study = Number(payload.traditional_study_hours);
  const dependency = Number(payload.perceived_ai_dependency);
  const anxiety = Number(payload.anxiety_level_during_exams);
  const total = ai + study;
  const epsilon = 0.25;
  return {
    Pre_Semester_GPA: Number(payload.pre_semester_gpa),
    Weekly_GenAI_Hours: ai,
    Tool_Diversity: Number(payload.tool_diversity),
    Traditional_Study_Hours: study,
    Perceived_AI_Dependency: dependency,
    Anxiety_Level_During_Exams: anxiety,
    AI_to_Study_Ratio: ai / (study + epsilon),
    Total_Learning_Hours: total,
    AI_Hours_Share: ai / (total + epsilon),
    AI_minus_Study_Hours: ai - study,
    Dependency_x_AI_Hours: dependency * ai,
    Anxiety_per_Study_Hour: anxiety / (study + epsilon),
    Anxiety_x_Dependency: anxiety * dependency,
    Anxiety_plus_Dependency: anxiety + dependency,
  } as Record<string, number>;
}

function transform(modelDefinition: CompactModel, payload: Record<string, unknown>) {
  const vector: number[] = [];
  const numeric = numericFeatures(payload);
  modelDefinition.numeric.features.forEach((feature, index) => {
    const supplied = numeric[feature];
    const clean = Number.isFinite(supplied)
      ? supplied
      : modelDefinition.numeric.median[index];
    const scale = modelDefinition.numeric.scale[index] || 1;
    vector.push((clean - modelDefinition.numeric.mean[index]) / scale);
  });
  modelDefinition.categorical.features.forEach((feature, index) => {
    const supplied = payload[requestKey(feature)];
    const clean = supplied === undefined || supplied === null
      ? modelDefinition.categorical.fallback[index]
      : supplied;
    for (const category of modelDefinition.categorical.categories[index]) {
      vector.push(String(clean) === String(category) ? 1 : 0);
    }
  });
  return vector;
}

function sigmoid(logit: number) {
  if (logit >= 0) {
    const exponential = Math.exp(-logit);
    return 1 / (1 + exponential);
  }
  const exponential = Math.exp(logit);
  return exponential / (1 + exponential);
}

function predict(modelDefinition: CompactModel, vector: number[]) {
  if (vector.length !== modelDefinition.linear.coefficients.length) {
    throw new Error("Compact model feature width mismatch.");
  }
  const logit = vector.reduce(
    (total, value, index) =>
      total + value * modelDefinition.linear.coefficients[index],
    modelDefinition.linear.intercept,
  );
  const highProbability = sigmoid(logit);
  const nonHighThreshold =
    modelDefinition.decision.nonHighProbabilityThreshold;
  const highThreshold = modelDefinition.decision.highProbabilityThreshold;
  let predictedClass: DisplayClass = "Review";
  let decisionStatus: "auto" | "review" = "review";
  if (highProbability >= highThreshold) {
    predictedClass = "High";
    decisionStatus = "auto";
  } else if (highProbability <= nonHighThreshold) {
    predictedClass = "NonHigh";
    decisionStatus = "auto";
  }
  return {
    predictedClass,
    decisionStatus,
    highProbability,
    confidence: Math.max(highProbability, 1 - highProbability),
  };
}

function recommendations(
  risk: DisplayClass,
  payload: Record<string, unknown>,
) {
  const items: string[] = [];
  if (Number(payload.perceived_ai_dependency) >= 6) {
    items.push("建议安排不依赖生成式 AI 的独立学习时段，并记录完成情况。");
  }
  if (Number(payload.anxiety_level_during_exams) >= 7) {
    items.push("考试焦虑较高时，可优先联系学校心理支持或学业辅导资源。");
  }
  if (
    Number(payload.weekly_genai_hours)
    > Number(payload.traditional_study_hours)
  ) {
    items.push("建议逐步提高传统学习时间占比，避免把生成式 AI 当作唯一学习路径。");
  }
  if (!items.length) {
    items.push("保持规律作息与学习记录，并在状态持续恶化时主动寻求支持。");
  }
  if (risk === "High") {
    items.unshift("结果提示高倦怠风险，请尽快与可信任的老师、辅导员或专业支持人员沟通。");
  }
  if (risk === "Review") {
    items.unshift("当前样本的模型置信度不足，请结合正式量表、访谈或专业人员判断复核。");
  }
  return items;
}

function responseSummary(result: ReturnType<typeof predict>) {
  const probability = (result.highProbability * 100).toFixed(1);
  if (result.predictedClass === "High") {
    return `高风险概率为 ${probability}%，达到 51.5% 的自动 High 阈值，结果为“高风险”。`;
  }
  if (result.predictedClass === "NonHigh") {
    return `高风险概率为 ${probability}%，不高于 20.0% 的自动 Non-high 阈值，结果为“非高风险”。`;
  }
  return `高风险概率为 ${probability}%，处于 20.0%–51.5% 的人工复核区间，本次不作自动判定。`;
}

export async function OPTIONS() {
  return new Response(null, {
    status: 204,
    headers: {
      "Cache-Control": "no-store",
      "X-Content-Type-Options": "nosniff",
      Allow: "GET, POST, OPTIONS",
    },
  });
}

export async function GET() {
  return Response.json(
    {
      ok: true,
      prediction_ready: true,
      model_version: model.modelVersion,
      trained_rows: model.trainedRows,
      validation_rows: model.validationRows,
      holdout_rows: model.holdoutRows,
      task: model.task,
      decision: model.decision,
      metrics: model.metrics,
    },
    { headers: JSON_HEADERS },
  );
}

export async function POST(request: Request) {
  const rawBody = await request.text();
  if (new TextEncoder().encode(rawBody).byteLength > MAX_BODY_BYTES) {
    return Response.json(
      { detail: "预测请求超过大小限制。" },
      { status: 413, headers: JSON_HEADERS },
    );
  }

  let parsedPayload: unknown;
  try {
    parsedPayload = JSON.parse(rawBody) as unknown;
  } catch {
    return Response.json(
      { detail: "请求体不是有效的 JSON。", error_code: "INVALID_JSON" },
      { status: 400, headers: JSON_HEADERS },
    );
  }

  if (
    typeof parsedPayload !== "object"
    || parsedPayload === null
    || Array.isArray(parsedPayload)
  ) {
    return Response.json(
      {
        detail: "请求体必须是包含 12 个指定字段的 JSON 对象。",
        error_code: "INVALID_JSON_OBJECT",
      },
      { status: 400, headers: JSON_HEADERS },
    );
  }

  const payload = parsedPayload as Record<string, unknown>;
  const unexpected = Object.keys(payload).filter(
    (field) => !REQUIRED_FIELD_SET.has(field),
  );
  if (unexpected.length) {
    return Response.json(
      {
        detail: "请求中包含模型不接受的额外字段，未执行预测。",
        error_code: "UNEXPECTED_FIELDS",
        unexpected_fields: unexpected,
      },
      { status: 422, headers: JSON_HEADERS },
    );
  }

  const missing = REQUIRED_FIELDS.filter(
    (field) =>
      payload[field] === undefined
      || payload[field] === null
      || payload[field] === "",
  );
  if (missing.length) {
    return Response.json(
      {
        detail: "信息尚未收集完整，未执行预测。",
        error_code: "MISSING_FIELDS",
        missing_fields: missing,
      },
      { status: 422, headers: JSON_HEADERS },
    );
  }
  const invalidNumeric = Object.entries(NUMERIC_LIMITS)
    .filter(([field, [minimum, maximum]]) => {
      const value = payload[field];
      return typeof value !== "number"
        || !Number.isFinite(value)
        || value < minimum
        || value > maximum;
    })
    .map(([field]) => field);
  if (invalidNumeric.length) {
    return Response.json(
      {
        detail: "数值字段必须是有限数值且处于允许范围内；本站部署将两类每周时长收紧为 0–40 小时（本地 metadata 可接受上限为 168 小时）。未执行预测。",
        error_code: "INVALID_NUMERIC_FIELDS",
        invalid_fields: invalidNumeric,
      },
      { status: 422, headers: JSON_HEADERS },
    );
  }

  const invalidInteger = INTEGER_FIELDS.filter(
    (field) => !Number.isInteger(payload[field]),
  );
  if (invalidInteger.length) {
    return Response.json(
      {
        detail: "tool_diversity、perceived_ai_dependency 与 anxiety_level_during_exams 必须使用各自范围内的整数，未执行预测。",
        error_code: "INVALID_INTEGER_FIELD",
        invalid_fields: invalidInteger,
      },
      { status: 422, headers: JSON_HEADERS },
    );
  }

  const invalidCategorical = Object.entries(CATEGORICAL_OPTIONS)
    .filter(([field, options]) => {
      const value = payload[field];
      return typeof value !== "string" || !options.includes(value);
    })
    .map(([field]) => field);
  if (invalidCategorical.length) {
    return Response.json(
      {
        detail: "分类字段必须使用模型训练时定义的类别，未执行预测。",
        error_code: "INVALID_CATEGORICAL_FIELDS",
        invalid_fields: invalidCategorical,
      },
      { status: 422, headers: JSON_HEADERS },
    );
  }

  if (typeof payload.paid_subscription !== "boolean") {
    return Response.json(
      {
        detail: "paid_subscription 必须是布尔值 true 或 false，未执行预测。",
        error_code: "INVALID_BOOLEAN_FIELD",
        invalid_fields: ["paid_subscription"],
      },
      { status: 422, headers: JSON_HEADERS },
    );
  }

  try {
    const result = predict(model, transform(model, payload));
    return Response.json(
      {
        predicted_class: result.predictedClass,
        decision_status: result.decisionStatus,
        canonical_status:
          result.predictedClass === "Review"
            ? "review_low_confidence"
            : result.predictedClass === "High"
              ? "auto_high"
              : "auto_non_high",
        high_risk_probability: result.highProbability,
        confidence: result.confidence,
        probabilities: {
          NonHigh: 1 - result.highProbability,
          High: result.highProbability,
        },
        summary: responseSummary(result),
        recommendations: recommendations(result.predictedClass, payload),
        disclaimer: model.safety.disclaimer,
        model_version: model.modelVersion,
        metrics: model.metrics,
      },
      { headers: JSON_HEADERS },
    );
  } catch {
    return Response.json(
      { detail: "真实预测模型当前不可用，未生成风险结论。" },
      { status: 503, headers: JSON_HEADERS },
    );
  }
}
