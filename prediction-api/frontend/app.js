(() => {
  "use strict";

  const API = Object.freeze({
    metadata: "/api/metadata",
    dashboard: "/api/dashboard",
    predict: "/api/predict",
    fairness: "/api/fairness",
    timeoutMs: 10000
  });

  const RISK_LABELS = Object.freeze({ Low: "低", Medium: "中", High: "高" });
  const GROUP_LABELS = Object.freeze({
    Humanities: "人文社科", Medical: "医学", Business: "商科", STEM: "理工科", Arts: "艺术类",
    Freshman: "大一", Sophomore: "大二", Junior: "大三", Senior: "大四", Graduate: "研究生"
  });
  const FEATURE_LABELS = Object.freeze({
    major_category: "专业类别", year_of_study: "所在年级", pre_semester_gpa: "上学期 GPA",
    weekly_genai_hours: "每周 AI 使用时长", primary_use_case: "AI 主要用途",
    prompt_engineering_skill: "提示词技能", tool_diversity: "AI 工具数量",
    paid_subscription: "付费订阅", traditional_study_hours: "传统学习时长",
    perceived_ai_dependency: "自评 AI 依赖", institutional_policy: "学校 AI 政策",
    anxiety_level_during_exams: "考试焦虑"
  });
  const NUMERIC_FIELDS = new Set([
    "pre_semester_gpa", "weekly_genai_hours", "tool_diversity", "traditional_study_hours",
    "perceived_ai_dependency", "anxiety_level_during_exams"
  ]);

  const byId = (id) => document.getElementById(id);
  const dom = {
    sidebar: byId("sidebar"), menuButton: byId("menu-button"), scrim: byId("scrim"),
    refreshButton: byId("refresh-button"), retryButton: byId("retry-button"),
    connectionBanner: byId("connection-banner"), connectionMessage: byId("connection-message"),
    systemState: byId("system-state"), metricGrid: byId("metric-grid"),
    dashboardUpdated: byId("dashboard-updated"), majorChart: byId("major-chart"),
    yearChart: byId("year-chart"), yearChartWrap: byId("year-chart-wrap"),
    globalShapChart: byId("global-shap-chart"), globalShapCaveat: byId("global-shap-caveat"),
    auditSummary: byId("audit-summary"), fairnessTable: byId("fairness-table"),
    form: byId("assessment-form"), progressBar: byId("progress-bar"), progressLabel: byId("progress-label"),
    dependencyOutput: byId("dependency-output"), anxietyOutput: byId("anxiety-output"),
    consent: byId("consent"), consentError: byId("consent-error"), submitButton: byId("submit-button"),
    resultEmpty: byId("result-empty"), resultLoading: byId("result-loading"),
    resultError: byId("result-error"), resultErrorMessage: byId("result-error-message"),
    resultContent: byId("result-content"), resultPanel: byId("result-panel"),
    modelVersion: byId("model-version"), riskGauge: byId("risk-gauge"), gaugeValue: byId("gauge-value"),
    riskProbability: byId("risk-probability"), riskLevel: byId("risk-level"), riskTitle: byId("risk-title"),
    riskSummary: byId("risk-summary-text"), probabilityBreakdown: byId("probability-breakdown"),
    shapChart: byId("shap-chart"), recommendationList: byId("recommendation-list"),
    resultDisclaimer: byId("result-disclaimer"), toastRegion: byId("toast-region")
  };

  const fieldNames = [
    "major_category", "year_of_study", "pre_semester_gpa", "traditional_study_hours",
    "weekly_genai_hours", "primary_use_case", "prompt_engineering_skill", "tool_diversity",
    "paid_subscription", "institutional_policy", "perceived_ai_dependency", "anxiety_level_during_exams"
  ];
  let metadataModelVersion = "—";
  let refreshInProgress = false;

  class ApiError extends Error {
    constructor(message, status = 0, detail = "") {
      super(message);
      this.name = "ApiError";
      this.status = status;
      this.detail = detail;
    }
  }

  async function apiRequest(url, options = {}) {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), API.timeoutMs);
    const headers = new Headers(options.headers || {});
    headers.set("Accept", "application/json");
    if (options.body) headers.set("Content-Type", "application/json");

    try {
      const response = await fetch(url, {
        ...options,
        headers,
        signal: controller.signal,
        credentials: "same-origin"
      });
      const contentType = response.headers.get("content-type") || "";
      const body = contentType.includes("application/json") ? await response.json() : null;
      if (!response.ok) {
        const detail = body?.detail || body?.message || `HTTP ${response.status}`;
        throw new ApiError("服务返回错误", response.status, String(detail));
      }
      if (!body || typeof body !== "object") throw new ApiError("服务返回的数据格式无效", response.status);
      return body;
    } catch (error) {
      if (error.name === "AbortError") throw new ApiError("请求超时", 0, "服务在 10 秒内未响应");
      if (error instanceof ApiError) throw error;
      throw new ApiError("无法连接服务", 0, error?.message || "网络请求失败");
    } finally {
      window.clearTimeout(timeout);
    }
  }

  function normalizedError(error) {
    if (!error) return "未知错误";
    if (error.status) return `${error.detail || error.message}（HTTP ${error.status}）`;
    return error.detail || error.message || "网络请求失败";
  }

  function setConnectionState(state, message = "") {
    dom.systemState.classList.remove("is-online", "is-offline");
    const strong = dom.systemState.querySelector("strong");
    const small = dom.systemState.querySelector("small");
    if (state === "online") {
      dom.systemState.classList.add("is-online");
      strong.textContent = "模型服务正常";
      small.textContent = "接口已连接";
      dom.connectionBanner.classList.add("is-hidden");
    } else {
      dom.systemState.classList.add("is-offline");
      strong.textContent = state === "partial" ? "部分服务未连接" : "服务未连接";
      small.textContent = "未生成替代数据";
      dom.connectionMessage.textContent = message || "无法读取模型服务，请启动后端后重试。页面不会生成或伪造预测结果。";
      dom.connectionBanner.classList.remove("is-hidden");
    }
  }

  function makeElement(tag, className, text) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text !== undefined && text !== null) element.textContent = String(text);
    return element;
  }

  function firstFinite(...values) {
    for (const value of values) {
      const number = Number(value);
      if (value !== null && value !== "" && Number.isFinite(number)) return number;
    }
    return null;
  }

  function asRatio(value) {
    if (value === null || value === undefined || value === "") return null;
    const number = Number(value);
    if (!Number.isFinite(number)) return null;
    const normalized = number > 1 ? number / 100 : number;
    return normalized < 0 || normalized > 1 ? null : normalized;
  }

  function formatPercent(value, digits = 1) {
    const ratio = asRatio(value);
    return ratio === null ? "—" : `${(ratio * 100).toFixed(digits)}%`;
  }

  function formatNumber(value, digits = 1) {
    if (value === null || value === undefined || value === "") return "—";
    const number = Number(value);
    if (!Number.isFinite(number)) return "—";
    return new Intl.NumberFormat("zh-CN", { maximumFractionDigits: digits }).format(number);
  }

  function formatDate(value) {
    if (!value) return "";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return String(value);
    return new Intl.DateTimeFormat("zh-CN", { dateStyle: "medium", timeStyle: "short" }).format(date);
  }

  function displayGroup(value, label) {
    return label || GROUP_LABELS[value] || value || "未命名群体";
  }

  function showToast(message, type = "info") {
    const toast = makeElement("div", `toast${type === "error" ? " is-error" : type === "success" ? " is-success" : ""}`, message);
    dom.toastRegion.appendChild(toast);
    window.setTimeout(() => {
      toast.style.opacity = "0";
      toast.style.transform = "translateY(8px)";
      window.setTimeout(() => toast.remove(), 220);
    }, 3600);
  }

  function applyMetadata(data) {
    metadataModelVersion = data?.model?.version || data?.model_version || "—";
    const schema = data.feature_schema || Object.fromEntries(
      Array.isArray(data.features) ? data.features.filter((item) => item?.name).map((item) => [item.name, item]) : []
    );
    if (!schema || typeof schema !== "object") return;

    for (const name of fieldNames) {
      const control = dom.form.elements[name];
      const config = schema[name];
      if (!config || !control) continue;
      if (control instanceof HTMLInputElement) {
        const minimum = firstFinite(config.minimum, config.min);
        const maximum = firstFinite(config.maximum, config.max);
        if (minimum !== null) control.min = String(minimum);
        if (maximum !== null) control.max = String(maximum);
        continue;
      }
      const select = control;
      if (!(select instanceof HTMLSelectElement)) continue;
      const options = config?.allowed_values || config?.options || config?.categories;
      if (!Array.isArray(options) || options.length === 0) continue;
      const placeholder = select.options[0]?.textContent || "请选择";
      select.replaceChildren(new Option(placeholder, ""));
      for (const item of options) {
        const value = typeof item === "object" ? item.value : item;
        const label = typeof item === "object" ? item.label : GROUP_LABELS[item] || item;
        if (value !== undefined && value !== null) select.add(new Option(String(label), String(value)));
      }
    }
  }

  function metricCard(title, value, unit, note, icon, glow) {
    const card = makeElement("article", "metric-card");
    card.style.setProperty("--card-glow", glow);
    const head = makeElement("div", "metric-card-head");
    head.append(makeElement("span", "", title), makeElement("i", "metric-icon", icon));
    const valueLine = makeElement("div", "metric-value");
    valueLine.append(makeElement("strong", "", value), makeElement("span", "", unit));
    card.append(head, valueLine, makeElement("p", "metric-note", note));
    return card;
  }

  function renderDashboard(data) {
    const summary = data.summary || data.dataset || data.overview || {};
    const metrics = data.model_metrics || data.metrics || {};
    const total = firstFinite(summary.total_students, summary.rows, data.total_students);
    const highRate = firstFinite(summary.high_risk_rate, data.high_risk_rate);
    const dependency = firstFinite(summary.avg_ai_dependency, summary.mean_ai_dependency, data.avg_ai_dependency);
    const f1 = firstFinite(metrics.macro_f1, metrics.f1_macro, data.macro_f1);

    dom.metricGrid.replaceChildren(
      metricCard("纳入分析样本", total === null ? "—" : formatNumber(total, 0), "人", total === null ? "接口未提供该指标" : "去身份化研究样本", "Σ", "rgba(87,169,255,.09)"),
      metricCard("数据集 High 标签占比", formatPercent(highRate), "", highRate === null ? "接口未提供该指标" : "仅为样本标签分布，不代表患病率", "△", "rgba(255,120,111,.09)"),
      metricCard("平均感知 AI 依赖", dependency === null ? "—" : formatNumber(dependency, 2), "/ 10", dependency === null ? "接口未提供该指标" : "1—10 分单项自评代理变量", "◎", "rgba(56,223,196,.09)"),
      metricCard("模型宏平均 F1", formatPercent(f1), "", f1 === null ? "接口未提供该指标" : "测试集多分类表现", "◇", "rgba(242,189,104,.09)")
    );
    dom.metricGrid.setAttribute("aria-busy", "false");
    const updated = data.updated_at || data.generated_at || data.model?.trained_at;
    dom.dashboardUpdated.textContent = updated ? `数据更新时间：${formatDate(updated)}` : "服务已连接 · 未提供更新时间";
    renderGlobalShap(data);
    renderMajorChart(data.major_comparison || data.comparisons?.majors || []);
    renderYearChart(data.year_comparison || data.comparisons?.years || []);
  }

  function renderDashboardUnavailable(reason) {
    const card = makeElement("article", "metric-card unavailable-card");
    card.append(makeElement("strong", "", "服务未连接"), makeElement("p", "", `无法读取总览数据：${reason}。未显示任何替代统计。`));
    dom.metricGrid.replaceChildren(card);
    dom.metricGrid.setAttribute("aria-busy", "false");
    dom.dashboardUpdated.textContent = "总览服务未连接";
    renderGlobalShap({}, `服务未连接，无法读取全局解释数据：${reason}`);
    renderMajorChart([], "服务未连接，无法读取专业对比数据");
    renderYearChart([], "服务未连接，无法读取年级对比数据");
  }

  function renderGlobalShap(data, emptyMessage = "接口未返回全局 SHAP 数据") {
    const explanation = data?.global_shap || {};
    const items = explanation.features || data?.top_shap_features || [];
    dom.globalShapChart.replaceChildren();
    if (!Array.isArray(items) || items.length === 0) {
      dom.globalShapChart.append(makeElement("div", "chart-empty", emptyMessage));
      return;
    }
    const normalized = items
      .filter((item) => item && typeof item === "object")
      .map((item, index) => ({
        rank: Number.isInteger(item.rank) ? item.rank : index + 1,
        label: item.label || FEATURE_LABELS[item.feature] || item.original_feature || item.source_field || item.feature || "未命名特征",
        importance: asRatio(item.normalized_importance ?? item.normalized ?? item.share)
      }))
      .filter((item) => item.importance !== null)
      .slice(0, 10);
    if (!normalized.length) {
      dom.globalShapChart.append(makeElement("div", "chart-empty", "全局 SHAP 数据格式无效"));
      return;
    }
    normalized.forEach((item) => {
      const row = makeElement("div", "global-shap-row");
      row.setAttribute("role", "listitem");
      row.setAttribute("aria-label", `第 ${item.rank} 位，${item.label}，归一化重要性 ${formatPercent(item.importance, 2)}`);
      row.append(makeElement("span", "global-shap-rank", String(item.rank)), makeElement("span", "global-shap-label", item.label));
      const track = makeElement("span", "global-shap-track");
      const fill = makeElement("i", "global-shap-fill");
      fill.style.setProperty("--importance", `${Math.max(item.importance * 100, 1)}%`);
      track.append(fill);
      const value = makeElement("strong", "global-shap-value", formatPercent(item.importance, 2));
      row.append(track, value);
      dom.globalShapChart.append(row);
    });
    if (explanation.caveat) {
      dom.globalShapCaveat.replaceChildren(makeElement("strong", "", "解释边界："), document.createTextNode(String(explanation.caveat)));
    }
  }

  function renderMajorChart(items, emptyMessage = "接口未返回专业对比数据") {
    dom.majorChart.replaceChildren();
    if (!Array.isArray(items) || items.length === 0) {
      dom.majorChart.append(makeElement("div", "chart-empty", emptyMessage));
      return;
    }
    const normalized = items
      .filter((item) => item && typeof item === "object")
      .map((item) => ({
        name: displayGroup(item.group || item.major || item.value, item.label || item.group_label),
        rate: asRatio(item.high_risk_rate ?? item.rate ?? item.value_rate)
      }))
      .filter((item) => item.rate !== null)
      .slice(0, 8);
    if (!normalized.length) {
      dom.majorChart.append(makeElement("div", "chart-empty", "专业对比数据格式无效"));
      return;
    }
    for (const item of normalized) {
      const group = makeElement("div", "bar-group");
      group.tabIndex = 0;
      group.setAttribute("role", "img");
      group.setAttribute("aria-label", `${item.name}高风险率 ${formatPercent(item.rate)}`);
      group.append(makeElement("span", "bar-value", formatPercent(item.rate)));
      const column = makeElement("div", "bar-column");
      const fill = makeElement("span", "bar-fill");
      fill.style.setProperty("--height", `${Math.max(item.rate * 100, 1.5)}%`);
      column.append(fill);
      group.append(column, makeElement("span", "bar-label", item.name));
      dom.majorChart.append(group);
    }
  }

  function svgNode(tag, attrs = {}, text = "") {
    const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, String(value)));
    if (text) node.textContent = text;
    return node;
  }

  function renderYearChart(items, emptyMessage = "接口未返回年级对比数据") {
    const empty = dom.yearChartWrap.querySelector(".chart-empty");
    dom.yearChart.replaceChildren();
    if (!Array.isArray(items) || items.length === 0) {
      empty.textContent = emptyMessage;
      empty.classList.remove("is-hidden");
      return;
    }
    const normalized = items
      .filter((item) => item && typeof item === "object")
      .map((item) => ({
        name: displayGroup(item.group || item.year || item.value, item.label || item.group_label),
        value: firstFinite(item.avg_ai_dependency, item.mean_dependency, item.dependency_score)
      }))
      .filter((item) => item.value !== null)
      .slice(0, 8);
    if (!normalized.length) {
      empty.textContent = "年级对比数据格式无效";
      empty.classList.remove("is-hidden");
      return;
    }
    empty.classList.add("is-hidden");

    const defs = svgNode("defs");
    const gradient = svgNode("linearGradient", { id: "area-gradient", x1: "0", x2: "0", y1: "0", y2: "1" });
    gradient.append(svgNode("stop", { offset: "0%", "stop-color": "#38dfc4", "stop-opacity": ".23" }), svgNode("stop", { offset: "100%", "stop-color": "#38dfc4", "stop-opacity": "0" }));
    defs.append(gradient);
    dom.yearChart.append(defs);
    const left = 55, right = 610, top = 25, bottom = 235;
    [0, 2.5, 5, 7.5, 10].forEach((tick) => {
      const y = bottom - (tick / 10) * (bottom - top);
      dom.yearChart.append(svgNode("line", { x1: left, x2: right, y1: y, y2: y, class: "chart-grid-line" }));
      dom.yearChart.append(svgNode("text", { x: left - 13, y: y + 4, class: "chart-axis-label", "text-anchor": "end" }, tick));
    });
    const span = normalized.length === 1 ? 0 : (right - left) / (normalized.length - 1);
    const points = normalized.map((item, index) => ({
      ...item,
      x: normalized.length === 1 ? (left + right) / 2 : left + index * span,
      y: bottom - (Math.max(0, Math.min(10, item.value)) / 10) * (bottom - top)
    }));
    const linePath = points.map((point, index) => `${index ? "L" : "M"}${point.x},${point.y}`).join(" ");
    const areaPath = `${linePath} L${points.at(-1).x},${bottom} L${points[0].x},${bottom} Z`;
    dom.yearChart.append(svgNode("path", { d: areaPath, class: "chart-area" }), svgNode("path", { d: linePath, class: "chart-series" }));
    points.forEach((point) => {
      dom.yearChart.append(svgNode("circle", { cx: point.x, cy: point.y, r: 5, class: "chart-point", tabindex: "0", role: "img", "aria-label": `${point.name}平均AI依赖 ${formatNumber(point.value, 2)}分` }));
      dom.yearChart.append(svgNode("text", { x: point.x, y: point.y - 13, class: "chart-point-label" }, formatNumber(point.value, 2)));
      dom.yearChart.append(svgNode("text", { x: point.x, y: bottom + 27, class: "chart-axis-label", "text-anchor": "middle" }, point.name));
    });
  }

  function normalizeAuditStatus(value) {
    const status = String(value || "").toLowerCase();
    if (["pass", "passed", "fair", "ok", "通过"].includes(status)) return "pass";
    if (["warning", "review", "caution", "关注", "警告"].includes(status)) return "warning";
    if (["fail", "failed", "unfair", "不通过"].includes(status)) return "fail";
    return "unknown";
  }

  function renderFairness(data) {
    const overallStatus = normalizeAuditStatus(data.status || data.overall?.status);
    const labelMap = { pass: "通过", warning: "需关注", fail: "未通过", unknown: "不可判定" };
    const score = dom.auditSummary.querySelector(".audit-score");
    const scoreClass = overallStatus === "pass" ? "" : overallStatus === "warning" ? "is-warning" : overallStatus === "fail" ? "is-fail" : "is-pending";
    score.className = `audit-score ${scoreClass}`.trim();
    score.querySelector("strong").textContent = labelMap[overallStatus];
    const maxGap = firstFinite(data.overall?.max_gap, data.max_gap);
    score.querySelector("span").textContent = maxGap === null ? "审计状态" : `最大差异 ${formatPercent(maxGap)}`;
    const copy = dom.auditSummary.querySelector("div:last-child");
    copy.querySelector("h3").textContent = `模型分组性能状态：${labelMap[overallStatus]}`;
    copy.querySelector("p").textContent = data.summary || "审计接口未提供文字摘要，请结合下表指标和服务端阈值进行人工复核。";

    const groups = data.groups || data.group_metrics || [];
    dom.fairnessTable.replaceChildren();
    if (!Array.isArray(groups) || groups.length === 0) {
      const row = document.createElement("tr");
      const cell = makeElement("td", "table-empty", "接口未返回分组审计指标");
      cell.colSpan = 7;
      row.append(cell);
      dom.fairnessTable.append(row);
      return;
    }
    groups.filter((item) => item && typeof item === "object").forEach((item) => {
      const row = document.createElement("tr");
      const status = normalizeAuditStatus(item.status);
      const values = [
        item.dimension_label || FEATURE_LABELS[item.dimension] || item.dimension || "—",
        displayGroup(item.group, item.group_label),
        formatNumber(item.sample_size ?? item.count, 0),
        formatPercent(item.true_positive_rate ?? item.tpr),
        formatPercent(item.false_positive_rate ?? item.fpr),
        formatPercent(item.selection_rate)
      ];
      values.forEach((value) => row.append(makeElement("td", "", value)));
      const statusCell = document.createElement("td");
      statusCell.append(makeElement("span", `status-badge status-${status}`, labelMap[status]));
      row.append(statusCell);
      dom.fairnessTable.append(row);
    });
    if (!dom.fairnessTable.children.length) {
      const row = document.createElement("tr");
      const cell = makeElement("td", "table-empty", "分组审计数据格式无效");
      cell.colSpan = 7;
      row.append(cell);
      dom.fairnessTable.append(row);
    }
  }

  function renderFairnessUnavailable(reason) {
    const score = dom.auditSummary.querySelector(".audit-score");
    score.className = "audit-score is-pending";
    score.querySelector("strong").textContent = "—";
    score.querySelector("span").textContent = "服务未连接";
    const copy = dom.auditSummary.querySelector("div:last-child");
    copy.querySelector("h3").textContent = "分组性能审计服务未连接";
    copy.querySelector("p").textContent = `无法读取审计结果：${reason}。页面未生成替代结论。`;
    const row = document.createElement("tr");
    const cell = makeElement("td", "table-empty", "服务未连接，无法读取分组审计指标");
    cell.colSpan = 7;
    row.append(cell);
    dom.fairnessTable.replaceChildren(row);
  }

  async function refreshAll() {
    if (refreshInProgress) return;
    refreshInProgress = true;
    dom.refreshButton.classList.add("is-spinning");
    dom.refreshButton.disabled = true;
    const requests = [
      ["元数据", API.metadata], ["总览", API.dashboard], ["分组性能审计", API.fairness]
    ];
    try {
      const results = await Promise.allSettled(requests.map(([, url]) => apiRequest(url)));
      const failures = [];

      if (results[0].status === "fulfilled") {
        try { applyMetadata(results[0].value); } catch (error) { failures.push("元数据"); }
      } else failures.push("元数据");

      if (results[1].status === "fulfilled") {
        try { renderDashboard(results[1].value); }
        catch (error) {
          failures.push("总览");
          renderDashboardUnavailable(`响应格式无效：${error.message}`);
        }
      } else {
        failures.push("总览");
        renderDashboardUnavailable(normalizedError(results[1].reason));
      }

      if (results[2].status === "fulfilled") {
        try { renderFairness(results[2].value); }
        catch (error) {
          failures.push("分组性能审计");
          renderFairnessUnavailable(`响应格式无效：${error.message}`);
        }
      } else {
        failures.push("分组性能审计");
        renderFairnessUnavailable(normalizedError(results[2].reason));
      }

      if (failures.length === 0) {
        setConnectionState("online");
      } else {
        const uniqueFailures = [...new Set(failures)];
        setConnectionState(uniqueFailures.length === requests.length ? "offline" : "partial", `${uniqueFailures.join("、")}服务未连接或响应无效。未返回的模块不会显示伪造数据，请启动或检查后端服务后重试。`);
      }
    } finally {
      dom.refreshButton.classList.remove("is-spinning");
      dom.refreshButton.disabled = false;
      refreshInProgress = false;
    }
  }

  function updateRange(input, output) {
    output.value = input.value;
    const min = Number(input.min), max = Number(input.max), value = Number(input.value);
    const progress = ((value - min) / (max - min)) * 100;
    input.style.background = `linear-gradient(90deg, var(--teal) 0 ${progress}%, rgba(255,255,255,.09) ${progress}% 100%)`;
  }

  function updateProgress() {
    const complete = fieldNames.filter((name) => String(dom.form.elements[name]?.value ?? "").trim() !== "").length;
    dom.progressLabel.textContent = `${complete} / ${fieldNames.length}`;
    dom.progressBar.style.width = `${(complete / fieldNames.length) * 100}%`;
  }

  function clearFieldError(field) {
    if (!field) return;
    const wrapper = field.closest(".field");
    if (!wrapper) return;
    wrapper.classList.remove("has-error");
    const error = wrapper.querySelector(".field-error");
    if (error) error.textContent = "";
    field.removeAttribute("aria-invalid");
    field.removeAttribute("aria-describedby");
  }

  function fieldErrorMessage(field) {
    const label = FEATURE_LABELS[field.name] || "该字段";
    if (field.validity.valueMissing) return field instanceof HTMLSelectElement ? `请选择${label}` : `请填写${label}`;
    if (field.validity.rangeUnderflow) return `${label}不能小于 ${field.min}`;
    if (field.validity.rangeOverflow) return `${label}不能大于 ${field.max}`;
    if (field.validity.stepMismatch) return `${label}的数值精度不符合要求`;
    if (field.validity.badInput) return `${label}必须是有效数字`;
    return `请检查${label}`;
  }

  function showFieldError(field) {
    if (!field) return;
    const wrapper = field.closest(".field");
    if (!wrapper) return;
    const error = wrapper.querySelector(".field-error");
    if (!error) return;
    const id = `${field.id}-error`;
    error.id = id;
    error.textContent = fieldErrorMessage(field);
    wrapper.classList.add("has-error");
    field.setAttribute("aria-invalid", "true");
    field.setAttribute("aria-describedby", id);
  }

  function validateForm() {
    let firstInvalid = null;
    fieldNames.forEach((name) => {
      const field = dom.form.elements[name];
      clearFieldError(field);
      if (!field.checkValidity()) {
        showFieldError(field);
        firstInvalid ||= field;
      }
    });
    dom.consentError.textContent = "";
    if (!dom.consent.checked) {
      dom.consentError.textContent = "请先阅读并确认评估边界与支持提示";
      dom.consent.setAttribute("aria-invalid", "true");
      firstInvalid ||= dom.consent;
    } else {
      dom.consent.removeAttribute("aria-invalid");
    }
    if (firstInvalid) firstInvalid.focus({ preventScroll: false });
    return !firstInvalid;
  }

  function serializeForm() {
    const payload = {};
    fieldNames.forEach((name) => {
      const field = dom.form.elements[name];
      if (NUMERIC_FIELDS.has(name)) payload[name] = Number(field.value);
      else if (name === "paid_subscription") payload[name] = field.value === "true";
      else payload[name] = field.value;
    });
    return payload;
  }

  function setResultState(state, errorMessage = "") {
    dom.resultEmpty.classList.toggle("is-hidden", state !== "empty");
    dom.resultLoading.classList.toggle("is-hidden", state !== "loading");
    dom.resultError.classList.toggle("is-hidden", state !== "error");
    dom.resultContent.classList.toggle("is-hidden", state !== "content");
    dom.resultPanel.setAttribute("aria-busy", String(state === "loading"));
    if (state === "error") dom.resultErrorMessage.textContent = errorMessage;
  }

  function normalizeRisk(value) {
    const key = String(value || "").trim().toLowerCase();
    if (["low", "低", "低风险"].includes(key)) return "Low";
    if (["medium", "moderate", "中", "中风险"].includes(key)) return "Medium";
    if (["high", "高", "高风险"].includes(key)) return "High";
    return null;
  }

  function probabilityFor(probabilities, riskClass) {
    if (!probabilities || typeof probabilities !== "object") return null;
    if (Array.isArray(probabilities)) {
      const item = probabilities.find((entry) => normalizeRisk(entry?.class || entry?.label) === riskClass);
      return asRatio(item?.probability ?? item?.value);
    }
    const key = Object.keys(probabilities).find((name) => normalizeRisk(name) === riskClass);
    return key ? asRatio(probabilities[key]) : null;
  }

  function renderPrediction(data) {
    const riskClass = normalizeRisk(data.predicted_class || data.risk_level || data.prediction);
    if (!riskClass) throw new ApiError("预测响应缺少有效风险等级");
    const probabilities = data.probabilities || data.class_probabilities;
    const probabilityMap = {
      Low: probabilityFor(probabilities, "Low"),
      Medium: probabilityFor(probabilities, "Medium"),
      High: probabilityFor(probabilities, "High")
    };
    if (Object.values(probabilityMap).some((value) => value === null)) throw new ApiError("预测响应缺少完整的 Low、Medium、High 概率");
    const probabilityTotal = Object.values(probabilityMap).reduce((sum, value) => sum + value, 0);
    if (Math.abs(probabilityTotal - 1) > 0.025) throw new ApiError("预测响应中的分类概率之和不等于 1");

    const level = RISK_LABELS[riskClass];
    const titleMap = { Low: "当前风险相对较低", Medium: "建议关注近期状态", High: "建议优先提供支持" };
    const defaultSummary = "接口未提供结果摘要，请结合概率、影响因素并由专业人员复核。";
    dom.resultContent.classList.remove("risk-low", "risk-medium", "risk-high");
    dom.resultContent.classList.add(`risk-${riskClass.toLowerCase()}`);
    dom.modelVersion.textContent = `模型版本 ${data.model_version || data.model?.version || metadataModelVersion}`;
    dom.riskProbability.textContent = formatPercent(probabilityMap.High);
    dom.riskLevel.textContent = `${level}风险`;
    dom.riskTitle.textContent = titleMap[riskClass];
    dom.riskSummary.textContent = data.summary || data.interpretation || defaultSummary;
    dom.riskGauge.setAttribute("aria-label", `高风险类别的模型估计概率 ${formatPercent(probabilityMap.High)}，预测等级为${level}风险`);
    dom.gaugeValue.style.strokeDashoffset = "204.2";
    window.requestAnimationFrame(() => {
      dom.gaugeValue.style.strokeDashoffset = String(204.2 * (1 - probabilityMap.High));
    });

    dom.probabilityBreakdown.replaceChildren();
    const colors = { Low: "var(--success)", Medium: "var(--warning)", High: "var(--danger)" };
    ["Low", "Medium", "High"].forEach((name) => {
      const item = makeElement("div", "prob-item");
      const row = makeElement("div");
      row.append(makeElement("span", "", `${RISK_LABELS[name]}风险`), makeElement("strong", "", formatPercent(probabilityMap[name])));
      const track = makeElement("div", "prob-track");
      const fill = makeElement("span");
      fill.style.width = `${probabilityMap[name] * 100}%`;
      fill.style.setProperty("--bar-color", colors[name]);
      track.append(fill);
      item.append(row, track);
      dom.probabilityBreakdown.append(item);
    });

    renderShap(data.explanation?.features || data.shap_values || []);
    renderRecommendations(data.recommendations || data.support_plan || []);
    dom.resultDisclaimer.textContent = data.disclaimer || "本结果仅供教育支持与研究参考，不构成医学或心理诊断，也不得作为处分依据。";
    setResultState("content");
  }

  function renderShap(items) {
    dom.shapChart.replaceChildren();
    if (!Array.isArray(items) || items.length === 0) {
      dom.shapChart.append(makeElement("div", "chart-empty compact-empty", "接口未返回局部 SHAP 解释"));
      return;
    }
    const normalized = items
      .filter((item) => item && typeof item === "object")
      .map((item) => ({
        feature: item.feature || item.name,
        label: item.label || FEATURE_LABELS[item.feature || item.name] || item.feature || item.name,
        value: item.value,
        shap: firstFinite(item.shap_value, item.shap, item.contribution)
      }))
      .filter((item) => item.label && item.shap !== null)
      .sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap))
      .slice(0, 7);
    if (!normalized.length) {
      dom.shapChart.append(makeElement("div", "chart-empty compact-empty", "SHAP 解释数据格式无效"));
      return;
    }
    const max = Math.max(...normalized.map((item) => Math.abs(item.shap)), 0.0001);
    normalized.forEach((item) => {
      const positive = item.shap >= 0;
      const row = makeElement("div", "shap-row");
      const label = makeElement("span", "shap-label", item.label);
      if (item.value !== undefined && item.value !== null) label.title = `${item.label}：${item.value}`;
      const axis = makeElement("div", "shap-axis");
      const bar = makeElement("span", `shap-bar ${positive ? "is-positive" : "is-negative"}`);
      bar.style.setProperty("--width", `${Math.max((Math.abs(item.shap) / max) * 48, 1)}%`);
      axis.append(bar);
      const value = makeElement("span", `shap-value ${positive ? "is-positive" : "is-negative"}`, `${positive ? "+" : ""}${item.shap.toFixed(3)}`);
      row.append(label, axis, value);
      dom.shapChart.append(row);
    });
  }

  function renderRecommendations(items) {
    dom.recommendationList.replaceChildren();
    if (!Array.isArray(items) || items.length === 0) {
      dom.recommendationList.append(makeElement("li", "recommendation-unavailable", "接口未返回个体支持建议，请由教师或专业人员结合实际情况制定方案。"));
      return;
    }
    items.slice(0, 6).forEach((item) => {
      const text = typeof item === "string" ? item : item && typeof item === "object" ? item.text || item.recommendation || item.title : "";
      if (text) dom.recommendationList.append(makeElement("li", "", text));
    });
    if (!dom.recommendationList.children.length) {
      dom.recommendationList.append(makeElement("li", "recommendation-unavailable", "支持建议数据格式无效，请联系系统维护人员。"));
    }
  }

  async function handleSubmit(event) {
    event.preventDefault();
    if (!validateForm()) {
      showToast("请先修正表单中的缺失或无效信息", "error");
      return;
    }
    dom.submitButton.disabled = true;
    dom.submitButton.classList.add("is-loading");
    setResultState("loading");
    dom.resultPanel.scrollIntoView({ behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "center" });
    try {
      const data = await apiRequest(API.predict, { method: "POST", body: JSON.stringify(serializeForm()) });
      renderPrediction(data);
      showToast("风险评估已完成，请结合解释信息审慎阅读", "success");
    } catch (error) {
      const reason = normalizedError(error);
      setResultState("error", `预测服务未连接或响应无效：${reason}。系统没有生成或补造评估结果。`);
      setConnectionState("partial", `预测服务未连接：${reason}。请检查后端后重试，页面不会生成伪造预测。`);
      showToast("评估失败：预测服务未连接", "error");
    } finally {
      dom.submitButton.disabled = false;
      dom.submitButton.classList.remove("is-loading");
    }
  }

  function closeMenu(restoreFocus = false) {
    const wasOpen = dom.sidebar.classList.contains("is-open");
    dom.sidebar.classList.remove("is-open");
    dom.menuButton.setAttribute("aria-expanded", "false");
    dom.menuButton.setAttribute("aria-label", "打开导航菜单");
    dom.scrim.hidden = true;
    document.body.classList.remove("menu-open");
    if (wasOpen && restoreFocus) dom.menuButton.focus();
  }

  function toggleMenu() {
    const open = !dom.sidebar.classList.contains("is-open");
    dom.sidebar.classList.toggle("is-open", open);
    dom.menuButton.setAttribute("aria-expanded", String(open));
    dom.menuButton.setAttribute("aria-label", open ? "关闭导航菜单" : "打开导航菜单");
    dom.scrim.hidden = !open;
    document.body.classList.toggle("menu-open", open);
    if (open) dom.sidebar.querySelector(".nav-item").focus();
  }

  function setupObservers() {
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const revealItems = document.querySelectorAll(".reveal");
    if (reduceMotion || !("IntersectionObserver" in window)) {
      revealItems.forEach((item) => item.classList.add("is-visible"));
    } else {
      const revealObserver = new IntersectionObserver((entries, observer) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("is-visible");
            observer.unobserve(entry.target);
          }
        });
      }, { threshold: 0.08 });
      revealItems.forEach((item) => revealObserver.observe(item));
    }

    const navItems = [...document.querySelectorAll(".nav-item")];
    const sections = navItems.map((item) => byId(item.dataset.section)).filter(Boolean);
    let navFrame = 0;

    const updateActiveNav = () => {
      navFrame = 0;
      const anchor = Math.min(window.innerHeight * 0.32, 280);
      let activeSection = sections[0];
      sections.forEach((section) => {
        if (section.getBoundingClientRect().top <= anchor) activeSection = section;
      });
      if (window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 2) {
        activeSection = sections.at(-1);
      }
      navItems.forEach((item) => {
        const active = item.dataset.section === activeSection?.id;
        item.classList.toggle("is-active", active);
        if (active) item.setAttribute("aria-current", "page"); else item.removeAttribute("aria-current");
      });
    };

    const scheduleNavUpdate = () => {
      if (navFrame) return;
      navFrame = window.requestAnimationFrame(updateActiveNav);
    };

    window.addEventListener("scroll", scheduleNavUpdate, { passive: true });
    window.addEventListener("hashchange", scheduleNavUpdate);
    scheduleNavUpdate();
  }

  function setupEvents() {
    dom.menuButton.addEventListener("click", toggleMenu);
    dom.scrim.addEventListener("click", () => closeMenu(true));
    document.addEventListener("keydown", (event) => { if (event.key === "Escape") closeMenu(true); });
    document.querySelectorAll(".nav-item").forEach((item) => item.addEventListener("click", () => closeMenu(false)));
    dom.refreshButton.addEventListener("click", refreshAll);
    dom.retryButton.addEventListener("click", refreshAll);
    dom.form.addEventListener("submit", handleSubmit);
    dom.form.addEventListener("input", (event) => {
      if (event.target.name && fieldNames.includes(event.target.name)) {
        if (event.target.checkValidity()) clearFieldError(event.target);
        updateProgress();
      }
      if (event.target.id === "perceived_ai_dependency") updateRange(event.target, dom.dependencyOutput);
      if (event.target.id === "anxiety_level_during_exams") updateRange(event.target, dom.anxietyOutput);
    });
    dom.form.addEventListener("change", (event) => {
      if (event.target.name && fieldNames.includes(event.target.name)) {
        if (event.target.checkValidity()) clearFieldError(event.target);
        updateProgress();
      }
      if (event.target === dom.consent && event.target.checked) {
        dom.consentError.textContent = "";
        dom.consent.removeAttribute("aria-invalid");
      }
    });
    dom.form.addEventListener("reset", () => window.requestAnimationFrame(() => {
      dom.form.querySelectorAll(".field").forEach((field) => field.classList.remove("has-error"));
      dom.form.querySelectorAll(".field-error").forEach((error) => error.textContent = "");
      dom.form.querySelectorAll("[aria-invalid]").forEach((field) => field.removeAttribute("aria-invalid"));
      dom.consentError.textContent = "";
      updateRange(dom.form.elements.perceived_ai_dependency, dom.dependencyOutput);
      updateRange(dom.form.elements.anxiety_level_during_exams, dom.anxietyOutput);
      updateProgress();
      setResultState("empty");
      showToast("评估表单已清空");
    }));
  }

  function init() {
    setupEvents();
    setupObservers();
    updateRange(dom.form.elements.perceived_ai_dependency, dom.dependencyOutput);
    updateRange(dom.form.elements.anxiety_level_during_exams, dom.anxietyOutput);
    updateProgress();
    refreshAll();
  }

  init();
})();
