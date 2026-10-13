# 知衡：AI依赖与学业倦怠预测系统前端

本目录是一套无外部依赖的中文单页前端，使用原生 HTML、CSS 与 JavaScript 实现。页面包含总体仪表盘、学生风险评估、预测概率与分层建议、局部 SHAP 解释、专业/年级对比、公平性审计，以及数据和伦理声明。

前端遵循一个重要原则：**接口不可用时明确显示“服务未连接”，绝不使用随机数、硬编码统计或演示数据冒充真实结果。**

## 文件

- `index.html`：页面结构、表单、语义化区域与无障碍标签。
- `styles.css`：响应式视觉系统、图表样式、加载/错误状态和动效。
- `app.js`：API 通信、表单校验、状态管理，以及原生 SVG/DOM 图表渲染。

## 运行

推荐由后端应用直接托管本目录，并保证前端页面与 `/api/*` 同源。也可以用任意静态服务器预览界面，但单独启动静态服务器时 API 请求会失败，页面将按设计显示“服务未连接”。

前端不依赖 CDN、字体服务、图表库或任何第三方 JavaScript。生产环境应使用 HTTPS，并由后端限制请求体大小、设置合理超时、记录模型版本且避免在日志中保存可识别个人身份的信息。

## API 总则

- 基础路径：同源 `/api`。
- 数据格式：请求和响应均为 UTF-8 JSON。
- 成功响应：`Content-Type: application/json`。
- 错误响应：建议采用 `application/problem+json`，至少包含 `detail`；前端也兼容 `{ "message": "..." }`。
- 数值比例统一使用 `0–1` 小数（例如 `0.274` 表示 `27.4%`）。前端为兼容旧接口也能识别 `0–100` 百分数，但新后端不应混用。
- 时间使用带时区的 ISO 8601 字符串。
- 枚举值保持英文稳定编码，中文仅作为展示标签。
- 所有端点均不得在失败时返回伪造的默认统计或预测。

错误示例：

```json
{
  "type": "https://example.edu/problems/model-unavailable",
  "title": "Model unavailable",
  "status": 503,
  "detail": "模型尚未加载，请稍后重试。"
}
```

## `GET /api/metadata`

用途：服务连通性检查、模型版本展示和表单枚举配置。

响应示例：

```json
{
  "api_version": "1.0",
  "service_name": "AI依赖与学业倦怠预测服务",
  "model": {
    "version": "burnout-xgb-2026.07.1",
    "target": "Burnout_Risk_Level",
    "classes": ["Low", "Medium", "High"],
    "trained_at": "2026-07-16T10:30:00+08:00"
  },
  "feature_schema": {
    "major_category": {
      "type": "categorical",
      "allowed_values": [
        { "value": "Humanities", "label": "人文社科" },
        { "value": "Medical", "label": "医学" },
        { "value": "Business", "label": "商科" },
        { "value": "STEM", "label": "理工科" },
        { "value": "Arts", "label": "艺术类" }
      ]
    },
    "year_of_study": {
      "type": "categorical",
      "allowed_values": [
        { "value": "Freshman", "label": "大一" },
        { "value": "Sophomore", "label": "大二" },
        { "value": "Junior", "label": "大三" },
        { "value": "Senior", "label": "大四" },
        { "value": "Graduate", "label": "研究生" }
      ]
    }
  }
}
```

`feature_schema` 可以包含全部 12 个输入字段。对于 `<select>` 字段，前端会优先使用接口返回的 `allowed_values` 更新选项；未提供时保留 HTML 内的合法枚举。

## `GET /api/dashboard`

用途：总体指标、专业/年级对比及全局SHAP前10项。所有统计和解释值均来自当前版本产物。

响应示例：

```json
{
  "updated_at": "2026-07-16T06:26:11.234949+00:00",
  "summary": {
    "total_students": 50000,
    "high_risk_rate": 0.24974,
    "avg_ai_dependency": 3.50536
  },
  "model_metrics": {
    "macro_f1": 0.5312617424929382,
    "selected_model": "Random Forest"
  },
  "major_comparison": [
    {
      "group": "Arts",
      "label": "艺术类",
      "count": 5933,
      "high_risk_rate": 0.226866677903253
    }
  ],
  "year_comparison": [
    {
      "group": "Freshman",
      "label": "大一",
      "count": 11031,
      "avg_ai_dependency": 3.4952406853413107
    }
  ],
  "global_shap": {
    "method": "mean absolute TreeSHAP",
    "explained_output": "Low/Medium/High 三类概率输出的跨样本、跨类别平均绝对归因强度",
    "features": [
      {
        "rank": 1,
        "feature": "weekly_genai_hours",
        "label": "每周生成式 AI 使用时长",
        "mean_abs_shap": 0.08554880483220348,
        "normalized_importance": 0.32627710318123854
      }
    ],
    "caveat": "全局 SHAP 对三类概率的绝对归因在解释样本与类别维度求均值；它不保留正负方向，也不表示因果关系。"
  }
}
```

示例只列出各数组的首项，真实接口返回全部5个专业、5个年级和全局SHAP前10项。全局SHAP是三类概率输出绝对归因的跨类别汇总，只表示模型整体使用某字段的强度；它不提供正负方向，也不同于个体预测所采用的序数期望风险SHAP。缺失指标会显示“— / 接口未提供该指标”，不会由前端估算；数组为空时显示明确空状态。

## `POST /api/predict`

用途：单个学生的研究性风险筛查。请求不包含姓名、学号、联系方式等直接身份信息。

请求体（字段名必须保持一致）：

```json
{
  "major_category": "STEM",
  "year_of_study": "Junior",
  "pre_semester_gpa": 3.25,
  "weekly_genai_hours": 8.5,
  "primary_use_case": "Debugging/Troubleshooting",
  "prompt_engineering_skill": "Intermediate",
  "tool_diversity": 3,
  "paid_subscription": false,
  "traditional_study_hours": 12.5,
  "perceived_ai_dependency": 6,
  "institutional_policy": "Allowed_With_Citation",
  "anxiety_level_during_exams": 7
}
```

字段约束：

| 字段 | 类型 | 约束/枚举 |
|---|---|---|
| `major_category` | string | `Humanities`, `Medical`, `Business`, `STEM`, `Arts` |
| `year_of_study` | string | `Freshman`, `Sophomore`, `Junior`, `Senior`, `Graduate` |
| `pre_semester_gpa` | number | `0–4` |
| `weekly_genai_hours` | number | `0–40` |
| `primary_use_case` | string | `Copywriting/Drafting`, `Ideation`, `Summarizing_Reading`, `Debugging/Troubleshooting`, `Direct_Answer_Generation` |
| `prompt_engineering_skill` | string | `Beginner`, `Intermediate`, `Advanced` |
| `tool_diversity` | integer | `1–5` |
| `paid_subscription` | boolean | `true` / `false` |
| `traditional_study_hours` | number | `0–40` |
| `perceived_ai_dependency` | integer | `1–10` |
| `institutional_policy` | string | `Allowed_With_Citation`, `Strict_Ban`, `Actively_Encouraged` |
| `anxiety_level_during_exams` | integer | `1–10` |

> 设计上不把 `Post_Semester_GPA` 与 `Skill_Retention_Score` 作为在线预测输入。这两个字段在早期预警时点尚不可获得，纳入会造成时间/目标泄漏，使离线指标虚高并削弱实际部署有效性。后端训练管线应采用相同特征集合。

成功响应示例：

```json
{
  "request_id": "5a7a60f4-207c-4aa0-871b-beb0bc8dd68e",
  "model_version": "burnout-xgb-2026.07.1",
  "generated_at": "2026-07-16T10:43:21+08:00",
  "predicted_class": "Medium",
  "probabilities": {
    "Low": 0.18,
    "Medium": 0.57,
    "High": 0.25
  },
  "summary": "模型提示当前处于中等风险区间，建议结合近期学习体验进行人工复核。",
  "explanation": {
    "method": "TreeSHAP",
    "base_value": 0.0,
    "features": [
      {
        "feature": "anxiety_level_during_exams",
        "label": "考试焦虑",
        "value": 7,
        "shap_value": 0.284
      },
      {
        "feature": "traditional_study_hours",
        "label": "传统学习时长",
        "value": 12.5,
        "shap_value": -0.116
      }
    ]
  },
  "recommendations": [
    "与辅导员共同复核近期学习负荷和睡眠状况。",
    "设置无 AI 学习时段，通过主动回忆检验知识掌握。"
  ],
  "disclaimer": "本结果仅供教育支持与研究参考，不构成医学或心理诊断，也不得作为处分依据。"
}
```

契约要求：

- `predicted_class` 必须是 `Low`、`Medium`、`High` 之一。
- `probabilities` 必须同时提供三类概率，值在 `0–1`，总和应在浮点误差范围内等于 1。
- `explanation.features[].shap_value` 为局部贡献：正值推动预测向更高风险，负值推动预测向更低风险。若多分类模型的原始 SHAP 定义不同，后端必须先转换为该可解释语义，或在契约中显式增加 `explained_class` 并同步修改前端文案。
- `recommendations` 必须由后端规则/专家策略层生成，不应由前端根据概率自行拼接。
- 建议后端在返回前完成 Pydantic/JSON Schema 校验。响应缺少风险等级或完整概率时，前端会将其视为无效响应并显示错误，而不会猜测。

## `GET /api/fairness`

用途：呈现按专业、年级等学生群体拆分的模型表现。后端应只返回样本量达到预设最小值的分组；小样本应合并、抑制或标记为不可判定，防止不稳定估计和再识别风险。

响应示例：

```json
{
  "evaluated_at": "2026-07-16T09:30:00+08:00",
  "status": "warning",
  "summary": "年级维度的最大 TPR 差异超过预警阈值，需要复核阈值与样本代表性。",
  "overall": {
    "status": "warning",
    "max_gap": 0.117,
    "metric": "true_positive_rate"
  },
  "thresholds": {
    "max_tpr_gap": 0.10,
    "max_fpr_gap": 0.10,
    "min_group_size": 50
  },
  "groups": [
    {
      "dimension": "major_category",
      "dimension_label": "专业类别",
      "group": "STEM",
      "group_label": "理工科",
      "sample_size": 412,
      "true_positive_rate": 0.81,
      "false_positive_rate": 0.14,
      "selection_rate": 0.29,
      "status": "pass"
    }
  ]
}
```

`status` 枚举为 `pass`、`warning`、`fail`。公平性不是一个单一数字：界面展示状态的同时保留样本量、TPR（召回真实高风险者）、FPR（误报率）和选择率，最终结论必须由人工结合场景审查。

## 前端状态与安全行为

- 每个请求有 10 秒超时；超时、网络错误、非 2xx 或非 JSON 响应均进入明确错误状态。
- 初始加载并行请求元数据、总览和公平性接口。部分失败时成功模块仍可使用，失败模块不会回填样例数据。
- 预测按钮在请求期间禁用，避免重复提交；失败后保留用户输入便于重试。
- 表单先做浏览器端范围校验；后端仍必须独立校验所有字段，不能信任前端。
- 浏览器端不使用 `localStorage`、`sessionStorage`、Cookie 或 IndexedDB 保存评估内容。
- 所有来自 API 的文本通过 `textContent` 写入页面，降低脚本注入风险。
- SHAP 解释只代表模型在该次预测中的数学依据，不自动证明因果关系或干预有效性。

## 无障碍与响应式

- 使用语义化标题、导航、表单、表格、状态区域与可见键盘焦点。
- 表单错误通过 `aria-invalid` / `aria-describedby` 关联到对应字段。
- 移动端导航支持遮罩、Esc 关闭与焦点进入。
- 图表同时保留文字数值，不只依赖颜色编码。
- 遵循 `prefers-reduced-motion: reduce`，关闭非必要动画和平滑滚动。
- 支持窄屏、平板、桌面和打印样式。

## 联调验收清单

1. 后端未启动：顶部、总览、对比与公平性模块均明确显示“服务未连接”，无任何随机统计。
2. `/api/metadata` 成功：侧栏显示服务在线，枚举选项可由元数据更新。
3. 表单缺失或越界：阻止提交、显示中文错误并聚焦首个无效字段。
4. `/api/predict` 成功：风险等级、三类概率、SHAP 与建议均来自响应。
5. `/api/predict` 返回缺失概率：显示响应无效，不推断缺失值。
6. Dashboard/Fairness 单独失败：成功模块保持可用，失败模块显示独立错误状态。
7. 360px 宽屏、键盘 Tab、200% 缩放和减少动态效果模式下可完整操作。
