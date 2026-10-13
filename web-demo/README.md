# 智学预警线上演示站

站内`public/project_contract.json`是主项目`00_统一事实契约.json`的部署副本。修改题名、字段、模型版本、阈值、指标或安全边界时，应先更新主项目契约，再同步副本并运行回归测试。

本站是毕业论文《基于生成式AI使用与感知依赖特征的学业倦怠风险预测系统设计与实现》的线上演示层。研究只有一个预测目标：数据集定义的学业倦怠风险 `Burnout_Risk_Level`。感知AI依赖是12项评估时点特征之一，不是第二个诊断目标。

## 唯一数据与模型口径

- 数据：Kaggle 公开二手数据“Impact of AI on Students”，50,000行、16列。由于抽样、量表施测和标签生成过程不可独立核验，项目按模拟/合成性质二手数据处理。
- 主三分类：随机森林，Low / Medium / High；独立测试集 Accuracy 52.34%、Macro-F1 53.13%、High召回率56.47%。
- 辅助部署任务：逻辑回归 High / Non-high 二分类。全量测试集 Accuracy 81.11%、ROC-AUC 79.90%。
- 选择性规则：`p(High) <= 0.200` 自动输出 Non-high；`p(High) >= 0.515` 自动输出 High；中间区间输出 Review/人工复核。
- 选择性结果：自动受理覆盖率71.24%，受理子集条件Accuracy 86.09%（Wilson 95% CI 85.27%–86.87%），28.76%进入人工复核。
- 必须同时报告：受理子集平衡准确率77.36%、Macro-F1 78.88%、High精确率72.48%、受理样本内High召回率61.49%、全部真实High的自动正确捕获率39.45%。

站点的可机读统一口径位于 `public/project_contract.json`，模型参数和指标分别位于 `public/compact_model.json` 与 `public/model_summary.json`。网页、API和测试从这些版本化文件取数。

## 在线输入契约

`POST /api/predict` 只接受包含精确12个模型输入字段的JSON对象：缺少字段或增加字段都会被拒绝。数值必须是JSON数值且为有限值；`tool_diversity`、`perceived_ai_dependency`、`anxiety_level_during_exams` 必须为整数；分类字段只能使用训练类别；`paid_subscription` 必须为布尔值。

在线部署将 `weekly_genai_hours` 和 `traditional_study_hours` 的范围收紧为0–40小时，以匹配演示表单和本研究的保守输入边界；本地模型 `metadata` 中0–168小时只是更宽的技术可接受上限。两者不是同一层级的业务规则，线上接口按0–40小时执行。

## 功能

- 数据总览与专业/年级精确联合统计；
- 三分类、全量二分类与选择性二分类的分母对照；
- LinearSHAP全局特征重要性；线上预测接口当前不返回个体贡献，个体LinearSHAP由同版辅助智能体提供；
- 12字段表单、High概率、双阈值状态和人工复核提示；
- 非诊断、不持久化个人输入、减少动态及移动端适配。

## 运行与验证

需要 Node.js 22.13 或更高版本。

```bash
npm install
npm test
```

`npm test` 会先生成生产构建，再检查服务端渲染、三类指标口径、双阈值、模型版本和人工复核文案。

## 使用边界

本站只是教育支持与方法验证原型，不构成医学或心理诊断，不得用于处分、标签化或其他自动化不利决定。任何High或Review结果都需由教师、辅导员或具备资质的专业人员结合正式量表、访谈和真实情境复核。
