# AI依赖与学业倦怠风险预测系统

**基于生成式AI使用与感知依赖特征的学业倦怠风险预测系统设计与实现**

本科毕业设计 · 可运行代码仓库　|　作者：胡林康

这是一个**可直接运行**的系统仓库：克隆下来就能启动 Web 端筛查界面，输入 12 项问卷式字段，得到学业倦怠风险的三分类概率、可解释归因和建议。

> ⚠️ **先读完这段再决定要不要用**
>
> 本系统是**教育支持性与研究性原型**，不是医学、临床或心理诊断工具。
> 训练数据为 Kaggle 公开的模拟/合成性质二手数据，抽样过程、量表施测与标签生成方式**无法独立核验**，
> 因此不支持任何因果结论、真实患病率估计或临床推断。
> **严禁**将输出结果用于诊断、处分、奖惩、标签化或其他针对个人的自动化不利决定。
> 完整边界见 [`docs/数据说明与伦理声明.md`](docs/数据说明与伦理声明.md)。

---

## 三分钟跑起来（推荐路线，不用装 Python）

```bash
git clone https://github.com/hulinkang9527/ai-burnout-risk-prediction-system.git
cd ai-burnout-risk-prediction-system/web-demo
npm install
npm run dev
```

打开终端里提示的地址即可使用。这条路线用的是**内置轻量模型**（`public/compact_model.json`，约 12 KB），
预测在站点自身的 Edge 推理路径中完成，**不联网、不连外部 API、不落库**，也没有任何云端依赖。

## 完整研究版（Python，含真实三分类模型与逐个体 SHAP）

```bash
cd prediction-api/backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

export BURNOUT_ARTIFACT_DIR=../../model-artifacts   # Windows: set BURNOUT_ARTIFACT_DIR=..\..\model-artifacts
uvicorn main:app --reload
```

访问 <http://127.0.0.1:8000/> 是同源托管的前端界面，<http://127.0.0.1:8000/docs> 是交互式接口文档。
这一版加载 `model-artifacts/` 中的真实训练产物，提供 Low/Medium/High 三分类概率、
聚合回 12 个原始字段的个体 TreeSHAP 解释，以及专业/年级分组性能审计。

> 说明：环境变量是必需的。原始代码按毕设目录结构默认去找 `03_数据分析/artifacts`，
> 本仓库把模型产物统一放在根目录的 `model-artifacts/`，因此需要用变量指过去，代码本身未做改动。

| 路线 | 适合谁 | 依赖 | 模型 |
| --- | --- | --- | --- |
| `web-demo/` | 想立刻体验界面的人 | Node.js ≥ 22.13 | 内置轻量模型，开箱即用 |
| `prediction-api/` | 想看完整建模链路与 SHAP 解释的人 | Python 3.13 + scikit-learn + shap | 真实随机森林三分类 + 高置信辅助模型 |

---

## 它在解决什么问题

学生使用生成式 AI 的频率、方式与主观依赖感，和学业倦怠之间是否存在可建模的关联？
如果有，能否在学生入学到期末之间**可用的一段信息**上，做出诚实的早期提示？

系统据此设计：只使用评估时点可得的 12 项原始字段（不含学号、不含期末成绩），
输出 Low / Medium / High 三档倦怠风险概率，并**明确告诉用户这个判断依据是什么**。

有意排除的信息：`Student_ID`（标识符泄漏）、`Post_Semester_GPA` 与 `Skill_Retention_Score`（评估时点之后才会产生的结果泄漏）。

---

## 模型结果（照实陈述，不含美化）

### 主模型：三分类随机森林

在独立测试集上：**Accuracy = 0.5234，Macro-F1 = 0.5313，Macro OvR ROC-AUC = 0.7063，High 类召回 = 0.5647**。
同数据集多数类基线为 Accuracy = 0.4229、Macro-F1 = 0.1981。

这个准确率**不高**，这是真实情况而非笔误。它的定位是"提示与排序"工具：
把有限的人力复核资源排到更可能需要关注的那批人身上，而不是替代判断。

### 辅助模型：高风险选择性二分类（Web 端内置的是这一路）

标签重述为 `High=1` / `Low,Medium=0` 后，用逻辑回归加双阈值拒判机制：

- 无拒判全量二分类：Accuracy **81.11%**，平衡准确率 67.72%，Macro-F1 70.12%
- 双阈值 `t_low=0.200` / `t_high=0.515`：**自动接受覆盖率 71.24%**，其余 28.76% 转人工复核
- 自动接受子集条件准确率 **86.09%**（95% Wilson 区间 [85.27%, 86.87%]）
- 相对全部真实 High 样本，自动且正确捕获的比例为 **39.45%**

**请注意两个容易被误用的数字**：
86.09% 是"被自动接受的那部分样本"上的条件准确率，**不是**全样本准确率，也**不是**三分类准确率；
39.45% 意味着自动环节会漏掉约六成真实高风险者，因此筛查流程不能只依赖自动输出。

完整指标、阈值选取依据、5 组随机种子复验与分组审计结果见 [`docs/模型卡.md`](docs/模型卡.md)。

---

## 仓库结构

```
ai-burnout-risk-prediction-system/
├── web-demo/                # 站点：Next/React + vinext，内置轻量模型，npm dev 即用
│   ├── app/                 # 页面与 /api/predict 推理端点
│   ├── public/              # compact_model.json 等静态产物
│   └── tests/               # 接口与渲染回归测试
├── prediction-api/          # 本地研究版：FastAPI + 静态前端
│   ├── backend/             # 模型加载、API 适配、TreeSHAP 局部解释、分组审计
│   └── frontend/            # 同源托管的前端界面
├── model-artifacts/         # 真实训练产物（含 86 MB 的 model.joblib）
├── analysis/                # 复现脚本：完整建模流水线与高置信二分类流水线
└── docs/                    # 模型卡、伦理声明、部署说明、测试报告等
```

`model-artifacts/model.joblib` 有 86 MB，这是为了让 `prediction-api` 克隆后不必重新训练就能跑起来，
属于有意为之。若只想体验界面，`web-demo/` 完全不需要它。

## 文档

| 文档 | 内容 |
| --- | --- |
| [`docs/模型卡.md`](docs/模型卡.md) | 任务边界、数据划分、指标、可解释性、分组评估、限制与维护建议 |
| [`docs/数据说明与伦理声明.md`](docs/数据说明与伦理声明.md) | 数据来源、可核验性边界、隐私处理、禁止用途 |
| [`docs/部署说明.md`](docs/部署说明.md) | 本地与容器化部署步骤 |
| [`docs/用户使用手册.md`](docs/用户使用手册.md) | 界面操作与结果解读 |
| [`docs/系统测试报告_内容稿.md`](docs/系统测试报告_内容稿.md) | 功能、接口、校验与降级测试结论 |
| [`docs/第三方材料与许可.md`](docs/第三方材料与许可.md) | 数据集与依赖许可清单 |

## 数据来源

Kaggle 公开数据集 "Impact of AI on Students"，50,000 条记录、16 列。
原始文件 SHA-256：`4d911088c4b12d60a450a9acae6b606f4119ebbb48679518e427a4fc00778472`。

由于抽样方式、量表施测过程与标签生成规则无法独立核验，本项目**按模拟/合成性质数据对待**，
结论不外推到任何真实院校群体。

## 许可

代码采用 [MIT License](LICENSE)。请以各三方依赖与数据集自身的许可为准，见 [`docs/第三方材料与许可.md`](docs/第三方材料与许可.md)。

---

**If you read only one thing:** this is a research prototype for educational support. It is not a diagnostic device, its headline accuracy is modest on purpose (we refuse to overstate it), and any high-stakes use against a person is out of scope.
