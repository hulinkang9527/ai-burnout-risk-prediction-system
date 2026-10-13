"use client";

import { CSSProperties, FormEvent, ReactNode, useEffect, useMemo, useRef, useState } from "react";
import modelSummary from "../public/model_summary.json";
import projectContract from "../public/project_contract.json";

const majors = [
  { value: "all", label: "全部专业" },
  { value: "STEM", label: "理工科" },
  { value: "Business", label: "商科" },
  { value: "Humanities", label: "人文社科" },
  { value: "Medical", label: "医学" },
  { value: "Arts", label: "艺术类" },
];

const years = [
  { value: "all", label: "全部年级" },
  { value: "Freshman", label: "大一" },
  { value: "Sophomore", label: "大二" },
  { value: "Junior", label: "大三" },
  { value: "Senior", label: "大四" },
  { value: "Graduate", label: "研究生" },
];

const majorData = [
  { id: "STEM", label: "理工科", n: 15059, burnout: 30.0, ai: 10.49, dep: 3.79 },
  { id: "Business", label: "商科", n: 12538, burnout: 24.3, ai: 8.28, dep: 3.49 },
  { id: "Medical", label: "医学", n: 6476, burnout: 23.2, ai: 7.55, dep: 3.4 },
  { id: "Arts", label: "艺术类", n: 5933, burnout: 22.7, ai: 7.27, dep: 3.36 },
  { id: "Humanities", label: "人文社科", n: 9994, burnout: 20.7, ai: 6.77, dep: 3.24 },
];

const yearData = [
  { id: "Freshman", label: "大一", n: 11031, burnout: 18.0, ai: 8.45, dep: 3.50 },
  { id: "Sophomore", label: "大二", n: 9860, burnout: 22.0, ai: 8.45, dep: 3.54 },
  { id: "Junior", label: "大三", n: 11045, burnout: 24.3, ai: 8.38, dep: 3.50 },
  { id: "Senior", label: "大四", n: 10634, burnout: 27.4, ai: 8.46, dep: 3.49 },
  { id: "Graduate", label: "研究生", n: 7430, burnout: 36.7, ai: 8.41, dep: 3.50 },
];

const relation = [
  { bin: "0–4h", n: 18998, gpa: 3.339, burnout: 9.8, dep: 2.57 },
  { bin: "4–8h", n: 11746, gpa: 3.361, burnout: 16.6, dep: 3.1 },
  { bin: "8–12h", n: 7143, gpa: 3.381, burnout: 25.6, dep: 3.66 },
  { bin: "12h+", n: 12113, gpa: 3.335, burnout: 56.5, dep: 5.28 },
];

const summaryMetrics = modelSummary.metrics;
const shap = modelSummary.shapFeatures.slice(0, 6).map((item) => ({
  label: item.label,
  value: item.normalized_importance * 100,
}));

const models = [
  {
    name: "High/Non-high 选择性逻辑回归",
    accuracy: summaryMetrics.selective_accuracy * 100,
    f1: summaryMetrics.selective_macro_f1 * 100,
    scope: `自动受理覆盖 ${(summaryMetrics.selective_coverage * 100).toFixed(2)}%`,
    accuracyLabel: "受理子集条件准确率",
    selected: true,
  },
  {
    name: "全量高风险二分类",
    accuracy: summaryMetrics.overall_accuracy * 100,
    f1: summaryMetrics.overall_macro_f1 * 100,
    scope: "不拒判的全部样本",
    accuracyLabel: "全量准确率",
    selected: false,
  },
  {
    name: "原三分类随机森林",
    accuracy: summaryMetrics.legacy_three_class_accuracy * 100,
    f1: summaryMetrics.legacy_three_class_macro_f1 * 100,
    scope: "Low / Medium / High 基线",
    accuracyLabel: "三分类准确率",
    selected: false,
  },
];

const jointData = [
  { major: "Arts", year: "Freshman", n: 1282, burnout: 16.38, ai: 7.3763, dep: 3.3432 },
  { major: "Arts", year: "Graduate", n: 844, burnout: 34.00, ai: 7.3516, dep: 3.4088 },
  { major: "Arts", year: "Junior", n: 1324, burnout: 21.60, ai: 7.1311, dep: 3.3248 },
  { major: "Arts", year: "Senior", n: 1301, burnout: 25.75, ai: 7.3034, dep: 3.3951 },
  { major: "Arts", year: "Sophomore", n: 1182, burnout: 19.29, ai: 7.2238, dep: 3.3613 },
  { major: "Business", year: "Freshman", n: 2831, burnout: 17.41, ai: 8.2203, dep: 3.4465 },
  { major: "Business", year: "Graduate", n: 1845, burnout: 35.99, ai: 8.3564, dep: 3.5127 },
  { major: "Business", year: "Junior", n: 2845, burnout: 24.08, ai: 8.3398, dep: 3.5283 },
  { major: "Business", year: "Senior", n: 2550, burnout: 27.45, ai: 8.1891, dep: 3.4341 },
  { major: "Business", year: "Sophomore", n: 2467, burnout: 20.63, ai: 8.2954, dep: 3.5330 },
  { major: "Humanities", year: "Freshman", n: 2174, burnout: 13.34, ai: 6.6438, dep: 3.2061 },
  { major: "Humanities", year: "Graduate", n: 1520, burnout: 33.68, ai: 6.8670, dep: 3.2783 },
  { major: "Humanities", year: "Junior", n: 2201, burnout: 19.81, ai: 6.6797, dep: 3.2363 },
  { major: "Humanities", year: "Senior", n: 2155, burnout: 22.92, ai: 6.7792, dep: 3.2107 },
  { major: "Humanities", year: "Sophomore", n: 1944, burnout: 17.54, ai: 6.9281, dep: 3.3035 },
  { major: "Medical", year: "Freshman", n: 1464, burnout: 16.33, ai: 7.6990, dep: 3.3702 },
  { major: "Medical", year: "Graduate", n: 953, burnout: 36.94, ai: 7.8238, dep: 3.4669 },
  { major: "Medical", year: "Junior", n: 1396, burnout: 21.99, ai: 7.5072, dep: 3.4391 },
  { major: "Medical", year: "Senior", n: 1355, burnout: 25.02, ai: 7.5140, dep: 3.3277 },
  { major: "Medical", year: "Sophomore", n: 1308, burnout: 20.11, ai: 7.2506, dep: 3.4159 },
  { major: "STEM", year: "Freshman", n: 3280, burnout: 22.90, ai: 10.5901, dep: 3.8442 },
  { major: "STEM", year: "Graduate", n: 2268, burnout: 40.26, ai: 10.1149, dep: 3.6896 },
  { major: "STEM", year: "Junior", n: 3279, burnout: 29.70, ai: 10.4197, dep: 3.7612 },
  { major: "STEM", year: "Senior", n: 3273, burnout: 31.99, ai: 10.6267, dep: 3.8283 },
  { major: "STEM", year: "Sophomore", n: 2959, burnout: 28.12, ai: 10.5862, dep: 3.8158 },
];

type Risk = {
  label: "非高风险" | "需人工复核" | "高风险";
  score: number;
  tone: string;
  summary: string;
};
type PredictionResponse = {
  predicted_class: "NonHigh" | "Review" | "High";
  decision_status: "auto" | "review";
  high_risk_probability: number;
  confidence: number;
  probabilities: Record<"NonHigh" | "High", number>;
  summary: string;
  recommendations: string[];
  disclaimer: string;
};

function ParticleField() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;

    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const pointer = { x: -1000, y: -1000 };
    let width = 0;
    let height = 0;
    let frame = 0;
    let particles: Array<{ x: number; y: number; vx: number; vy: number; r: number }> = [];

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      width = rect.width;
      height = rect.height;
      canvas.width = width * ratio;
      canvas.height = height * ratio;
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      const count = Math.max(18, Math.min(42, Math.floor(width / 34)));
      particles = Array.from({ length: count }, (_, index) => ({
        x: (index * 97.3) % Math.max(width, 1),
        y: (index * 53.7) % Math.max(height, 1),
        vx: ((index % 5) - 2) * 0.055,
        vy: (((index * 3) % 5) - 2) * 0.045,
        r: 1.2 + (index % 3) * 0.45,
      }));
    };

    const draw = () => {
      context.clearRect(0, 0, width, height);
      particles.forEach((particle, index) => {
        if (!reduceMotion) {
          particle.x += particle.vx;
          particle.y += particle.vy;
          if (particle.x < -8) particle.x = width + 8;
          if (particle.x > width + 8) particle.x = -8;
          if (particle.y < -8) particle.y = height + 8;
          if (particle.y > height + 8) particle.y = -8;
          const dx = pointer.x - particle.x;
          const dy = pointer.y - particle.y;
          if (dx * dx + dy * dy < 11000) {
            particle.x -= dx * 0.00045;
            particle.y -= dy * 0.00045;
          }
        }
        context.beginPath();
        context.arc(particle.x, particle.y, particle.r, 0, Math.PI * 2);
        context.fillStyle = index % 4 === 0 ? "rgba(91,143,249,.42)" : "rgba(23,105,232,.22)";
        context.fill();
        for (let next = index + 1; next < particles.length; next += 1) {
          const other = particles[next];
          const dx = particle.x - other.x;
          const dy = particle.y - other.y;
          const distance = Math.sqrt(dx * dx + dy * dy);
          if (distance < 88) {
            context.beginPath();
            context.moveTo(particle.x, particle.y);
            context.lineTo(other.x, other.y);
            context.strokeStyle = `rgba(23,105,232,${0.11 * (1 - distance / 88)})`;
            context.lineWidth = 0.7;
            context.stroke();
          }
        }
      });
      if (!reduceMotion) frame = window.requestAnimationFrame(draw);
    };

    const onPointerMove = (event: PointerEvent) => {
      const rect = canvas.getBoundingClientRect();
      pointer.x = event.clientX - rect.left;
      pointer.y = event.clientY - rect.top;
    };
    const onPointerLeave = () => {
      pointer.x = -1000;
      pointer.y = -1000;
    };

    resize();
    draw();
    window.addEventListener("resize", resize);
    canvas.addEventListener("pointermove", onPointerMove);
    canvas.addEventListener("pointerleave", onPointerLeave);
    return () => {
      window.cancelAnimationFrame(frame);
      window.removeEventListener("resize", resize);
      canvas.removeEventListener("pointermove", onPointerMove);
      canvas.removeEventListener("pointerleave", onPointerLeave);
    };
  }, []);

  return <canvas ref={canvasRef} className="particleField" aria-hidden="true" />;
}

function Magnet({ children, className = "", padding = 90, strength = 12 }: { children: ReactNode; className?: string; padding?: number; strength?: number }) {
  const elementRef = useRef<HTMLDivElement>(null);
  const [position, setPosition] = useState({ x: 0, y: 0, active: false });

  useEffect(() => {
    const onPointerMove = (event: PointerEvent) => {
      const element = elementRef.current;
      if (!element || window.matchMedia("(pointer: coarse)").matches) return;
      const rect = element.getBoundingClientRect();
      const inRange = event.clientX >= rect.left - padding && event.clientX <= rect.right + padding && event.clientY >= rect.top - padding && event.clientY <= rect.bottom + padding;
      if (!inRange) {
        if (position.active) setPosition({ x: 0, y: 0, active: false });
        return;
      }
      setPosition({
        x: (event.clientX - rect.left - rect.width / 2) / strength,
        y: (event.clientY - rect.top - rect.height / 2) / strength,
        active: true,
      });
    };
    window.addEventListener("pointermove", onPointerMove, { passive: true });
    return () => window.removeEventListener("pointermove", onPointerMove);
  }, [padding, strength, position.active]);

  return <div ref={elementRef} className={`magnetWrap ${className}`} style={{ transform: `translate3d(${position.x}px, ${position.y}px, 0)`, transition: position.active ? "transform .24s ease-out" : "transform .55s ease-in-out" }}>{children}</div>;
}

function IntroSequence() {
  const [visible, setVisible] = useState(true);

  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      const reducedMotionTimer = window.setTimeout(() => setVisible(false), 0);
      return () => window.clearTimeout(reducedMotionTimer);
    }
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const timer = window.setTimeout(() => {
      setVisible(false);
      document.body.style.overflow = previousOverflow;
    }, 4700);
    return () => {
      window.clearTimeout(timer);
      document.body.style.overflow = previousOverflow;
    };
  }, []);

  const skipIntro = () => {
    setVisible(false);
    document.body.style.overflow = "";
  };

  if (!visible) return null;

  return (
    <div className="introSequence" aria-label="系统欢迎动画">
      <div className="introPanel introPanelA" aria-hidden="true" />
      <div className="introPanel introPanelB" aria-hidden="true" />
      <button className="introSkip liquid-glass" type="button" onClick={skipIntro}>跳过动画</button>
      <div className="introStage">
        <div className="introLogoCrop">
          {/* The intro animation relies on the source image's exact intrinsic crop. */}
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src="/system-logo.png" alt={`${projectContract.shortName}系统标志`} />
        </div>
        <p className="introWelcome">欢迎来到{projectContract.shortName}研究原型</p>
        <div className="introParticles" aria-hidden="true">
          {Array.from({ length: 54 }, (_, index) => (
            <i
              key={index}
              style={{
                "--particle-x": `${((index * 47) % 320) - 160}px`,
                "--particle-y": `${((index * 73) % 190) - 95}px`,
                "--particle-delay": `${(index % 9) * 0.024}s`,
                left: `${4 + ((index * 37) % 92)}%`,
                top: `${18 + ((index * 53) % 64)}%`,
              } as CSSProperties}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

function CinematicLanding() {
  return (
    <section className="cinematicLanding" id="landing" aria-label="系统欢迎首页">
      <video className="cinematicVideo" autoPlay loop muted playsInline aria-hidden="true">
        <source src="https://d8j0ntlcm91z4.cloudfront.net/user_38xzZboKViGWJOttwIXH07lWA1P/hf_20260314_131748_f2ca2a28-fed7-44c8-b9a9-bd9acdd5ec31.mp4" type="video/mp4" />
      </video>
      <div className="cinematicContent">
        <p className="cinematicKicker animate-fade-rise">STUDENT AI &amp; ACADEMIC WELLBEING</p>
        <h1 className="animate-fade-rise">
          理解 <em>AI 使用信号</em>，<br />辅助 <em>倦怠风险复核。</em>
        </h1>
        <p className="cinematicLead animate-fade-rise-delay">
          基于 50,000 条 Kaggle 公开二手记录与可解释机器学习，只识别数据集定义的学业倦怠风险；感知 AI 依赖仅为输入特征。
        </p>
        <div className="cinematicActions animate-fade-rise-delay-2">
          <a className="liquid-glass journeyButton" href="#home">进入系统 <span>↓</span></a>
          <a className="cinematicTextLink" href="#assessment">直接开始评估 <span>↗</span></a>
        </div>
      </div>
      <p className="cinematicMeta">研究原型 · 模拟/合成性质二手数据 · 非诊断用途</p>
    </section>
  );
}

export default function Home() {
  const [major, setMajor] = useState("all");
  const [year, setYear] = useState("all");
  const [risk, setRisk] = useState<Risk | null>(null);
  const [activeSection, setActiveSection] = useState("home");
  const [formStage, setFormStage] = useState(1);
  const [dependency, setDependency] = useState(4);
  const [anxiety, setAnxiety] = useState(5);
  const [modelChoice, setModelChoice] = useState(models[0].name);
  const [predictionStatus, setPredictionStatus] = useState<"idle" | "loading" | "error">("idle");
  const [predictionError, setPredictionError] = useState("");

  useEffect(() => {
    const sections = ["home", "dashboard", "model", "assessment"]
      .map((id) => document.getElementById(id))
      .filter(Boolean) as HTMLElement[];
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((entry) => entry.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
        if (visible?.target.id) setActiveSection(visible.target.id);
      },
      { rootMargin: "-18% 0px -58%", threshold: [0.05, 0.25, 0.5] },
    );
    sections.forEach((section) => observer.observe(section));
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const items = Array.from(document.querySelectorAll<HTMLElement>(".reveal"));
    const observer = new IntersectionObserver(
      (entries) => entries.forEach((entry) => entry.isIntersecting && entry.target.classList.add("isVisible")),
      { threshold: 0.08 },
    );
    items.forEach((item) => observer.observe(item));
    return () => observer.disconnect();
  }, []);

  const selected = useMemo(() => {
    const m = majorData.find((item) => item.id === major);
    const y = yearData.find((item) => item.id === year);
    const joint = m && y ? jointData.find((item) => item.major === m.id && item.year === y.id) : undefined;
    return {
      sample: joint?.n ?? m?.n ?? y?.n ?? projectContract.data.rows,
      burnout: joint?.burnout ?? m?.burnout ?? y?.burnout ?? 24.974,
      ai: joint?.ai ?? m?.ai ?? y?.ai ?? 8.4278,
      dep: joint?.dep ?? m?.dep ?? y?.dep ?? 3.5054,
    };
  }, [major, year]);

  async function predict(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    setPredictionStatus("loading");
    setPredictionError("");
    setRisk(null);

    const payload = {
      major_category: String(form.get("major")),
      year_of_study: String(form.get("year")),
      pre_semester_gpa: Number(form.get("gpa")),
      weekly_genai_hours: Number(form.get("aiHours")),
      primary_use_case: String(form.get("primaryUseCase")),
      prompt_engineering_skill: String(form.get("promptSkill")),
      tool_diversity: Number(form.get("toolDiversity")),
      paid_subscription: String(form.get("paidSubscription")) === "true",
      traditional_study_hours: Number(form.get("studyHours")),
      perceived_ai_dependency: dependency,
      institutional_policy: String(form.get("institutionalPolicy")),
      anxiety_level_during_exams: anxiety,
    };

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        const problem = await response.json().catch(() => null) as { detail?: string } | null;
        throw new Error(problem?.detail || "真实预测服务暂时不可用");
      }
      const result = await response.json() as PredictionResponse;
      const labelMap = {
        NonHigh: "非高风险",
        Review: "需人工复核",
        High: "高风险",
      } as const;
      const toneMap = { NonHigh: "low", Review: "medium", High: "high" } as const;
      const highRiskProbability = Math.round(result.high_risk_probability * 100);
      setRisk({
        label: labelMap[result.predicted_class],
        score: highRiskProbability,
        tone: toneMap[result.predicted_class],
        summary: `${result.summary} ${result.recommendations.slice(0, 2).join(" ")}`,
      });
      setFormStage(3);
      setPredictionStatus("idle");
    } catch (error) {
      setPredictionStatus("error");
      setPredictionError(error instanceof Error ? error.message : "真实预测服务暂时不可用");
    }
  }

  const navItems = [
    { id: "dashboard", label: "数据看板" },
    { id: "model", label: "模型表现" },
    { id: "assessment", label: "风险评估" },
  ];
  const aiDelta = selected.ai - 8.43;
  const depDelta = selected.dep - 3.51;
  const selectedModel = models.find((item) => item.name === modelChoice) ?? models[0];
  const behaviorSignal = Math.round((dependency + anxiety) / 2);

  return (
    <main>
      <IntroSequence />
      <nav className="nav" aria-label="主导航">
        <a className="brand" href="#landing" onClick={() => setActiveSection("home")} aria-label="智学预警，返回欢迎首页"><span>AI</span><b>智学预警</b></a>
        <div className="navlinks">
          {navItems.map((item) => <a key={item.id} href={`#${item.id}`} className={activeSection === item.id ? "active" : ""} aria-current={activeSection === item.id ? "page" : undefined} onClick={() => setActiveSection(item.id)}>{item.label}</a>)}
        </div>
        <a className="navCta liquid-glass" href="#assessment">开始评估 <span>→</span></a>
      </nav>

      <CinematicLanding />

      <section className="hero" id="home">
        <ParticleField />
        <div className="heroCopy reveal isVisible">
          <p className="eyebrow"><i /> 基于 50,000 条公开二手记录</p>
          <h1>理解 AI 使用信号，<br /><em>辅助倦怠风险复核。</em></h1>
          <p className="lead">系统只预测数据集定义的学业倦怠风险，感知 AI 依赖仅为输入特征。High / Non-high 辅助筛查对不确定记录明确转入人工复核。</p>
          <div className="heroActions">
            <Magnet padding={58} strength={10}><a className="primary" href="#assessment">开始风险评估 <span>→</span></a></Magnet>
            <Magnet padding={48} strength={14}><a className="secondary" href="#dashboard">查看数据看板 <span>↗</span></a></Magnet>
          </div>
          <div className="trust" aria-label="项目说明"><span><b>研</b>研究原型</span><span><b>二</b>二手数据</span><span><b>非</b>非诊断用途</span></div>
        </div>
        <div className="heroVisual reveal isVisible" aria-label="研究指标概览">
          <div className="orb orbA" />
          <Magnet className="mainCardMagnet" padding={120} strength={20}><article className="glassCard mainCard">
            <div className="cardTop"><span className="miniIcon">◎</span><small>学业状态洞察</small><b>实时演示</b></div>
            <div className="gauge"><div className="gaugeInner"><strong>25.0%</strong><span>数据集 High 标签占比</span></div></div>
            <div className="trendHead"><span>AI使用时长增加</span><b>倦怠风险趋势 ↑</b></div>
            <div className="miniTrend" aria-label="AI使用时长增加与倦怠风险变化趋势">
              {relation.map((item) => <i key={item.bin} style={{ height: `${Math.max(18, item.burnout * 1.25)}%` }} title={`${item.bin}：高倦怠风险 ${item.burnout}%`} />)}
            </div>
            <div className="miniStats"><div><span>平均AI使用</span><b>8.43 <i>小时/周</i></b></div><div><span>平均学期后GPA</span><b>3.35 <i>/ 4.0（仅描述）</i></b></div></div>
          </article></Magnet>
          <article className="floatCard floatTop"><span>模型 AUC</span><strong>{summaryMetrics.roc_auc.toFixed(3)}</strong><i>High vs 非 High</i></article>
          <article className="floatCard floatBottom"><span>核心特征</span><strong>{shap[0].label}</strong><i>SHAP {shap[0].value.toFixed(1)}%</i></article>
        </div>
      </section>

      <section className="dashboard section" id="dashboard">
        <div className="sectionHead reveal">
          <div><p className="eyebrow"><i /> DATA OVERVIEW</p><h2>二手数据可视化看板</h2><p>筛选不同专业与年级，观察数据中AI使用、感知依赖与High标签的描述性差异；不解释为真实人群患病率。</p></div>
          <div className="filters">
            <label>专业<select value={major} onChange={(event) => setMajor(event.target.value)}>{majors.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}</select></label>
            <label>年级<select value={year} onChange={(event) => setYear(event.target.value)}>{years.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}</select></label>
          </div>
        </div>

        <div className="kpiGrid reveal">
          <article><span className="kpiIcon blue">人</span><div><small>当前样本</small><strong>{selected.sample.toLocaleString()}</strong><em>覆盖 5 类专业 · 5 个年级</em></div></article>
          <article><span className="kpiIcon cyan">AI</span><div><small>平均AI使用</small><strong>{selected.ai.toFixed(2)} <i>小时/周</i></strong><em>较总体均值 {aiDelta >= 0 ? "+" : ""}{aiDelta.toFixed(2)} 小时</em></div></article>
          <article><span className="kpiIcon purple">依</span><div><small>平均依赖评分</small><strong>{selected.dep.toFixed(2)} <i>/ 10</i></strong><em>较总体均值 {depDelta >= 0 ? "+" : ""}{depDelta.toFixed(2)}</em></div></article>
          <article className="riskKpi"><span className="kpiIcon coral">标</span><div><small>High标签占比</small><strong>{selected.burnout.toFixed(1)} <i>%</i></strong><em>当前筛选记录的描述性比例</em></div></article>
        </div>

        <div className="chartGrid reveal">
          <article className="panel relationPanel">
            <div className="panelTitle"><div><span>关系分析</span><h3>AI使用时长、学期后GPA与High标签</h3></div><div className="legend"><i className="dot blueDot" /> High标签占比 <i className="dot purpleDot" /> 平均学期后GPA（不入模）</div></div>
            <div className="relationChart">
              <div className="axisLabels"><span>60%</span><span>40%</span><span>20%</span><span>0%</span></div>
              <div className="gridLines"><i /><i /><i /><i /></div>
              <div className="columns">
                {relation.map((item) => <div className="column" key={item.bin} tabIndex={0} aria-label={`${item.bin}，样本 ${item.n.toLocaleString()}，平均 GPA ${item.gpa.toFixed(3)}，高倦怠风险率 ${item.burnout}%`}>
                  <span className="chartTooltip"><b>{item.bin}</b><em>样本人数 {item.n.toLocaleString()}</em><em>平均 GPA {item.gpa.toFixed(3)}</em><em>高风险率 {item.burnout}%</em></span>
                  <span className="gpaPoint" style={{ bottom: `${(item.gpa - 3.25) * 170 + 39}%` }} />
                  <div className="bar" style={{ height: `${item.burnout * 1.45}%` }}><b>{item.burnout}%</b></div>
                  <strong>{item.bin}</strong><small>n={item.n.toLocaleString()}</small>
                </div>)}
              </div>
            </div>
            <div className="insight"><span className="insightIcon">↗</span><p><b>描述性发现</b>数据中每周AI使用 12 小时以上组的High标签占比为 56.5%，学期后GPA未随使用时长持续上升。<mark>仅限该二手数据，不证明因果</mark></p></div>
          </article>

          <article className="panel groupPanel">
            <div className="panelTitle"><div><span>群体比较</span><h3>各专业High标签占比</h3></div><small>全部年级</small></div>
            <div className="hBars">{majorData.map((item, index) => <div className={major === item.id ? "active" : ""} key={item.id}><b className="rank">{String(index + 1).padStart(2, "0")}</b><span>{item.label}</span><i><b style={{ width: `${item.burnout * 2.65}%` }} /></i><strong>{item.burnout.toFixed(1)}%</strong></div>)}</div>
            <div className="note"><span>i</span> 理工科组High标签占比最高；该差异可能来自数据生成、年级结构或其他未观测因素。</div>
          </article>
        </div>
      </section>

      <section className="model section" id="model">
        <div className="sectionHead lightHead reveal"><div><p className="eyebrow"><i /> MODEL & EXPLAINABILITY</p><h2>模型表现与可解释性</h2><p>86.09%仅对应 High / Non-high 辅助任务的自动受理子集，不是三分类或全样本准确率。</p></div><label className="modelSelector"><span>参考口径</span><select value={modelChoice} onChange={(event) => setModelChoice(event.target.value)}>{models.map((item) => <option key={item.name} value={item.name}>{item.name}</option>)}</select></label></div>
        <div className="modelGrid reveal">
          <article className="panel modelCompare"><div className="panelTitle"><div><span>独立测试集</span><h3>不同任务口径对照</h3></div><small>固定 10,000 条留出样本</small></div>
            <div className="selectedModelCard" aria-live="polite"><div><small>当前参考口径</small><strong>{selectedModel.name}</strong></div><div><small>{selectedModel.accuracyLabel}</small><b>{selectedModel.accuracy.toFixed(2)}%</b></div><div><small>Macro-F1</small><b>{selectedModel.f1.toFixed(2)}%</b></div></div>
            <div className="modelBars">{models.map((item) => <div key={item.name} className={item.name === modelChoice ? "selected" : ""}><span>{item.name}{item.selected ? <b>推荐</b> : item.name === modelChoice ? <b className="currentBadge">当前</b> : null}</span><i><em style={{ width: `${item.accuracy}%` }} /></i><strong>{item.accuracy.toFixed(2)}%</strong><small>Macro-F1 {item.f1.toFixed(2)}% · {item.scope}</small></div>)}</div>
            <p className="baseline">自动受理子集条件准确率 {(summaryMetrics.selective_accuracy * 100).toFixed(2)}%（Wilson 95% CI {(summaryMetrics.selective_wilson_lower_95 * 100).toFixed(2)}%–{(summaryMetrics.selective_wilson_upper_95 * 100).toFixed(2)}%），覆盖 {(summaryMetrics.selective_coverage * 100).toFixed(2)}%；其余 {(summaryMetrics.abstention_rate * 100).toFixed(2)}% 进入人工复核。受理子集 Macro-F1 {(summaryMetrics.selective_macro_f1 * 100).toFixed(2)}%，平衡准确率 {(summaryMetrics.selective_balanced_accuracy * 100).toFixed(2)}%。High 精确率 {(summaryMetrics.high_precision * 100).toFixed(2)}%，受理样本内 High 召回率 {(summaryMetrics.high_recall_within_accepted * 100).toFixed(2)}%，全部真实 High 的自动正确捕获率 {(summaryMetrics.auto_high_recall_all_high * 100).toFixed(2)}%。全量二分类准确率为 {(summaryMetrics.overall_accuracy * 100).toFixed(2)}%，原三分类随机森林仍为 {(summaryMetrics.legacy_three_class_accuracy * 100).toFixed(2)}%，不同任务不可直接等同。</p>
            <p className="baseline">判定规则：p(High) ≤ 20.0% 时自动输出 Non-high，p(High) ≥ 51.5% 时自动输出 High；20.0%–51.5% 为 Review 区，本系统不作自动判定。</p>
          </article>
          <article className="panel shapPanel"><div className="panelTitle"><div><span>全局解释</span><h3>SHAP 特征重要性</h3></div><small>归一化贡献</small></div>
            <div className="shapBars">{shap.map((item, index) => <div key={item.label}><b>{index + 1}</b><span>{item.label}</span><i><em style={{ width: `${Math.min(100, item.value * 3.8)}%` }} /></i><strong>{item.value.toFixed(1)}%</strong></div>)}</div>
            <p className="baseline">LinearSHAP 解释模型行为与关联，不代表特征会因果地导致倦怠；工程特征贡献按其原始父字段均分展示。</p>
          </article>
        </div>
      </section>

      <section className="assessment section" id="assessment">
        <ParticleField />
        <div className="assessmentIntro reveal"><p className="eyebrow"><i /> PERSONAL ASSESSMENT</p><h2>学业倦怠风险评估</h2><p>填写当前常态每周情况；近期波动较大时可按近4周平均。结果仅作教育支持参考，全程无需姓名或学号。</p>
          <div className="behaviorVisual" style={{ "--dependencyShift": `${dependency * 1.3}%`, "--anxietyShift": `${anxiety * 1.1}%` } as CSSProperties} aria-label={`当前行为信号 ${behaviorSignal}/10`}><div className="orbit orbitOne" /><div className="orbit orbitTwo" /><span className="node nodeAi">AI</span><span className="node nodeStudy">学习</span><span className="node nodeRest">节律</span><b><span>实时行为信号</span><strong>{behaviorSignal}<i>/10</i></strong></b></div>
          <div className="liveSignalGrid" aria-live="polite"><div><span>AI 依赖</span><b>{dependency}/10</b><i><em style={{ width: `${dependency * 10}%` }} /></i></div><div><span>考试焦虑</span><b>{anxiety}/10</b><i><em style={{ width: `${anxiety * 10}%` }} /></i></div></div>
          <ul><li><span>01</span><div><b>无需身份信息</b><small>不要求姓名或学号</small></div></li><li><span>02</span><div><b>即时反馈</b><small>应用代码不写入业务数据库</small></div></li><li><span>03</span><div><b>审慎解释</b><small>不替代专业评估</small></div></li></ul>
        </div>
        <form className="assessmentCard reveal" onSubmit={predict} onFocusCapture={(event) => {
          const name = (event.target as unknown as { name?: string }).name ?? "";
          if (["major", "year", "gpa"].includes(name)) setFormStage(1);
          if (["aiHours", "studyHours", "dependency", "anxiety"].includes(name)) setFormStage(2);
        }}>
          <div className="progressNav" aria-label="评估进度">
            {["基本信息", "学习行为", "评估结果"].map((label, index) => <div key={label} className={formStage >= index + 1 ? "active" : ""}><span>{index + 1}</span><b>{label}</b></div>)}
          </div>
          <div className="formHead"><div><span>第 {formStage} 步 · {formStage === 1 ? "基本信息" : formStage === 2 ? "学习行为" : "评估结果"}</span><h3>学生学习状态信息</h3></div><small>约 1 分钟</small></div>
          <div className="formGrid">
            <label>专业类别<select name="major" required defaultValue=""><option value="" disabled>请选择专业</option>{majors.slice(1).map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}</select></label>
            <label>所在年级<select name="year" required defaultValue=""><option value="" disabled>请选择年级</option>{years.slice(1).map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}</select></label>
            <label>上学期 GPA <span>0–4.0</span><input name="gpa" type="number" min="0" max="4" step="0.01" placeholder="如 3.20" required /></label>
            <label>每周 AI 使用时长 <span>小时</span><input name="aiHours" type="number" min="0" max="40" step="0.5" placeholder="如 8" required /></label>
            <label>每周传统学习时长 <span>小时</span><input name="studyHours" type="number" min="0" max="40" step="0.5" placeholder="如 12" required /></label>
            <label>AI 主要用途<select name="primaryUseCase" required defaultValue=""><option value="" disabled>请选择主要用途</option><option value="Copywriting/Drafting">写作与起草</option><option value="Ideation">创意构思</option><option value="Summarizing_Reading">阅读总结</option><option value="Debugging/Troubleshooting">调试与问题排查</option><option value="Direct_Answer_Generation">直接生成答案</option></select></label>
            <label>提示工程能力<select name="promptSkill" required defaultValue=""><option value="" disabled>请选择熟练程度</option><option value="Beginner">初级</option><option value="Intermediate">中级</option><option value="Advanced">高级</option></select></label>
            <label>使用过的 AI 工具数量 <span>1–5</span><input name="toolDiversity" type="number" min="1" max="5" step="1" placeholder="如 3" required /></label>
            <label>是否付费订阅<select name="paidSubscription" required defaultValue=""><option value="" disabled>请选择</option><option value="false">否</option><option value="true">是</option></select></label>
            <label className="wide">学校 AI 使用政策<select name="institutionalPolicy" required defaultValue=""><option value="" disabled>请选择学校政策</option><option value="Allowed_With_Citation">允许但须引用</option><option value="Strict_Ban">严格禁止</option><option value="Actively_Encouraged">积极鼓励</option></select></label>
            <label>感知 AI 依赖 <output className="rangeValue">{dependency}/10</output><input name="dependency" type="range" min="1" max="10" value={dependency} onChange={(event) => setDependency(Number(event.target.value))} /></label>
            <label className="wide">考试焦虑程度 <output className="rangeValue">{anxiety}/10</output><input name="anxiety" type="range" min="1" max="10" value={anxiety} onChange={(event) => setAnxiety(Number(event.target.value))} /></label>
          </div>
          <p className="formHint">提交后由选择性逻辑回归模型完成 High / Non-high 辅助筛查；高风险概率位于 20.0%–51.5% 时明确输出“需人工复核”，不会自动判定。结果不是医学或心理诊断。</p>
          <button className="submit" type="submit" disabled={predictionStatus === "loading"}>{predictionStatus === "loading" ? "正在调用真实模型…" : "生成风险评估"} <span>→</span></button>
          {predictionStatus === "error" && <p className="predictionError" role="alert">本次没有生成风险结论：{predictionError}</p>}
          {risk ? <div className={`result ${risk.tone}`} aria-live="polite"><div className="resultScore" style={{ "--score": `${risk.score * 3.6}deg` } as CSSProperties}><div><span>{risk.label}</span><strong>{risk.score}<i>%高风险概率</i></strong></div></div><p><b>模型评估与建议</b>{risk.summary}<button className="resultReset" type="button" onClick={() => { setRisk(null); setPredictionStatus("idle"); setPredictionError(""); setFormStage(1); setDependency(4); setAnxiety(5); }}>重新评估</button></p></div> : <p className="privacy">12项输入会发送至同源预测接口；应用代码不写入业务数据库，也不在浏览器中持久保存</p>}
        </form>
      </section>

      <footer><div className="brand"><span>AI</span><b>{projectContract.shortName}</b></div><p>{projectContract.canonicalTitle} · 教育数据研究原型</p><div><a href="#home">返回顶部 ↑</a></div></footer>
    </main>
  );
}
