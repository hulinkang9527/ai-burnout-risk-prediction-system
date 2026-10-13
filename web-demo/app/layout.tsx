import type { Metadata } from "next";
import "./globals.css";
import projectContract from "../public/project_contract.json";

export const metadata: Metadata = {
  metadataBase: new URL("https://ai-student-burnout-risk.linkanghu2.chatgpt.site"),
  title: `智学预警｜${projectContract.canonicalTitle}`,
  description: "仅预测数据集定义的学业倦怠风险，感知AI依赖为输入特征。High/Non-high辅助筛查自动受理子集条件准确率86.09%、覆盖率71.24%；原三分类准确率52.34%。",
  icons: { icon: "/favicon.svg", shortcut: "/favicon.svg" },
  openGraph: {
    title: "智学预警｜理解AI使用信号，辅助倦怠风险复核",
    description: "模拟/合成性质二手数据研究原型 · High/Non-high辅助筛查 · 覆盖率71.24% · Review区不作自动判定",
  },
  twitter: { card: "summary" },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="zh-CN"><body>{children}</body></html>;
}
