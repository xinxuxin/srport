"use client";

import { motion } from "framer-motion";
import { useEffect, useMemo, useState } from "react";

import { fetchModelInfo, ModelInfo } from "../lib/api";
import { useLanguage } from "./language-provider";
import { MetricCard } from "./metric-card";
import { ModelExplainer } from "./model-explainer";

const fadeUp = {
  initial: { opacity: 0, y: 18 },
  animate: { opacity: 1, y: 0 }
};

const copy = {
  en: {
    eyebrow: "Model Page",
    title: "Architecture, deployment metadata, and explainability in one place.",
    body:
      "This page turns EPNet from a paper implementation into a presentable solution asset: architecture summary, deployment metadata, and a customer-friendly talk track.",
    version: "Version",
    deviceTarget: "Device target",
    gitCommit: "Git commit",
    apiVersion: "API version",
    deployment: "Deployment metadata",
    checkpointSource: "Checkpoint source",
    checkpointPath: "Checkpoint path",
    buildTime: "Build time",
    paperTitle: "Paper title",
    noCheckpoint: "No checkpoint configured",
    presenterNotes: "Presenter notes",
    note1:
      "Lead with the business framing: EPNet is positioned as an efficient SR backbone, not a brute-force quality-only model.",
    note2:
      "Then explain the three-stage flow: detail extraction with PFEM, multiscale context with ESPM, and low-cost reconstruction with PixelShuffle.",
    note3:
      "Finally connect to deployment: versioning, checkpoint provenance, build metadata, and runtime target are surfaced as first-class product attributes.",
    availableCheckpoints: "Available checkpoints",
    availableCheckpointsBody:
      "Use this list to explain which scales or training artifacts are currently deployed in the demo.",
    loadError: "Unable to load model metadata."
  },
  zh: {
    eyebrow: "模型解释页",
    title: "把架构、部署版本信息和可解释叙事放在同一个页面里。",
    body:
      "这个页面把 EPNet 从“论文复现代码”变成更像解决方案资产的形态：既能讲清架构，也能讲清部署版本、权重来源和面向客户的演示话术。",
    version: "模型版本",
    deviceTarget: "目标设备",
    gitCommit: "Git 提交",
    apiVersion: "API 版本",
    deployment: "部署信息",
    checkpointSource: "Checkpoint 来源",
    checkpointPath: "Checkpoint 路径",
    buildTime: "构建时间",
    paperTitle: "论文标题",
    noCheckpoint: "当前未配置 checkpoint",
    presenterNotes: "演示讲解提纲",
    note1:
      "先讲业务定位：EPNet 强调的是高效的超分辨率主干网络，而不是只追求峰值画质的重型模型。",
    note2:
      "再讲三段式结构：PFEM 负责细节提取，ESPM 负责多尺度上下文建模，最后通过 PixelShuffle 完成低成本重建。",
    note3:
      "最后连到部署视角：版本、权重来源、构建时间和运行目标，都已经被提升为产品级元数据。",
    availableCheckpoints: "可用 checkpoints",
    availableCheckpointsBody:
      "这里可以直接展示当前 Demo 实际挂载了哪些权重、支持哪些倍率，用来回答“这是不是可部署资产”的问题。",
    loadError: "无法加载模型元数据。"
  }
} as const;

export function ModelPage() {
  const { language } = useLanguage();
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const text = useMemo(() => copy[language], [language]);

  useEffect(() => {
    async function load() {
      try {
        setModel(await fetchModelInfo());
      } catch (loadError) {
        setError(loadError instanceof Error ? loadError.message : text.loadError);
      }
    }

    void load();
  }, [text.loadError]);

  return (
    <main className="relative overflow-hidden px-4 py-6 md:px-8">
      <div className="mx-auto max-w-7xl space-y-6">
        <motion.section
          {...fadeUp}
          transition={{ duration: 0.4 }}
          className="panel hero-grid relative overflow-hidden px-6 py-8 md:px-10 md:py-12"
        >
          <div className="absolute inset-0 bg-hero-grid opacity-40" />
          <div className="relative grid gap-6 lg:grid-cols-[1.15fr_0.85fr]">
            <div>
              <p className="mono text-xs uppercase tracking-[0.4em] text-ember">{text.eyebrow}</p>
              <h1 className="mt-4 max-w-3xl text-4xl font-semibold leading-tight md:text-6xl">
                {text.title}
              </h1>
              <p className="mt-4 max-w-2xl text-base text-ink/72 md:text-lg">{text.body}</p>
            </div>

            <div className="grid gap-3 sm:grid-cols-2">
              <MetricCard label={text.version} value={model?.deployment.model_version ?? "Loading"} accent="ember" />
              <MetricCard
                label={text.deviceTarget}
                value={model?.deployment.device_target ?? "Loading"}
                accent="moss"
              />
              <MetricCard label={text.gitCommit} value={model?.deployment.git_commit ?? "Loading"} />
              <MetricCard label={text.apiVersion} value={model?.deployment.api_version ?? "Loading"} />
            </div>
          </div>
        </motion.section>

        {error ? (
          <div className="rounded-[24px] bg-ember/10 px-4 py-3 text-sm text-ember">{error}</div>
        ) : null}

        <ModelExplainer model={model} />

        <motion.section
          {...fadeUp}
          transition={{ duration: 0.45, delay: 0.06 }}
          className="grid gap-6 lg:grid-cols-[0.95fr_1.05fr]"
        >
          <div className="rounded-[28px] border border-white/55 bg-white/80 p-6 shadow-panel">
            <p className="mono text-xs uppercase tracking-[0.28em] text-dusk/60">{text.deployment}</p>
            <div className="mt-5 space-y-4 text-sm text-ink/75">
              <div className="flex items-center justify-between gap-4">
                <span>{text.checkpointSource}</span>
                <span className="mono">{model?.deployment.checkpoint_source ?? "Loading"}</span>
              </div>
              <div className="flex items-center justify-between gap-4">
                <span>{text.checkpointPath}</span>
                <span className="mono text-right text-xs">
                  {model?.checkpoint_path ?? text.noCheckpoint}
                </span>
              </div>
              <div className="flex items-center justify-between gap-4">
                <span>{text.buildTime}</span>
                <span className="mono text-right text-xs">{model?.deployment.build_time ?? "Loading"}</span>
              </div>
              <div className="flex items-center justify-between gap-4">
                <span>{text.paperTitle}</span>
                <span className="mono text-right text-xs">{model?.paper_title ?? "Loading"}</span>
              </div>
            </div>
          </div>

          <div className="rounded-[28px] border border-white/55 bg-white/80 p-6 shadow-panel">
            <p className="mono text-xs uppercase tracking-[0.28em] text-dusk/60">{text.presenterNotes}</p>
            <div className="mt-5 space-y-4 text-sm leading-6 text-ink/72">
              <p>{text.note1}</p>
              <p>{text.note2}</p>
              <p>{text.note3}</p>
            </div>
          </div>
        </motion.section>

        <motion.section
          {...fadeUp}
          transition={{ duration: 0.45, delay: 0.1 }}
          className="rounded-[28px] border border-white/55 bg-white/80 p-6 shadow-panel"
        >
          <div className="flex items-center justify-between gap-4">
            <div>
              <p className="mono text-xs uppercase tracking-[0.28em] text-dusk/60">
                {text.availableCheckpoints}
              </p>
              <p className="mt-3 max-w-3xl text-sm text-ink/70">{text.availableCheckpointsBody}</p>
            </div>
            <span className="mono rounded-full bg-dusk px-4 py-2 text-xs uppercase tracking-[0.24em] text-white">
              {model?.available_checkpoints.length ?? 0} loaded
            </span>
          </div>
          <div className="mt-5 grid gap-4 md:grid-cols-2 xl:grid-cols-3">
            {(model?.available_checkpoints ?? []).map((checkpoint) => (
              <div
                key={checkpoint.name}
                className="rounded-[24px] border border-dusk/10 bg-sand p-4 text-sm text-ink/72"
              >
                <p className="mono text-xs uppercase tracking-[0.28em] text-ink/45">
                  x{checkpoint.scale}
                </p>
                <p className="mt-3 text-lg font-semibold">{checkpoint.name}</p>
                <p className="mt-2 break-all text-xs text-ink/55">
                  {checkpoint.checkpoint_path ?? text.noCheckpoint}
                </p>
                <p className="mt-3 text-xs text-ink/55">{checkpoint.model_version}</p>
              </div>
            ))}
          </div>
        </motion.section>
      </div>
    </main>
  );
}
