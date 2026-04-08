"use client";

import { motion } from "framer-motion";
import { useEffect, useState } from "react";

import { fetchModelInfo, ModelInfo } from "../lib/api";
import { MetricCard } from "./metric-card";
import { ModelExplainer } from "./model-explainer";

const fadeUp = {
  initial: { opacity: 0, y: 18 },
  animate: { opacity: 1, y: 0 }
};

export function ModelPage() {
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      try {
        setModel(await fetchModelInfo());
      } catch (loadError) {
        setError(loadError instanceof Error ? loadError.message : "Unable to load model metadata.");
      }
    }

    void load();
  }, []);

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
              <p className="mono text-xs uppercase tracking-[0.4em] text-ember">Model Page</p>
              <h1 className="mt-4 max-w-3xl text-4xl font-semibold leading-tight md:text-6xl">
                Architecture, deployment metadata, and explainability in one place.
              </h1>
              <p className="mt-4 max-w-2xl text-base text-ink/72 md:text-lg">
                This page turns EPNet from a paper implementation into a presentable solution asset:
                architecture summary, deployment metadata, and a customer-friendly talk track.
              </p>
            </div>

            <div className="grid gap-3 sm:grid-cols-2">
              <MetricCard
                label="Version"
                value={model?.deployment.model_version ?? "Loading"}
                accent="ember"
              />
              <MetricCard
                label="Device target"
                value={model?.deployment.device_target ?? "Loading"}
                accent="moss"
              />
              <MetricCard
                label="Git commit"
                value={model?.deployment.git_commit ?? "Loading"}
              />
              <MetricCard
                label="API version"
                value={model?.deployment.api_version ?? "Loading"}
              />
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
            <p className="mono text-xs uppercase tracking-[0.28em] text-dusk/60">Deployment metadata</p>
            <div className="mt-5 space-y-4 text-sm text-ink/75">
              <div className="flex items-center justify-between gap-4">
                <span>Checkpoint source</span>
                <span className="mono">{model?.deployment.checkpoint_source ?? "Loading"}</span>
              </div>
              <div className="flex items-center justify-between gap-4">
                <span>Checkpoint path</span>
                <span className="mono text-right text-xs">
                  {model?.checkpoint_path ?? "No checkpoint configured"}
                </span>
              </div>
              <div className="flex items-center justify-between gap-4">
                <span>Build time</span>
                <span className="mono text-right text-xs">{model?.deployment.build_time ?? "Loading"}</span>
              </div>
              <div className="flex items-center justify-between gap-4">
                <span>Paper title</span>
                <span className="mono text-right text-xs">{model?.paper_title ?? "Loading"}</span>
              </div>
            </div>
          </div>

          <div className="rounded-[28px] border border-white/55 bg-white/80 p-6 shadow-panel">
            <p className="mono text-xs uppercase tracking-[0.28em] text-dusk/60">Presenter notes</p>
            <div className="mt-5 space-y-4 text-sm leading-6 text-ink/72">
              <p>
                Lead with the business framing: EPNet is positioned as an efficient SR backbone,
                not a brute-force quality-only model.
              </p>
              <p>
                Then explain the three-stage flow: detail extraction with PFEM, multiscale context
                with ESPM, and low-cost reconstruction with PixelShuffle.
              </p>
              <p>
                Finally connect to deployment: versioning, checkpoint provenance, build metadata,
                and runtime target are surfaced as first-class product attributes.
              </p>
            </div>
          </div>
        </motion.section>
      </div>
    </main>
  );
}
