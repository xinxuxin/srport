"use client";

import Link from "next/link";
import { motion } from "framer-motion";

import type { ModelInfo } from "../lib/api";
import { MetricCard } from "./metric-card";

type ModelExplainerProps = {
  model: ModelInfo | null;
  compact?: boolean;
};

const explainerCards = [
  {
    title: "PFEM",
    subtitle: "Pyramid Feature Extraction Module",
    body: "Stacks LFEB, a modified Swin-style attention block, and ESAB to recover local textures plus longer-range context."
  },
  {
    title: "ESPM",
    subtitle: "Efficient Spatial Pyramid Module",
    body: "Builds a lightweight multiscale branch that captures broader structure without inflating compute cost."
  },
  {
    title: "Reconstruction",
    subtitle: "Fusion + PixelShuffle",
    body: "Fuses PFEM and ESPM outputs, projects features with a 3x3 convolution, then upsamples with PixelShuffle."
  }
];

export function ModelExplainer({ model, compact = false }: ModelExplainerProps) {
  const pfemBlocks = String(model?.architecture["num_pfem"] ?? 4);
  const windowSize = String(model?.architecture["window_size"] ?? 8);
  const embedDim = String(model?.architecture["embed_dim"] ?? 40);

  return (
    <section className="space-y-6">
      <div className="flex items-end justify-between gap-4">
        <div>
          <p className="mono text-xs uppercase tracking-[0.28em] text-dusk/60">Model explainer</p>
          <h2 className="mt-2 text-3xl font-semibold">EPNet in product language</h2>
          <p className="mt-3 max-w-3xl text-sm text-ink/65">
            The network splits into a feature-rich detail branch, a lightweight pyramid context
            branch, and a reconstruction head. This framing is easier to explain to customers than
            raw paper terminology alone.
          </p>
        </div>
        {compact ? (
          <Link
            href="/model"
            className="mono rounded-full bg-dusk px-4 py-2 text-xs uppercase tracking-[0.28em] text-white"
          >
            Open model page
          </Link>
        ) : null}
      </div>

      <div className="grid gap-4 lg:grid-cols-[1.1fr_0.9fr]">
        <div className="rounded-[28px] border border-white/55 bg-white/80 p-5 shadow-panel">
          <div className="flex flex-wrap items-center gap-3">
            <span className="mono rounded-full bg-ember/12 px-3 py-2 text-xs uppercase tracking-[0.28em] text-ember">
              Input image
            </span>
            <div className="h-px flex-1 bg-ink/10" />
            <span className="mono rounded-full bg-dusk/12 px-3 py-2 text-xs uppercase tracking-[0.28em] text-dusk">
              Super-resolved output
            </span>
          </div>

          <div className="mt-8 grid gap-4 md:grid-cols-3">
            {explainerCards.map((card, index) => (
              <motion.div
                key={card.title}
                initial={{ opacity: 0, y: 18 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, amount: 0.4 }}
                transition={{ duration: 0.4, delay: index * 0.08 }}
                className="relative rounded-[24px] border border-ink/8 bg-sand/75 p-5"
              >
                <div className="absolute -top-3 left-5 rounded-full bg-dusk px-3 py-1 text-[11px] uppercase tracking-[0.28em] text-white">
                  Stage {index + 1}
                </div>
                <div className="pt-4">
                  <h3 className="text-lg font-semibold">{card.title}</h3>
                  <p className="mt-1 text-xs uppercase tracking-[0.24em] text-ink/45">
                    {card.subtitle}
                  </p>
                  <p className="mt-4 text-sm leading-6 text-ink/70">{card.body}</p>
                </div>
              </motion.div>
            ))}
          </div>

          <div className="mt-8 overflow-hidden rounded-[24px] border border-ink/8 bg-white/80 p-5">
            <p className="mono text-xs uppercase tracking-[0.28em] text-ink/55">Dataflow</p>
            <div className="mt-5 grid gap-4 lg:grid-cols-[0.9fr_1.2fr_0.9fr]">
              <div className="rounded-[22px] bg-sand p-4">
                <p className="text-sm font-semibold">Shallow 3x3 conv</p>
                <p className="mt-2 text-sm text-ink/65">
                  Maps RGB pixels into the EPNet feature space.
                </p>
              </div>

              <div className="relative rounded-[22px] border border-dusk/10 bg-gradient-to-br from-dusk/[0.06] to-ember/[0.06] p-4">
                <div className="grid gap-3 md:grid-cols-2">
                  <div className="rounded-[18px] bg-white/80 p-4">
                    <p className="text-sm font-semibold">PFEM branch</p>
                    <p className="mt-2 text-xs text-ink/60">
                      Local detail + transformer-guided context.
                    </p>
                  </div>
                  <div className="rounded-[18px] bg-white/80 p-4">
                    <p className="text-sm font-semibold">ESPM branch</p>
                    <p className="mt-2 text-xs text-ink/60">
                      Efficient pyramid context for multiscale structure.
                    </p>
                  </div>
                </div>

                <motion.div
                  animate={{ x: ["0%", "88%"] }}
                  transition={{ duration: 2.1, repeat: Number.POSITIVE_INFINITY, ease: "easeInOut" }}
                  className="absolute bottom-4 left-4 h-2 w-16 rounded-full bg-gradient-to-r from-ember to-dusk"
                />
              </div>

              <div className="rounded-[22px] bg-sand p-4">
                <p className="text-sm font-semibold">Fusion + PixelShuffle</p>
                <p className="mt-2 text-sm text-ink/65">
                  Reconstructs the final higher-resolution image with low overhead.
                </p>
              </div>
            </div>
          </div>
        </div>

        <div className="space-y-4">
          <MetricCard
            label="PFEM blocks"
            value={pfemBlocks}
            accent="ember"
            detail="Default paper-grounded configuration"
          />
          <MetricCard
            label="Window size"
            value={windowSize}
            accent="moss"
            detail="Modified Swin-style attention window"
          />
          <MetricCard
            label="Embed dim"
            value={embedDim}
            detail="Lightweight feature width for demo efficiency"
          />
          <div className="rounded-[28px] border border-white/55 bg-white/80 p-5 shadow-panel">
            <p className="mono text-xs uppercase tracking-[0.28em] text-dusk/60">Talk track</p>
            <ul className="mt-4 space-y-3 text-sm text-ink/70">
              <li>EPNet preserves detail with PFEM while keeping compute bounded with ESPM.</li>
              <li>
                The architecture is explainable as a two-branch feature extractor plus a lean
                reconstruction head.
              </li>
              <li>
                This makes it easy to discuss quality, efficiency, and edge deployment tradeoffs in
                one narrative.
              </li>
            </ul>
          </div>
        </div>
      </div>
    </section>
  );
}
