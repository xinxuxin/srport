"use client";

import { motion } from "framer-motion";
import { useEffect, useMemo, useState } from "react";

type Stage = {
  key: string;
  label: string;
  description: string;
  duration_ms: number;
};

type PipelineTimelineProps = {
  isLoading: boolean;
  stages: Stage[] | null;
};

const fallbackStages: Stage[] = [
  {
    key: "upload",
    label: "Upload",
    description: "Ingress into the API boundary.",
    duration_ms: 0
  },
  {
    key: "validate",
    label: "Validate",
    description: "Size, type, and pixel guardrails.",
    duration_ms: 0
  },
  {
    key: "decode",
    label: "Decode",
    description: "Load and normalize source pixels.",
    duration_ms: 0
  },
  {
    key: "preprocess",
    label: "Preprocess",
    description: "Transform to EPNet tensor input.",
    duration_ms: 0
  },
  {
    key: "infer",
    label: "Infer",
    description: "Run PFEM, ESPM, and reconstruction.",
    duration_ms: 0
  },
  {
    key: "encode",
    label: "Encode",
    description: "Serialize the SR output artifact.",
    duration_ms: 0
  },
  {
    key: "log",
    label: "Log",
    description: "Persist analytics and telemetry.",
    duration_ms: 0
  }
];

export function PipelineTimeline({ isLoading, stages }: PipelineTimelineProps) {
  const activeStages = useMemo(() => stages ?? fallbackStages, [stages]);
  const [activeIndex, setActiveIndex] = useState(-1);

  useEffect(() => {
    if (!isLoading) {
      setActiveIndex(stages ? stages.length - 1 : -1);
      return;
    }

    setActiveIndex(0);
    const interval = window.setInterval(() => {
      setActiveIndex((value) => (value + 1) % fallbackStages.length);
    }, 520);
    return () => window.clearInterval(interval);
  }, [isLoading, stages]);

  const maxDuration = Math.max(...activeStages.map((stage) => stage.duration_ms), 1);

  return (
    <div className="rounded-[28px] border border-white/55 bg-white/75 p-5 shadow-panel">
      <div className="flex items-center justify-between">
        <div>
          <p className="mono text-xs uppercase tracking-[0.28em] text-dusk/60">Pipeline</p>
          <h3 className="mt-2 text-2xl font-semibold">System timeline</h3>
        </div>
        <div className="mono rounded-full bg-dusk px-3 py-2 text-[11px] uppercase tracking-[0.28em] text-white">
          {isLoading ? "Live request" : "Last request"}
        </div>
      </div>

      <div className="mt-6 space-y-3">
        {activeStages.map((stage, index) => {
          const isActive = isLoading ? index === activeIndex : index <= activeIndex;
          const widthPercent =
            stage.duration_ms > 0 ? Math.max((stage.duration_ms / maxDuration) * 100, 8) : 8;

          return (
            <motion.div
              key={stage.key}
              layout
              initial={{ opacity: 0, x: -16 }}
              animate={{ opacity: 1, x: 0 }}
              className={`rounded-[22px] border p-4 transition ${
                isActive
                  ? "border-dusk/20 bg-gradient-to-r from-dusk/[0.06] to-ember/[0.06]"
                  : "border-ink/8 bg-sand/70"
              }`}
            >
              <div className="flex items-start justify-between gap-4">
                <div>
                  <div className="flex items-center gap-3">
                    <div
                      className={`flex h-9 w-9 items-center justify-center rounded-full text-sm font-semibold ${
                        isActive ? "bg-dusk text-white" : "bg-white text-ink/45"
                      }`}
                    >
                      {index + 1}
                    </div>
                    <div>
                      <p className="text-sm font-semibold">{stage.label}</p>
                      <p className="text-xs text-ink/55">{stage.description}</p>
                    </div>
                  </div>
                </div>
                <div className="text-right">
                  <p className="mono text-xs uppercase tracking-[0.24em] text-ink/45">Duration</p>
                  <p className="mt-2 text-sm font-medium">
                    {stage.duration_ms > 0 ? `${stage.duration_ms.toFixed(1)} ms` : "pending"}
                  </p>
                </div>
              </div>

              <div className="mt-4 h-2 overflow-hidden rounded-full bg-white/80">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${isActive ? widthPercent : 6}%` }}
                  transition={{ duration: 0.45, ease: "easeOut" }}
                  className="h-full rounded-full bg-gradient-to-r from-ember to-dusk"
                />
              </div>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}
