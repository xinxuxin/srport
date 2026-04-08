"use client";

import Link from "next/link";
import { AnimatePresence, motion } from "framer-motion";
import { ChangeEvent, DragEvent, useEffect, useMemo, useState } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";

import {
  AnalyticsSummary,
  InferenceResponse,
  ModelInfo,
  fetchAnalyticsSummary,
  fetchModelInfo,
  resolveApiAssetUrl,
  superResolve
} from "../lib/api";
import { AnimatedCounter } from "./animated-counter";
import { MetricCard } from "./metric-card";
import { ModelExplainer } from "./model-explainer";
import { PipelineTimeline } from "./pipeline-timeline";
import { ZoomableCompare } from "./zoomable-compare";

const fadeUp = {
  initial: { opacity: 0, y: 18 },
  animate: { opacity: 1, y: 0 }
};

const MAX_UPLOAD_BYTES = 12 * 1024 * 1024;
const ALLOWED_FILE_TYPES = new Set(["image/png", "image/jpeg", "image/jpg", "image/webp", "image/bmp"]);

function formatCount(value: number): string {
  if (value >= 1_000_000_000) {
    return `${(value / 1_000_000_000).toFixed(1)}B`;
  }
  if (value >= 1_000_000) {
    return `${(value / 1_000_000).toFixed(1)}M`;
  }
  if (value >= 1_000) {
    return `${(value / 1_000).toFixed(1)}K`;
  }
  return `${Math.round(value)}`;
}

function formatResolution(width: number, height: number): string {
  return `${width} x ${height}`;
}

function formatLatency(value: number): string {
  return `${value.toFixed(1)} ms`;
}

function formatShortDate(value: string): string {
  return new Date(value).toLocaleString();
}

export function EpnetDashboard() {
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [analytics, setAnalytics] = useState<AnalyticsSummary | null>(null);
  const [result, setResult] = useState<InferenceResponse | null>(null);
  const [inputPreview, setInputPreview] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    return () => {
      if (inputPreview) {
        URL.revokeObjectURL(inputPreview);
      }
    };
  }, [inputPreview]);

  async function refreshAnalytics() {
    try {
      setAnalytics(await fetchAnalyticsSummary());
    } catch (refreshError) {
      console.error(refreshError);
    }
  }

  useEffect(() => {
    async function load() {
      try {
        const [modelInfo, analyticsInfo] = await Promise.all([
          fetchModelInfo(),
          fetchAnalyticsSummary()
        ]);
        setModel(modelInfo);
        setAnalytics(analyticsInfo);
      } catch (loadError) {
        setError(loadError instanceof Error ? loadError.message : "Unable to connect to the backend.");
      }
    }

    void load();
  }, []);

  async function handleFile(file: File) {
    if (!ALLOWED_FILE_TYPES.has(file.type)) {
      setError("Please upload a PNG, JPEG, WEBP, or BMP image.");
      return;
    }
    if (file.size > MAX_UPLOAD_BYTES) {
      setError("The uploaded file is too large for the demo limit.");
      return;
    }
    setError(null);
    setIsLoading(true);
    setInputPreview((previous) => {
      if (previous) {
        URL.revokeObjectURL(previous);
      }
      return URL.createObjectURL(file);
    });
    try {
      const payload = await superResolve(file);
      setResult(payload);
      setModel(payload.model);
      await refreshAnalytics();
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "Inference failed. Please check the backend logs."
      );
    } finally {
      setIsLoading(false);
    }
  }

  function onFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (file) {
      void handleFile(file);
    }
  }

  function onDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    setIsDragging(false);
    const file = event.dataTransfer.files?.[0];
    if (file) {
      void handleFile(file);
    }
  }

  const outputPreview = useMemo(() => {
    if (!result) {
      return null;
    }
    return resolveApiAssetUrl(result.output_image_url);
  }, [result]);

  return (
    <main className="relative overflow-hidden px-4 py-6 md:px-8">
      <div className="mx-auto max-w-7xl space-y-6">
        <section className="panel hero-grid relative overflow-hidden px-6 py-8 md:px-10 md:py-12">
          <div className="absolute inset-0 bg-hero-grid opacity-40" />
          <div className="relative grid gap-8 lg:grid-cols-[1.3fr_0.9fr]">
            <div>
              <motion.p
                {...fadeUp}
                transition={{ duration: 0.38 }}
                className="mono text-xs uppercase tracking-[0.4em] text-ember"
              >
                EPNet Demo
              </motion.p>
              <motion.h1
                {...fadeUp}
                transition={{ duration: 0.42, delay: 0.05 }}
                className="mt-4 max-w-3xl text-4xl font-semibold leading-tight md:text-6xl"
              >
                Efficient pyramid super-resolution with a system-level story.
              </motion.h1>
              <motion.p
                {...fadeUp}
                transition={{ duration: 0.45, delay: 0.1 }}
                className="mt-4 max-w-2xl text-base text-ink/72 md:text-lg"
              >
                This demo packages EPNet as a presentable AI solution: model telemetry, deployment
                metadata, a live inference timeline, and usage analytics in one polished workflow.
              </motion.p>

              <motion.div
                {...fadeUp}
                transition={{ duration: 0.45, delay: 0.15 }}
                className="mt-6 grid gap-3 sm:grid-cols-3"
              >
                <MetricCard
                  label="Parameters"
                  value={
                    model ? (
                      <AnimatedCounter value={model.parameter_count} format={formatCount} />
                    ) : (
                      "Loading"
                    )
                  }
                  accent="ember"
                />
                <MetricCard
                  label="Estimated MACs"
                  value={
                    model ? (
                      <AnimatedCounter value={model.estimated_macs} format={formatCount} />
                    ) : (
                      "Loading"
                    )
                  }
                  accent="moss"
                />
                <MetricCard
                  label="Reference latency"
                  value={
                    model ? (
                      <AnimatedCounter value={model.reference_latency_ms} format={formatLatency} />
                    ) : (
                      "Loading"
                    )
                  }
                />
              </motion.div>
            </div>

            <motion.div
              {...fadeUp}
              transition={{ duration: 0.45, delay: 0.2 }}
              className="space-y-4"
            >
              <div className="panel border border-white/60 bg-sand/80 p-5">
                <p className="mono text-xs uppercase tracking-[0.28em] text-dusk/70">Deployment panel</p>
                <div className="mt-4 space-y-3 text-sm text-ink/80">
                  <div className="flex items-center justify-between gap-4">
                    <span>Model version</span>
                    <span className="mono">{model?.deployment.model_version ?? "Loading"}</span>
                  </div>
                  <div className="flex items-center justify-between gap-4">
                    <span>Checkpoint source</span>
                    <span className="mono">{model?.deployment.checkpoint_source ?? "Loading"}</span>
                  </div>
                  <div className="flex items-center justify-between gap-4">
                    <span>Device target</span>
                    <span className="mono">{model?.deployment.device_target ?? "Loading"}</span>
                  </div>
                  <div className="flex items-center justify-between gap-4">
                    <span>Git commit</span>
                    <span className="mono">{model?.deployment.git_commit ?? "Loading"}</span>
                  </div>
                  <div className="rounded-2xl bg-white/70 p-4 text-xs text-ink/65">
                    <p className="mono uppercase tracking-[0.22em] text-ink/45">Build time</p>
                    <p className="mt-2 break-all">{model?.deployment.build_time ?? "Loading"}</p>
                  </div>
                </div>
              </div>

              <div className="rounded-[28px] border border-dusk/10 bg-gradient-to-br from-dusk/[0.08] to-ember/[0.08] p-5">
                <p className="mono text-xs uppercase tracking-[0.28em] text-dusk/60">Presenter shortcut</p>
                <p className="mt-3 text-sm text-ink/72">
                  Need architecture context mid-demo? Jump to the model explanation page with the
                  deployment metadata already surfaced.
                </p>
                <Link
                  href="/model"
                  className="mono mt-4 inline-flex rounded-full bg-dusk px-4 py-2 text-xs uppercase tracking-[0.28em] text-white"
                >
                  Open model page
                </Link>
              </div>
            </motion.div>
          </div>
        </section>

        <section className="grid gap-6 lg:grid-cols-[0.96fr_1.04fr]">
          <motion.div
            {...fadeUp}
            transition={{ duration: 0.45, delay: 0.04 }}
            className="panel p-5 md:p-6"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="mono text-xs uppercase tracking-[0.28em] text-ember">Inference</p>
                <h2 className="mt-2 text-2xl font-semibold">Drag, drop, and enhance</h2>
              </div>
              <AnimatePresence>
                {isLoading ? (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.95 }}
                    className="mono rounded-full bg-dusk px-3 py-2 text-xs uppercase tracking-[0.26em] text-white"
                  >
                    Request active
                  </motion.div>
                ) : null}
              </AnimatePresence>
            </div>

            <label
              onDragOver={(event) => {
                event.preventDefault();
                setIsDragging(true);
              }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={onDrop}
              className={`mt-6 flex min-h-80 cursor-pointer flex-col items-center justify-center rounded-[30px] border-2 border-dashed px-6 text-center transition ${
                isDragging
                  ? "scale-[1.01] border-ember bg-ember/12 shadow-[0_24px_70px_rgba(194,84,47,0.18)]"
                  : "border-dusk/20 bg-white/60 hover:border-dusk/45 hover:shadow-[0_20px_50px_rgba(19,32,47,0.08)]"
              }`}
            >
              <input
                type="file"
                accept="image/*"
                className="hidden"
                onChange={onFileChange}
                data-testid="upload-input"
              />
              <motion.div
                animate={{
                  scale: isDragging ? 1.04 : 1,
                  rotate: isDragging ? -1.5 : 0,
                  y: isDragging ? -4 : 0
                }}
                transition={{ type: "spring", stiffness: 180, damping: 16 }}
                className="space-y-4"
              >
                <div className="mx-auto flex h-20 w-20 items-center justify-center rounded-[26px] bg-dusk/10 p-4">
                  <motion.div
                    animate={{ rotate: isDragging ? 10 : 0, scale: isDragging ? 1.06 : 1 }}
                    className="h-full w-full rounded-[18px] bg-gradient-to-br from-dusk via-moss to-ember"
                  />
                </div>
                <div>
                  <p className="text-lg font-medium">Drop a low-resolution image here</p>
                  <p className="mt-2 text-sm text-ink/65">
                    PNG, JPEG, WEBP, or BMP. The pipeline records latency, model metadata, and
                    usage analytics for every request.
                  </p>
                </div>
                <span className="mono inline-flex rounded-full bg-ink px-4 py-2 text-xs uppercase tracking-[0.3em] text-white">
                  Choose image
                </span>
              </motion.div>
            </label>

            <AnimatePresence>
              {error ? (
                <motion.div
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: 12 }}
                  className="mt-4 rounded-2xl bg-ember/10 px-4 py-3 text-sm text-ember"
                  data-testid="upload-error"
                >
                  {error}
                </motion.div>
              ) : null}
            </AnimatePresence>

            <div className="mt-6">
              <PipelineTimeline isLoading={isLoading} stages={result?.pipeline.stages ?? null} />
            </div>
          </motion.div>

          <motion.div
            {...fadeUp}
            transition={{ duration: 0.45, delay: 0.08 }}
            className="panel p-5 md:p-6"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="mono text-xs uppercase tracking-[0.28em] text-moss">Result</p>
                <h2 className="mt-2 text-2xl font-semibold">Before, after, and inspected</h2>
              </div>
              {result ? (
                <div className="mono text-xs text-ink/55">Request {result.request_id.slice(0, 8)}</div>
              ) : null}
            </div>

            <div className="mt-6 grid gap-4 md:grid-cols-2">
              <div className="rounded-[24px] bg-sand p-3">
                <p className="mono px-2 text-xs uppercase tracking-[0.28em] text-ink/55">Input</p>
                <div className="mt-3 aspect-square overflow-hidden rounded-[20px] bg-white">
                  {inputPreview ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img src={inputPreview} alt="Input preview" className="h-full w-full object-cover" />
                  ) : (
                    <div className="flex h-full items-center justify-center text-sm text-ink/45">
                      Awaiting upload
                    </div>
                  )}
                </div>
              </div>
              <div className="rounded-[24px] bg-sand p-3">
                <div className="flex items-center justify-between px-2">
                  <p className="mono text-xs uppercase tracking-[0.28em] text-ink/55">Output</p>
                  <p className="text-xs text-ink/45">Hover to inspect details</p>
                </div>
                <div className="mt-3 aspect-square overflow-hidden rounded-[20px] bg-white">
                  <ZoomableCompare inputPreview={inputPreview} outputPreview={outputPreview} />
                </div>
              </div>
            </div>

            <div className="mt-4 grid gap-3 sm:grid-cols-4">
              <MetricCard
                label="Input"
                value={
                  result
                    ? formatResolution(result.input.resolution.width, result.input.resolution.height)
                    : "Waiting"
                }
              />
              <MetricCard
                label="Output"
                value={
                  result
                    ? formatResolution(result.output.resolution.width, result.output.resolution.height)
                    : "Waiting"
                }
                accent="ember"
              />
              <MetricCard
                label="Latency"
                value={
                  result ? (
                    <AnimatedCounter value={result.runtime.latency_ms} format={formatLatency} />
                  ) : (
                    "Waiting"
                  )
                }
                accent="moss"
              />
              <MetricCard
                label="FLOPs"
                value={
                  result ? (
                    <AnimatedCounter value={result.runtime.estimated_flops} format={formatCount} />
                  ) : (
                    "Waiting"
                  )
                }
              />
            </div>
          </motion.div>
        </section>

        <motion.section
          {...fadeUp}
          transition={{ duration: 0.45, delay: 0.12 }}
        >
          <ModelExplainer model={model} compact />
        </motion.section>

        <motion.section
          {...fadeUp}
          transition={{ duration: 0.5, delay: 0.16 }}
          className="grid gap-6 lg:grid-cols-[1.05fr_0.95fr]"
        >
          <div className="panel p-5 md:p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="mono text-xs uppercase tracking-[0.28em] text-dusk/60">Analytics</p>
                <h2 className="mt-2 text-2xl font-semibold">Usage and runtime history</h2>
              </div>
              <button
                type="button"
                onClick={() => void refreshAnalytics()}
                className="mono rounded-full bg-ink px-4 py-2 text-xs uppercase tracking-[0.26em] text-white"
              >
                Refresh
              </button>
            </div>

            <div className="mt-6 grid gap-4 sm:grid-cols-3">
              <MetricCard
                label="Requests"
                value={
                  analytics ? (
                    <AnimatedCounter value={analytics.total_requests} format={formatCount} />
                  ) : (
                    "0"
                  )
                }
                accent="ember"
                detail="Stored usage records"
                testId="requests-metric"
              />
              <MetricCard
                label="Avg latency"
                value={
                  analytics ? (
                    <AnimatedCounter value={analytics.average_latency_ms} format={formatLatency} />
                  ) : (
                    "0 ms"
                  )
                }
                accent="moss"
              />
              <MetricCard
                label="Avg megapixels"
                value={
                  analytics ? (
                    <AnimatedCounter
                      value={analytics.average_output_megapixels}
                      format={(value) => value.toFixed(2)}
                    />
                  ) : (
                    "0.00"
                  )
                }
              />
            </div>

            <div className="mt-6 grid gap-4 lg:grid-cols-2">
              <div className="rounded-[24px] bg-sand p-4">
                <p className="mono text-xs uppercase tracking-[0.28em] text-ink/55">Latency series</p>
                <div className="mt-4 h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={analytics?.latency_series ?? []}>
                      <defs>
                        <linearGradient id="latencyFill" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#22415d" stopOpacity={0.55} />
                          <stop offset="95%" stopColor="#22415d" stopOpacity={0.08} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid stroke="rgba(19,32,47,0.08)" vertical={false} />
                      <XAxis dataKey="label" tick={{ fill: "#506170", fontSize: 11 }} />
                      <YAxis tick={{ fill: "#506170", fontSize: 11 }} />
                      <Tooltip />
                      <Area
                        type="monotone"
                        dataKey="value"
                        stroke="#22415d"
                        fill="url(#latencyFill)"
                        strokeWidth={2}
                        isAnimationActive
                        animationDuration={450}
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>
              <div className="rounded-[24px] bg-sand p-4">
                <p className="mono text-xs uppercase tracking-[0.28em] text-ink/55">
                  Upscale distribution
                </p>
                <div className="mt-4 h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={analytics?.upscale_distribution ?? []}>
                      <CartesianGrid stroke="rgba(19,32,47,0.08)" vertical={false} />
                      <XAxis dataKey="label" tick={{ fill: "#506170", fontSize: 11 }} />
                      <YAxis tick={{ fill: "#506170", fontSize: 11 }} />
                      <Tooltip />
                      <Bar
                        dataKey="value"
                        fill="#c2542f"
                        radius={[12, 12, 0, 0]}
                        isAnimationActive
                        animationDuration={450}
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          </div>

          <div className="panel p-5 md:p-6">
            <p className="mono text-xs uppercase tracking-[0.28em] text-dusk/60">Recent usage</p>
            <h2 className="mt-2 text-2xl font-semibold">Inference activity</h2>
            <div className="mt-6 space-y-3">
              {(analytics?.recent_events ?? []).length > 0 ? (
                analytics?.recent_events.map((event) => (
                  <motion.div
                    key={event.request_id}
                    layout
                    className="rounded-[24px] border border-ink/8 bg-white/70 p-4"
                  >
                    <div className="flex items-center justify-between gap-4">
                      <div>
                        <p className="mono text-xs uppercase tracking-[0.28em] text-ink/45">
                          {formatShortDate(event.created_at)}
                        </p>
                        <p className="mt-2 text-sm">
                          {formatResolution(
                            event.input_resolution.width,
                            event.input_resolution.height
                          )}{" "}
                          to{" "}
                          {formatResolution(
                            event.output_resolution.width,
                            event.output_resolution.height
                          )}
                        </p>
                      </div>
                      <div className="text-right">
                        <p className="text-sm font-medium">{formatLatency(event.latency_ms)}</p>
                        <p className="mono mt-1 text-xs text-ink/45">x{event.upscale}</p>
                      </div>
                    </div>
                  </motion.div>
                ))
              ) : (
                <div className="rounded-[24px] bg-sand p-5 text-sm text-ink/60">
                  No usage yet. Run your first image through the model to populate the dashboard.
                </div>
              )}
            </div>
          </div>
        </motion.section>
      </div>
    </main>
  );
}
