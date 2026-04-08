"use client";

import { AnimatePresence, motion } from "framer-motion";
import { ChangeEvent, DragEvent, useEffect, useState } from "react";
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
  superResolve
} from "../lib/api";
import { MetricCard } from "./metric-card";

const fadeUp = {
  initial: { opacity: 0, y: 18 },
  animate: { opacity: 1, y: 0 }
};

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
  return `${value}`;
}

function formatResolution(width: number, height: number): string {
  return `${width} x ${height}`;
}

function formatLatency(value: number): string {
  return `${value.toFixed(1)} ms`;
}

export function EpnetDashboard() {
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [analytics, setAnalytics] = useState<AnalyticsSummary | null>(null);
  const [result, setResult] = useState<InferenceResponse | null>(null);
  const [inputPreview, setInputPreview] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

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
    setError(null);
    setIsLoading(true);
    setInputPreview(URL.createObjectURL(file));
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

  const outputPreview = result
    ? `data:image/${result.output.format.toLowerCase()};base64,${result.output_image_base64}`
    : null;

  return (
    <main className="relative overflow-hidden px-4 py-6 md:px-8">
      <div className="mx-auto max-w-7xl space-y-6">
        <motion.section
          {...fadeUp}
          transition={{ duration: 0.45 }}
          className="panel hero-grid relative overflow-hidden px-6 py-8 md:px-10 md:py-12"
        >
          <div className="absolute inset-0 bg-hero-grid opacity-40" />
          <div className="relative grid gap-8 lg:grid-cols-[1.3fr_0.9fr]">
            <div>
              <p className="mono text-xs uppercase tracking-[0.4em] text-ember">EPNet Demo</p>
              <h1 className="mt-4 max-w-3xl text-4xl font-semibold leading-tight md:text-6xl">
                Efficient pyramid super-resolution with live telemetry.
              </h1>
              <p className="mt-4 max-w-2xl text-base text-ink/72 md:text-lg">
                Upload a low-resolution image and run it through the EPNet reconstruction pipeline.
                The demo reports parameter count, estimated MACs and FLOPs, latency, resolution
                changes, and usage analytics in one place.
              </p>

              <div className="mt-6 grid gap-3 sm:grid-cols-3">
                <MetricCard
                  label="Parameters"
                  value={model ? formatCount(model.parameter_count) : "Loading"}
                  accent="ember"
                />
                <MetricCard
                  label="Estimated MACs"
                  value={model ? formatCount(model.estimated_macs) : "Loading"}
                  accent="moss"
                />
                <MetricCard
                  label="Reference Latency"
                  value={model ? formatLatency(model.reference_latency_ms) : "Loading"}
                />
              </div>
            </div>

            <div className="panel border border-white/60 bg-sand/80 p-5">
              <p className="mono text-xs uppercase tracking-[0.28em] text-dusk/70">Model Snapshot</p>
              <div className="mt-4 space-y-3 text-sm text-ink/80">
                <div className="flex items-center justify-between gap-4">
                  <span>Status</span>
                  <span
                    className={`rounded-full px-3 py-1 text-xs font-medium ${
                      model?.checkpoint_loaded
                        ? "bg-moss/15 text-moss"
                        : "bg-ember/15 text-ember"
                    }`}
                  >
                    {model?.checkpoint_loaded ? "Checkpoint loaded" : "No checkpoint yet"}
                  </span>
                </div>
                <div className="flex items-center justify-between gap-4">
                  <span>Upscale factor</span>
                  <span className="mono">x{model?.upscale ?? 4}</span>
                </div>
                <div className="flex items-center justify-between gap-4">
                  <span>Weights source</span>
                  <span className="mono">{model?.weights_source ?? "pending"}</span>
                </div>
                <div className="rounded-2xl bg-white/70 p-4 text-xs text-ink/65">
                  {model?.paper_title ?? "Loading paper metadata..."}
                </div>
              </div>
            </div>
          </div>
        </motion.section>

        <section className="grid gap-6 lg:grid-cols-[0.95fr_1.05fr]">
          <motion.div
            {...fadeUp}
            transition={{ duration: 0.5, delay: 0.05 }}
            className="panel p-5 md:p-6"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="mono text-xs uppercase tracking-[0.28em] text-ember">Inference</p>
                <h2 className="mt-2 text-2xl font-semibold">Drag, drop, enhance</h2>
              </div>
              {isLoading ? (
                <div className="mono rounded-full bg-dusk px-3 py-1 text-xs text-white">
                  Processing
                </div>
              ) : null}
            </div>

            <label
              onDragOver={(event) => {
                event.preventDefault();
                setIsDragging(true);
              }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={onDrop}
              className={`mt-6 flex min-h-72 cursor-pointer flex-col items-center justify-center rounded-[28px] border-2 border-dashed px-6 text-center transition ${
                isDragging
                  ? "border-ember bg-ember/10"
                  : "border-dusk/20 bg-white/60 hover:border-dusk/50"
              }`}
            >
              <input type="file" accept="image/*" className="hidden" onChange={onFileChange} />
              <motion.div
                animate={{ scale: isDragging ? 1.02 : 1, rotate: isDragging ? -1 : 0 }}
                transition={{ type: "spring", stiffness: 180, damping: 16 }}
                className="space-y-4"
              >
                <div className="mx-auto h-16 w-16 rounded-2xl bg-dusk/10 p-4">
                  <div className="h-full rounded-xl bg-gradient-to-br from-dusk to-ember" />
                </div>
                <div>
                  <p className="text-lg font-medium">Drop a low-resolution image here</p>
                  <p className="mt-2 text-sm text-ink/65">
                    PNG, JPEG, WEBP, or BMP. The backend returns the enhanced output plus runtime
                    telemetry.
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
                >
                  {error}
                </motion.div>
              ) : null}
            </AnimatePresence>
          </motion.div>

          <motion.div
            {...fadeUp}
            transition={{ duration: 0.5, delay: 0.1 }}
            className="panel p-5 md:p-6"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="mono text-xs uppercase tracking-[0.28em] text-moss">Result</p>
                <h2 className="mt-2 text-2xl font-semibold">Before and after</h2>
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
                <p className="mono px-2 text-xs uppercase tracking-[0.28em] text-ink/55">Output</p>
                <div className="mt-3 aspect-square overflow-hidden rounded-[20px] bg-white">
                  {outputPreview ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img
                      src={outputPreview}
                      alt="Super-resolved output"
                      className="h-full w-full object-cover"
                    />
                  ) : (
                    <div className="flex h-full items-center justify-center text-sm text-ink/45">
                      EPNet output will appear here
                    </div>
                  )}
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
                value={result ? formatLatency(result.runtime.latency_ms) : "Waiting"}
                accent="moss"
              />
              <MetricCard
                label="FLOPs"
                value={result ? formatCount(result.runtime.estimated_flops) : "Waiting"}
              />
            </div>
          </motion.div>
        </section>

        <motion.section
          {...fadeUp}
          transition={{ duration: 0.5, delay: 0.15 }}
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
                value={analytics ? `${analytics.total_requests}` : "0"}
                accent="ember"
              />
              <MetricCard
                label="Avg latency"
                value={analytics ? formatLatency(analytics.average_latency_ms) : "0 ms"}
                accent="moss"
              />
              <MetricCard
                label="Avg megapixels"
                value={analytics ? analytics.average_output_megapixels.toFixed(2) : "0.00"}
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
                      <Bar dataKey="value" fill="#c2542f" radius={[12, 12, 0, 0]} />
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
                  <div
                    key={event.request_id}
                    className="rounded-[24px] border border-ink/8 bg-white/70 p-4"
                  >
                    <div className="flex items-center justify-between gap-4">
                      <div>
                        <p className="mono text-xs uppercase tracking-[0.28em] text-ink/45">
                          {new Date(event.created_at).toLocaleString()}
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
                  </div>
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
