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
  BatchInferenceResponse,
  CheckpointOption,
  EventPreview,
  HistoryEvent,
  InferenceResponse,
  ModelInfo,
  PipelineStage,
  Resolution,
  batchSuperResolve,
  fetchAnalyticsSummary,
  fetchHistoryEvent,
  fetchModelInfo,
  resolveApiAssetUrl,
  superResolve
} from "../lib/api";
import { useLanguage } from "./language-provider";
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
const SESSION_STORAGE_KEY = "epnet-demo-session";

type Mode = "single" | "batch";

type QueueItem = {
  id: string;
  name: string;
  size: number;
  status: "queued" | "running" | "done" | "error";
  requestId?: string;
};

type DisplayVariant = {
  method: string;
  label: string;
  imageUrl: string;
  format: string;
  bytes: number;
  isPrimary: boolean;
};

type DisplayResult = {
  requestId: string;
  createdAt: string;
  sessionId: string;
  method: string;
  checkpointName: string;
  scale: number;
  outputFormat: string;
  tileSize: number;
  inputResolution: Resolution;
  outputResolution: Resolution;
  latencyMs: number;
  cpuTimeMs: number;
  parameterCount: number;
  estimatedMacs: number;
  estimatedFlops: number;
  inputImageUrl: string;
  inputBytes: number;
  variants: DisplayVariant[];
  pipelineStages: PipelineStage[];
};

const copy = {
  en: {
    demoEyebrow: "EPNet Demo",
    heroTitle: "Efficient pyramid super-resolution for deployment-ready demos.",
    heroBody:
      "This demo packages EPNet as a presentable AI solution: model telemetry, deployment metadata, a live inference timeline, bilingual UX, and usage analytics in one polished workflow.",
    parameters: "Parameters",
    estimatedMacs: "Estimated Multi-Adds",
    estimatedMemory: "Est. memory",
    referenceLatency: "Reference latency",
    deploymentPanel: "Deployment panel",
    modelVersion: "Model version",
    checkpointSource: "Checkpoint source",
    deviceTarget: "Device target",
    gitCommit: "Git commit",
    buildTime: "Build time",
    presenterShortcut: "Presenter shortcut",
    presenterShortcutBody:
      "Need architecture context mid-demo? Jump to the model explanation page with deployment metadata and checkpoint inventory already surfaced.",
    openModelPage: "Open model page",
    controlsEyebrow: "Solution controls",
    controlsTitle: "Drive the demo like a product",
    controlsBody:
      "Switch languages, methods, scales, checkpoints, output format, tile mode, and session identity without leaving the main flow.",
    singleMode: "Single image",
    batchMode: "Batch queue",
    sessionId: "Demo session ID",
    resetSession: "Reset",
    method: "Method",
    scale: "Scale",
    checkpoint: "Checkpoint",
    autoCheckpoint: "Auto select",
    outputFormat: "Output format",
    tileMode: "Tile inference",
    tileOff: "Off",
    tileHint: "Memory-friendly for larger images.",
    uploadEyebrow: "Inference",
    uploadTitle: "Drag, drop, and enhance",
    requestActive: "Request active",
    uploadBody:
      "PNG, JPEG, WEBP, or BMP. Every request records latency, model metadata, shareable artifacts, and analytics.",
    chooseImage: "Choose image",
    chooseImages: "Choose images",
    queueTitle: "Batch queue",
    queueEmpty: "No files queued yet. Switch to batch mode and upload multiple images to demo aggregated stats.",
    aggregateTitle: "Batch aggregate",
    totalFiles: "Total files",
    completedFiles: "Completed",
    averageLatency: "Avg latency",
    totalMegapixels: "Output MP",
    resultEyebrow: "Result",
    resultTitle: "Before, after, and compared",
    requestLabel: "Request",
    historyBadge: "History replay",
    download: "Download result",
    share: "Copy share link",
    copied: "Share link copied.",
    shareFailed: "Unable to copy the share link.",
    commandCopied: "Reproduction command copied.",
    input: "Input",
    output: "Output",
    latency: "Latency",
    cpuTime: "CPU time",
    multiadds: "Multi-Adds",
    memoryUsage: "Est. memory",
    flops: "FLOPs",
    noResult: "EPNet output will appear here",
    compareGallery: "A/B compare gallery",
    compareGalleryBody:
      "Click a method card to compare the same request across EPNet, bicubic, and baseline upsampling.",
    oneClickRepro: "One-click reproducibility",
    oneClickReproBody:
      "These commands mirror the current demo settings so you can pivot from UI to CLI during the presentation.",
    copyCommand: "Copy command",
    trainCommand: "Train command",
    evalCommand: "Eval command",
    inferCommand: "Infer command",
    analyticsEyebrow: "Analytics",
    analyticsTitle: "Usage and runtime history",
    requests: "Requests",
    sessions: "Sessions",
    avgLatency: "Avg latency",
    p50Latency: "P50 latency",
    p95Latency: "P95 latency",
    avgMegapixels: "Avg megapixels",
    latencySeries: "Latency series",
    upscaleDistribution: "Upscale distribution",
    recentUsage: "Recent usage",
    inferenceActivity: "Inference activity",
    replay: "Replay",
    recentShare: "Share",
    recentEmpty: "No usage yet. Run your first image through the model to populate the dashboard.",
    scaleMismatch:
      "This EPNet scale is not available in the loaded checkpoints. Choose another scale or switch to Bicubic/Baseline.",
    historyLoaded: "History replay loaded.",
    latestRequest: "Latest request"
  },
  zh: {
    demoEyebrow: "EPNet 演示",
    heroTitle: "面向部署演示的高效金字塔超分辨率方案。",
    heroBody:
      "这个 Demo 把 EPNet 打包成更适合展示的 AI 解决方案：模型指标、部署版本信息、动画化流水线、中英文界面，以及使用分析都放在同一条产品链路里。",
    parameters: "参数量",
    estimatedMacs: "预估 Multi-Adds",
    estimatedMemory: "预估内存",
    referenceLatency: "参考延迟",
    deploymentPanel: "部署版本面板",
    modelVersion: "模型版本",
    checkpointSource: "Checkpoint 来源",
    deviceTarget: "目标设备",
    gitCommit: "Git 提交",
    buildTime: "构建时间",
    presenterShortcut: "演示快捷入口",
    presenterShortcutBody:
      "如果讲到一半需要切到模型结构页，这里可以快速跳转，并且保留部署元数据和 checkpoint 信息。",
    openModelPage: "打开模型解释页",
    controlsEyebrow: "方案控制台",
    controlsTitle: "像产品一样驱动整个 Demo",
    controlsBody:
      "你可以直接在主流程里切换语言、推理方法、倍率、checkpoint、输出格式、tile 模式和 session 身份。",
    singleMode: "单图模式",
    batchMode: "批量队列",
    sessionId: "演示 Session ID",
    resetSession: "重置",
    method: "方法",
    scale: "倍率",
    checkpoint: "Checkpoint",
    autoCheckpoint: "自动选择",
    outputFormat: "输出格式",
    tileMode: "Tile 推理",
    tileOff: "关闭",
    tileHint: "更适合大图的内存友好模式。",
    uploadEyebrow: "推理",
    uploadTitle: "拖拽上传并完成增强",
    requestActive: "请求进行中",
    uploadBody:
      "支持 PNG、JPEG、WEBP、BMP。每次请求都会记录延迟、模型信息、可分享产物和使用分析。",
    chooseImage: "选择图片",
    chooseImages: "选择多张图片",
    queueTitle: "批量队列",
    queueEmpty: "当前还没有批量文件。切到批量模式后上传多张图片，就能展示聚合统计。",
    aggregateTitle: "批量聚合结果",
    totalFiles: "总文件数",
    completedFiles: "完成数",
    averageLatency: "平均延迟",
    totalMegapixels: "总输出 MP",
    resultEyebrow: "结果",
    resultTitle: "前后对比与多方法比较",
    requestLabel: "请求",
    historyBadge: "历史回放",
    download: "下载结果",
    share: "复制分享链接",
    copied: "分享链接已复制。",
    shareFailed: "无法复制分享链接。",
    commandCopied: "命令已复制。",
    input: "输入",
    output: "输出",
    latency: "延迟",
    cpuTime: "CPU 时间",
    multiadds: "Multi-Adds",
    memoryUsage: "预估内存",
    flops: "FLOPs",
    noResult: "超分结果会显示在这里",
    compareGallery: "A/B 对比画廊",
    compareGalleryBody:
      "点击不同方法的卡片，就能对同一请求在 EPNet、Bicubic 和 Baseline 之间做对比。",
    oneClickRepro: "一键复现实验",
    oneClickReproBody:
      "这里的命令会跟随当前 Demo 配置变化，适合在演示时从前端直接切换到 CLI 复现实验。",
    copyCommand: "复制命令",
    trainCommand: "训练命令",
    evalCommand: "评估命令",
    inferCommand: "推理命令",
    analyticsEyebrow: "分析",
    analyticsTitle: "使用与运行时历史",
    requests: "请求数",
    sessions: "会话数",
    avgLatency: "平均延迟",
    p50Latency: "P50 延迟",
    p95Latency: "P95 延迟",
    avgMegapixels: "平均输出 MP",
    latencySeries: "延迟曲线",
    upscaleDistribution: "倍率分布",
    recentUsage: "最近使用记录",
    inferenceActivity: "推理活动",
    replay: "回放",
    recentShare: "分享",
    recentEmpty: "还没有使用记录。先跑一次推理，仪表盘就会自动填充。",
    scaleMismatch: "当前已加载的 EPNet checkpoint 不支持这个倍率，请切换倍率或改用 Bicubic/Baseline。",
    historyLoaded: "已加载历史回放。",
    latestRequest: "最近请求"
  }
} as const;

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

function formatMemoryBytes(value: number): string {
  if (value >= 1024 ** 3) {
    return `${(value / 1024 ** 3).toFixed(2)} GB`;
  }
  if (value >= 1024 ** 2) {
    return `${(value / 1024 ** 2).toFixed(1)} MB`;
  }
  if (value >= 1024) {
    return `${(value / 1024).toFixed(1)} KB`;
  }
  return `${Math.round(value)} B`;
}

function formatShortDate(value: string): string {
  return new Date(value).toLocaleString();
}

function formatMethodLabel(method: string): string {
  if (method === "epnet") {
    return "EPNet";
  }
  if (method === "baseline") {
    return "Baseline";
  }
  if (method === "bicubic") {
    return "Bicubic";
  }
  return method;
}

function buildSessionId(): string {
  if (typeof window !== "undefined" && "randomUUID" in window.crypto) {
    return `demo-${window.crypto.randomUUID().slice(0, 8)}`;
  }
  return `demo-${Date.now().toString(36)}`;
}

function normalizeInference(payload: InferenceResponse): DisplayResult {
  return {
    requestId: payload.request_id,
    createdAt: payload.created_at,
    sessionId: payload.options.session_id,
    method: payload.options.method,
    checkpointName: payload.options.checkpoint_name,
    scale: payload.options.scale,
    outputFormat: payload.output.format,
    tileSize: payload.options.tile_size,
    inputResolution: payload.input.resolution,
    outputResolution: payload.output.resolution,
    latencyMs: payload.runtime.latency_ms,
    cpuTimeMs: payload.runtime.cpu_time_ms,
    parameterCount: payload.runtime.parameter_count,
    estimatedMacs: payload.runtime.estimated_macs,
    estimatedFlops: payload.runtime.estimated_flops,
    inputImageUrl: resolveApiAssetUrl(payload.input_image_url),
    inputBytes: payload.input.bytes,
    pipelineStages: payload.pipeline.stages,
    variants: [
      {
        method: payload.options.method,
        label: formatMethodLabel(payload.options.method),
        imageUrl: resolveApiAssetUrl(payload.output_image_url),
        format: payload.output.format,
        bytes: payload.output.bytes,
        isPrimary: true
      },
      ...payload.comparisons.map((comparison) => ({
        method: comparison.method,
        label: comparison.label,
        imageUrl: resolveApiAssetUrl(comparison.image_url),
        format: comparison.format,
        bytes: comparison.bytes,
        isPrimary: false
      }))
    ]
  };
}

function normalizeHistory(payload: HistoryEvent): DisplayResult {
  return {
    requestId: payload.request_id,
    createdAt: payload.created_at,
    sessionId: payload.session_id,
    method: payload.method,
    checkpointName: payload.checkpoint_name,
    scale: payload.upscale,
    outputFormat: payload.output_format,
    tileSize: payload.tile_size,
    inputResolution: payload.input_resolution,
    outputResolution: payload.output_resolution,
    latencyMs: payload.latency_ms,
    cpuTimeMs: 0,
    parameterCount: payload.parameter_count,
    estimatedMacs: payload.estimated_macs,
    estimatedFlops: payload.estimated_flops,
    inputImageUrl: resolveApiAssetUrl(payload.input_image_url),
    inputBytes: 0,
    pipelineStages: [],
    variants: [
      {
        method: payload.method,
        label: formatMethodLabel(payload.method),
        imageUrl: resolveApiAssetUrl(payload.output_image_url),
        format: payload.output_format,
        bytes: 0,
        isPrimary: true
      }
    ]
  };
}

async function copyText(text: string) {
  await navigator.clipboard.writeText(text);
}

export function EpnetDashboard() {
  const { language } = useLanguage();
  const text = useMemo(() => copy[language], [language]);

  const [model, setModel] = useState<ModelInfo | null>(null);
  const [analytics, setAnalytics] = useState<AnalyticsSummary | null>(null);
  const [result, setResult] = useState<DisplayResult | null>(null);
  const [batchPayload, setBatchPayload] = useState<BatchInferenceResponse | null>(null);
  const [inputPreview, setInputPreview] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [mode, setMode] = useState<Mode>("single");
  const [sessionId, setSessionId] = useState("demo-session");
  const [method, setMethod] = useState("epnet");
  const [scale, setScale] = useState(4);
  const [checkpointName, setCheckpointName] = useState("");
  const [outputFormat, setOutputFormat] = useState("PNG");
  const [tileSize, setTileSize] = useState(0);
  const [selectedVariantMethod, setSelectedVariantMethod] = useState("epnet");
  const [queueItems, setQueueItems] = useState<QueueItem[]>([]);

  useEffect(() => {
    const storedSession = window.localStorage.getItem(SESSION_STORAGE_KEY);
    if (storedSession) {
      setSessionId(storedSession);
      return;
    }
    const generated = buildSessionId();
    setSessionId(generated);
    window.localStorage.setItem(SESSION_STORAGE_KEY, generated);
  }, []);

  useEffect(() => {
    if (sessionId) {
      window.localStorage.setItem(SESSION_STORAGE_KEY, sessionId);
    }
  }, [sessionId]);

  useEffect(() => {
    return () => {
      if (inputPreview) {
        URL.revokeObjectURL(inputPreview);
      }
    };
  }, [inputPreview]);

  useEffect(() => {
    if (!notice) {
      return;
    }
    const timer = window.setTimeout(() => setNotice(null), 2200);
    return () => window.clearTimeout(timer);
  }, [notice]);

  async function refreshAnalytics() {
    try {
      setAnalytics(await fetchAnalyticsSummary());
    } catch (refreshError) {
      console.error(refreshError);
    }
  }

  async function loadHistory(requestId: string) {
    const history = await fetchHistoryEvent(requestId);
    const normalized = normalizeHistory(history);
    setResult(normalized);
    setSelectedVariantMethod(normalized.method);
    setInputPreview(null);
    setNotice(text.historyLoaded);
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

  useEffect(() => {
    if (!model || checkpointName || method !== "epnet") {
      return;
    }
    const preferred =
      model.available_checkpoints.find((checkpoint) => checkpoint.scale === 4) ??
      model.available_checkpoints[0];
    if (preferred) {
      setCheckpointName(preferred.name);
      if (preferred.scale !== scale) {
        setScale(preferred.scale);
      }
    }
  }, [checkpointName, method, model, scale]);

  useEffect(() => {
    const requestId = new URLSearchParams(window.location.search).get("request_id");
    if (requestId) {
      void (async () => {
        try {
          const history = await fetchHistoryEvent(requestId);
          const normalized = normalizeHistory(history);
          setResult(normalized);
          setSelectedVariantMethod(normalized.method);
          setInputPreview(null);
          setNotice(text.historyLoaded);
        } catch (historyError) {
          setError(
            historyError instanceof Error ? historyError.message : "Unable to load shared result."
          );
        }
      })();
    }
  }, [text.historyLoaded]);

  const epnetCheckpoints = useMemo(
    () => model?.available_checkpoints ?? [],
    [model]
  );

  const epnetScales = useMemo(() => {
    const unique = new Set(epnetCheckpoints.map((checkpoint) => checkpoint.scale));
    return Array.from(unique).sort((left, right) => left - right);
  }, [epnetCheckpoints]);

  useEffect(() => {
    if (method === "epnet" && epnetScales.length > 0 && !epnetScales.includes(scale)) {
      setScale(epnetScales[0]);
      setError(text.scaleMismatch);
    }
  }, [epnetScales, method, scale, text.scaleMismatch]);

  const checkpointOptions = useMemo(() => {
    return epnetCheckpoints.filter((checkpoint) => checkpoint.scale === scale);
  }, [epnetCheckpoints, scale]);

  useEffect(() => {
    if (method !== "epnet") {
      setCheckpointName("");
      return;
    }
    if (
      checkpointName &&
      !checkpointOptions.some((checkpoint) => checkpoint.name === checkpointName)
    ) {
      setCheckpointName("");
      return;
    }
    if (!checkpointName && checkpointOptions[0]) {
      setCheckpointName(checkpointOptions[0].name);
    }
  }, [checkpointName, checkpointOptions, method]);

  useEffect(() => {
    if (method !== "epnet" || !checkpointName) {
      return;
    }
    void (async () => {
      try {
        const info = await fetchModelInfo(checkpointName);
        setModel(info);
      } catch (loadError) {
        setError(loadError instanceof Error ? loadError.message : "Unable to load checkpoint info.");
      }
    })();
  }, [checkpointName, method]);

  const availableScales = useMemo(() => {
    if (method === "epnet" && epnetScales.length > 0) {
      return epnetScales;
    }
    return model?.supported_scales ?? [2, 3, 4];
  }, [epnetScales, method, model?.supported_scales]);

  const activeVariant = useMemo(() => {
    if (!result) {
      return null;
    }
    return (
      result.variants.find((variant) => variant.method === selectedVariantMethod) ?? result.variants[0]
    );
  }, [result, selectedVariantMethod]);

  const activeInputPreview = inputPreview ?? result?.inputImageUrl ?? null;
  const activeOutputPreview = activeVariant?.imageUrl ?? null;

  const batchAggregate = batchPayload?.aggregate ?? null;

  const commandSet = useMemo(() => {
    const selectedCheckpoint =
      checkpointOptions.find((checkpoint) => checkpoint.name === checkpointName) ?? checkpointOptions[0];
    const checkpointPath = selectedCheckpoint?.checkpoint_path ?? "checkpoints/demo_x4.pt";
    return {
      infer: `PYTHONPATH=src .venv/bin/epnet-infer --checkpoint ${checkpointPath} --input data/samples/demo_input.png --output outputs/demo_output.png`,
      eval: `PYTHONPATH=src .venv/bin/epnet-eval --checkpoint ${checkpointPath} --hr-dir path/to/benchmark_hr`,
      train: `PYTHONPATH=src .venv/bin/epnet-train --output checkpoints/epnet_x${scale}.pt --scale ${scale} --train-dir path/to/div2k_train_hr --steps 1000000`
    };
  }, [checkpointName, checkpointOptions, scale]);

  function updateShareUrl(requestId: string) {
    const url = new URL(window.location.href);
    url.searchParams.set("request_id", requestId);
    window.history.replaceState({}, "", url);
  }

  async function copyShareLink(requestId: string) {
    try {
      const url = new URL(window.location.href);
      url.searchParams.set("request_id", requestId);
      await copyText(url.toString());
      setNotice(text.copied);
    } catch {
      setError(text.shareFailed);
    }
  }

  async function copyCommand(command: string) {
    try {
      await copyText(command);
      setNotice(text.commandCopied);
    } catch {
      setError(text.shareFailed);
    }
  }

  function resetSession() {
    const generated = buildSessionId();
    setSessionId(generated);
    window.localStorage.setItem(SESSION_STORAGE_KEY, generated);
  }

  async function handleFiles(files: File[]) {
    if (files.length === 0) {
      return;
    }

    for (const file of files) {
      if (!ALLOWED_FILE_TYPES.has(file.type)) {
        setError("Please upload a PNG, JPEG, WEBP, or BMP image.");
        return;
      }
      if (file.size > MAX_UPLOAD_BYTES) {
        setError("The uploaded file is too large for the demo limit.");
        return;
      }
    }

    setError(null);
    setIsLoading(true);
    setNotice(null);

    if (mode === "single") {
      const file = files[0];
      setInputPreview((previous) => {
        if (previous) {
          URL.revokeObjectURL(previous);
        }
        return URL.createObjectURL(file);
      });
      setBatchPayload(null);

      try {
        const payload = await superResolve(file, {
          sessionId,
          method,
          scale,
          outputFormat,
          tileSize,
          checkpointName: checkpointName || undefined
        });
        const normalized = normalizeInference(payload);
        setResult(normalized);
        setSelectedVariantMethod(normalized.method);
        setModel(payload.model);
        updateShareUrl(payload.request_id);
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
      return;
    }

    setInputPreview((previous) => {
      if (previous) {
        URL.revokeObjectURL(previous);
      }
      return null;
    });
    setQueueItems(
      files.map((file) => ({
        id: `${file.name}-${file.size}-${file.lastModified}`,
        name: file.name,
        size: file.size,
        status: "running"
      }))
    );

    try {
      const payload = await batchSuperResolve(files, {
        sessionId,
        method,
        scale,
        outputFormat,
        tileSize,
        checkpointName: checkpointName || undefined
      });
      setBatchPayload(payload);
      const normalizedResults = payload.results.map(normalizeInference);
      if (normalizedResults[0]) {
        setResult(normalizedResults[0]);
        setSelectedVariantMethod(normalizedResults[0].method);
        updateShareUrl(normalizedResults[0].requestId);
      }
      if (payload.results[0]) {
        setModel(payload.results[0].model);
      }
      setQueueItems((current) =>
        current.map((item, index) => ({
          ...item,
          status: "done",
          requestId: payload.results[index]?.request_id
        }))
      );
      await refreshAnalytics();
    } catch (requestError) {
      setQueueItems((current) => current.map((item) => ({ ...item, status: "error" })));
      setError(
        requestError instanceof Error
          ? requestError.message
          : "Batch inference failed. Please check the backend logs."
      );
    } finally {
      setIsLoading(false);
    }
  }

  function onFileChange(event: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(event.target.files ?? []);
    if (files.length > 0) {
      void handleFiles(files);
    }
  }

  function onDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    setIsDragging(false);
    const files = Array.from(event.dataTransfer.files ?? []);
    if (files.length > 0) {
      void handleFiles(mode === "single" ? [files[0]] : files);
    }
  }

  function renderCheckpointOption(checkpoint: CheckpointOption) {
    return (
      <option key={checkpoint.name} value={checkpoint.name}>
        {checkpoint.name}
      </option>
    );
  }

  function renderRecentEvent(event: EventPreview) {
    return (
      <motion.div
        key={event.request_id}
        layout
        className="rounded-[24px] border border-ink/8 bg-white/70 p-4"
      >
        <div className="grid gap-4 md:grid-cols-[92px_1fr_auto] md:items-center">
          <div className="overflow-hidden rounded-[18px] bg-sand">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={resolveApiAssetUrl(event.output_image_url)}
              alt={`Result ${event.request_id}`}
              className="h-24 w-full object-cover"
            />
          </div>
          <div>
            <p className="mono text-xs uppercase tracking-[0.28em] text-ink/45">
              {formatShortDate(event.created_at)}
            </p>
            <p className="mt-2 text-sm">
              {formatMethodLabel(event.method)} · x{event.upscale} ·{" "}
              {formatResolution(event.input_resolution.width, event.input_resolution.height)} to{" "}
              {formatResolution(event.output_resolution.width, event.output_resolution.height)}
            </p>
            <p className="mt-1 text-xs text-ink/55">
              {event.checkpoint_name} · {event.output_format} · {event.session_id}
            </p>
          </div>
          <div className="flex flex-col items-end gap-2">
            <p className="text-sm font-medium">{formatLatency(event.latency_ms)}</p>
            <div className="flex gap-2">
              <button
                type="button"
                onClick={() => void loadHistory(event.request_id)}
                className="mono rounded-full bg-dusk px-3 py-2 text-[11px] uppercase tracking-[0.24em] text-white"
              >
                {text.replay}
              </button>
              <button
                type="button"
                onClick={() => void copyShareLink(event.request_id)}
                className="mono rounded-full border border-dusk/15 px-3 py-2 text-[11px] uppercase tracking-[0.24em] text-ink/70"
              >
                {text.recentShare}
              </button>
            </div>
          </div>
        </div>
      </motion.div>
    );
  }

  return (
    <main className="relative overflow-hidden px-4 py-6 md:px-8">
      <div className="mx-auto max-w-7xl space-y-6">
        <AnimatePresence>
          {notice ? (
            <motion.div
              initial={{ opacity: 0, y: -12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -12 }}
              className="fixed right-4 top-24 z-50 rounded-full bg-dusk px-4 py-2 text-sm text-white shadow-panel"
            >
              {notice}
            </motion.div>
          ) : null}
        </AnimatePresence>

        <section className="panel hero-grid relative overflow-hidden px-6 py-8 md:px-10 md:py-12">
          <div className="absolute inset-0 bg-hero-grid opacity-40" />
          <div className="relative grid gap-8 lg:grid-cols-[1.3fr_0.9fr]">
            <div>
              <motion.p
                {...fadeUp}
                transition={{ duration: 0.38 }}
                className="mono text-xs uppercase tracking-[0.4em] text-ember"
              >
                {text.demoEyebrow}
              </motion.p>
              <motion.h1
                {...fadeUp}
                transition={{ duration: 0.42, delay: 0.05 }}
                className="mt-4 max-w-3xl text-4xl font-semibold leading-tight md:text-6xl"
              >
                {text.heroTitle}
              </motion.h1>
              <motion.p
                {...fadeUp}
                transition={{ duration: 0.45, delay: 0.1 }}
                className="mt-4 max-w-2xl text-base text-ink/72 md:text-lg"
              >
                {text.heroBody}
              </motion.p>

              <motion.div
                {...fadeUp}
                transition={{ duration: 0.45, delay: 0.15 }}
                className="mt-6 grid gap-3 sm:grid-cols-2 xl:grid-cols-4"
              >
                <MetricCard
                  label={text.parameters}
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
                  label={text.estimatedMacs}
                  value={
                    model ? (
                      <AnimatedCounter value={model.estimated_multiadds} format={formatCount} />
                    ) : (
                      "Loading"
                    )
                  }
                  accent="moss"
                />
                <MetricCard
                  label={text.estimatedMemory}
                  value={
                    model ? (
                      <AnimatedCounter value={model.estimated_memory_bytes} format={formatMemoryBytes} />
                    ) : (
                      "Loading"
                    )
                  }
                />
                <MetricCard
                  label={text.referenceLatency}
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
                <p className="mono text-xs uppercase tracking-[0.28em] text-dusk/70">
                  {text.deploymentPanel}
                </p>
                <div className="mt-4 space-y-3 text-sm text-ink/80">
                  <div className="flex items-center justify-between gap-4">
                    <span>{text.modelVersion}</span>
                    <span className="mono">{model?.deployment.model_version ?? "Loading"}</span>
                  </div>
                  <div className="flex items-center justify-between gap-4">
                    <span>{text.checkpointSource}</span>
                    <span className="mono break-all text-right">
                      {model?.deployment.checkpoint_source ?? "Loading"}
                    </span>
                  </div>
                  <div className="flex items-center justify-between gap-4">
                    <span>{text.deviceTarget}</span>
                    <span className="mono break-all text-right">
                      {model?.deployment.device_target ?? "Loading"}
                    </span>
                  </div>
                  <div className="flex items-center justify-between gap-4">
                    <span>{text.gitCommit}</span>
                    <span className="mono break-all text-right">
                      {model?.deployment.git_commit ?? "Loading"}
                    </span>
                  </div>
                  <div className="rounded-2xl bg-white/70 p-4 text-xs text-ink/65">
                    <p className="mono uppercase tracking-[0.22em] text-ink/45">{text.buildTime}</p>
                    <p className="mt-2 break-all">{model?.deployment.build_time ?? "Loading"}</p>
                  </div>
                </div>
              </div>

              <div className="rounded-[28px] border border-dusk/10 bg-gradient-to-br from-dusk/[0.08] to-ember/[0.08] p-5">
                <p className="mono text-xs uppercase tracking-[0.28em] text-dusk/60">
                  {text.presenterShortcut}
                </p>
                <p className="mt-3 text-sm text-ink/72">{text.presenterShortcutBody}</p>
                <Link
                  href="/model"
                  className="mono mt-4 inline-flex rounded-full bg-dusk px-4 py-2 text-xs uppercase tracking-[0.28em] text-white"
                >
                  {text.openModelPage}
                </Link>
              </div>
            </motion.div>
          </div>
        </section>

        <section className="grid gap-6 xl:grid-cols-[0.74fr_1.26fr]">
          <motion.div
            {...fadeUp}
            transition={{ duration: 0.45, delay: 0.04 }}
            className="panel p-5 md:p-6"
          >
            <p className="mono text-xs uppercase tracking-[0.28em] text-ember">{text.controlsEyebrow}</p>
            <h2 className="mt-2 text-2xl font-semibold">{text.controlsTitle}</h2>
            <p className="mt-3 text-sm text-ink/70">{text.controlsBody}</p>

            <div className="mt-6 grid gap-4">
              <div className="grid gap-3 sm:grid-cols-2">
                <button
                  type="button"
                  onClick={() => setMode("single")}
                  className={`rounded-[22px] px-4 py-3 text-left transition ${
                    mode === "single" ? "bg-dusk text-white shadow-panel" : "bg-sand text-ink/72"
                  }`}
                >
                  {text.singleMode}
                </button>
                <button
                  type="button"
                  onClick={() => setMode("batch")}
                  className={`rounded-[22px] px-4 py-3 text-left transition ${
                    mode === "batch" ? "bg-dusk text-white shadow-panel" : "bg-sand text-ink/72"
                  }`}
                >
                  {text.batchMode}
                </button>
              </div>

              <div className="rounded-[24px] bg-sand p-4">
                <label className="mono text-xs uppercase tracking-[0.28em] text-ink/55">
                  {text.sessionId}
                </label>
                <div className="mt-3 flex gap-2">
                  <input
                    value={sessionId}
                    onChange={(event) => setSessionId(event.target.value)}
                    className="w-full rounded-2xl border border-dusk/10 bg-white px-4 py-3 text-sm outline-none focus:border-dusk/40"
                  />
                  <button
                    type="button"
                    onClick={resetSession}
                    className="mono rounded-2xl bg-ink px-4 py-3 text-xs uppercase tracking-[0.22em] text-white"
                  >
                    {text.resetSession}
                  </button>
                </div>
              </div>

              <div className="grid gap-4 sm:grid-cols-2">
                <div className="rounded-[24px] bg-sand p-4">
                  <label className="mono text-xs uppercase tracking-[0.28em] text-ink/55">
                    {text.method}
                  </label>
                  <select
                    value={method}
                    onChange={(event) => setMethod(event.target.value)}
                    className="mt-3 w-full rounded-2xl border border-dusk/10 bg-white px-4 py-3 text-sm outline-none"
                  >
                    {(model?.supported_methods ?? ["epnet", "bicubic", "baseline"]).map((option) => (
                      <option key={option} value={option}>
                        {formatMethodLabel(option)}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="rounded-[24px] bg-sand p-4">
                  <label className="mono text-xs uppercase tracking-[0.28em] text-ink/55">
                    {text.scale}
                  </label>
                  <select
                    value={scale}
                    onChange={(event) => setScale(Number(event.target.value))}
                    className="mt-3 w-full rounded-2xl border border-dusk/10 bg-white px-4 py-3 text-sm outline-none"
                  >
                    {availableScales.map((option) => (
                      <option key={option} value={option}>
                        x{option}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="rounded-[24px] bg-sand p-4">
                  <label className="mono text-xs uppercase tracking-[0.28em] text-ink/55">
                    {text.checkpoint}
                  </label>
                  <select
                    value={checkpointName}
                    onChange={(event) => setCheckpointName(event.target.value)}
                    disabled={method !== "epnet"}
                    className="mt-3 w-full rounded-2xl border border-dusk/10 bg-white px-4 py-3 text-sm outline-none disabled:opacity-55"
                  >
                    <option value="">{text.autoCheckpoint}</option>
                    {checkpointOptions.map(renderCheckpointOption)}
                  </select>
                </div>
                <div className="rounded-[24px] bg-sand p-4">
                  <label className="mono text-xs uppercase tracking-[0.28em] text-ink/55">
                    {text.outputFormat}
                  </label>
                  <select
                    value={outputFormat}
                    onChange={(event) => setOutputFormat(event.target.value)}
                    className="mt-3 w-full rounded-2xl border border-dusk/10 bg-white px-4 py-3 text-sm outline-none"
                  >
                    {["PNG", "JPEG", "WEBP", "BMP"].map((option) => (
                      <option key={option} value={option}>
                        {option}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              <div className="rounded-[24px] bg-sand p-4">
                <div className="flex items-center justify-between gap-4">
                  <div>
                    <label className="mono text-xs uppercase tracking-[0.28em] text-ink/55">
                      {text.tileMode}
                    </label>
                    <p className="mt-2 text-xs text-ink/55">{text.tileHint}</p>
                  </div>
                  <select
                    value={tileSize}
                    onChange={(event) => setTileSize(Number(event.target.value))}
                    disabled={method !== "epnet"}
                    className="rounded-2xl border border-dusk/10 bg-white px-4 py-3 text-sm outline-none disabled:opacity-55"
                  >
                    <option value={0}>{text.tileOff}</option>
                    <option value={256}>256</option>
                    <option value={384}>384</option>
                    <option value={512}>512</option>
                  </select>
                </div>
              </div>
            </div>
          </motion.div>

          <motion.div
            {...fadeUp}
            transition={{ duration: 0.45, delay: 0.08 }}
            className="panel p-5 md:p-6"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="mono text-xs uppercase tracking-[0.28em] text-ember">{text.uploadEyebrow}</p>
                <h2 className="mt-2 text-2xl font-semibold">{text.uploadTitle}</h2>
              </div>
              <AnimatePresence>
                {isLoading ? (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.95 }}
                    className="mono rounded-full bg-dusk px-3 py-2 text-xs uppercase tracking-[0.26em] text-white"
                  >
                    {text.requestActive}
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
              className={`mt-6 flex min-h-72 cursor-pointer flex-col items-center justify-center rounded-[30px] border-2 border-dashed px-6 text-center transition ${
                isDragging
                  ? "scale-[1.01] border-ember bg-ember/12 shadow-[0_24px_70px_rgba(194,84,47,0.18)]"
                  : "border-dusk/20 bg-white/60 hover:border-dusk/45 hover:shadow-[0_20px_50px_rgba(19,32,47,0.08)]"
              }`}
            >
              <input
                type="file"
                accept="image/*"
                multiple={mode === "batch"}
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
                  <p className="text-lg font-medium">
                    {mode === "single" ? text.chooseImage : text.chooseImages}
                  </p>
                  <p className="mt-2 text-sm text-ink/65">{text.uploadBody}</p>
                </div>
                <span className="mono inline-flex rounded-full bg-ink px-4 py-2 text-xs uppercase tracking-[0.3em] text-white">
                  {mode === "single" ? text.chooseImage : text.chooseImages}
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
              <PipelineTimeline isLoading={isLoading} stages={result?.pipelineStages ?? null} />
            </div>

            <div className="mt-6 rounded-[24px] bg-sand p-4">
              <div className="flex items-center justify-between gap-4">
                <div>
                  <p className="mono text-xs uppercase tracking-[0.28em] text-ink/55">
                    {text.queueTitle}
                  </p>
                  {batchAggregate ? (
                    <p className="mt-2 text-xs text-ink/55">{text.aggregateTitle}</p>
                  ) : null}
                </div>
                {batchAggregate ? (
                  <span className="mono rounded-full bg-dusk px-3 py-2 text-[11px] uppercase tracking-[0.24em] text-white">
                    {batchAggregate.completed_files}/{batchAggregate.total_files}
                  </span>
                ) : null}
              </div>

              {queueItems.length > 0 ? (
                <div className="mt-4 space-y-3">
                  {queueItems.map((item) => (
                    <button
                      key={item.id}
                      type="button"
                      onClick={() => {
                        const selected = batchPayload?.results.find(
                          (entry) => entry.request_id === item.requestId
                        );
                        if (selected) {
                          const normalized = normalizeInference(selected);
                          setResult(normalized);
                          setSelectedVariantMethod(normalized.method);
                        }
                      }}
                      className="flex w-full items-center justify-between rounded-[20px] border border-dusk/10 bg-white/70 px-4 py-3 text-left"
                    >
                      <div>
                        <p className="text-sm font-medium">{item.name}</p>
                        <p className="text-xs text-ink/55">{formatCount(item.size)}B</p>
                      </div>
                      <span className="mono text-[11px] uppercase tracking-[0.24em] text-ink/50">
                        {item.status}
                      </span>
                    </button>
                  ))}
                </div>
              ) : (
                <div className="mt-4 rounded-[20px] bg-white/70 px-4 py-5 text-sm text-ink/60">
                  {text.queueEmpty}
                </div>
              )}

              {batchAggregate ? (
                <div className="mt-4 grid gap-3 sm:grid-cols-4">
                  <MetricCard label={text.totalFiles} value={batchAggregate.total_files} accent="ember" />
                  <MetricCard label={text.completedFiles} value={batchAggregate.completed_files} />
                  <MetricCard
                    label={text.averageLatency}
                    value={<AnimatedCounter value={batchAggregate.average_latency_ms} format={formatLatency} />}
                    accent="moss"
                  />
                  <MetricCard
                    label={text.totalMegapixels}
                    value={batchAggregate.total_output_megapixels.toFixed(2)}
                  />
                </div>
              ) : null}
            </div>
          </motion.div>
        </section>

        <section className="grid gap-6 lg:grid-cols-[1.02fr_0.98fr]">
          <motion.div
            {...fadeUp}
            transition={{ duration: 0.45, delay: 0.1 }}
            className="panel p-5 md:p-6"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="mono text-xs uppercase tracking-[0.28em] text-moss">{text.resultEyebrow}</p>
                <h2 className="mt-2 text-2xl font-semibold">{text.resultTitle}</h2>
              </div>
              {result ? (
                <div className="min-w-0 text-right">
                  <div className="mono text-xs text-ink/55">
                    {text.requestLabel} {result.requestId.slice(0, 8)}
                  </div>
                  <div className="mt-1 text-xs text-ink/55">
                    {result.pipelineStages.length === 0 ? text.historyBadge : text.latestRequest}
                  </div>
                </div>
              ) : null}
            </div>

            <div className="mt-6 grid gap-4 md:grid-cols-2">
              <div className="rounded-[24px] bg-sand p-3">
                <p className="mono px-2 text-xs uppercase tracking-[0.28em] text-ink/55">{text.input}</p>
                <div className="mt-3 aspect-square overflow-hidden rounded-[20px] bg-white">
                  {activeInputPreview ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img src={activeInputPreview} alt="Input preview" className="h-full w-full object-cover" />
                  ) : (
                    <div className="flex h-full items-center justify-center text-sm text-ink/45">
                      {text.noResult}
                    </div>
                  )}
                </div>
              </div>
              <div className="rounded-[24px] bg-sand p-3">
                <div className="flex items-center justify-between px-2">
                  <p className="mono text-xs uppercase tracking-[0.28em] text-ink/55">{text.output}</p>
                  <p className="text-xs text-ink/45">{activeVariant?.label ?? "EPNet"}</p>
                </div>
                <div className="mt-3 aspect-square overflow-hidden rounded-[20px] bg-white">
                  <ZoomableCompare inputPreview={activeInputPreview} outputPreview={activeOutputPreview} />
                </div>
              </div>
            </div>

            <div className="mt-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
              <MetricCard
                label={text.input}
                value={
                  result
                    ? formatResolution(result.inputResolution.width, result.inputResolution.height)
                    : "Waiting"
                }
              />
              <MetricCard
                label={text.output}
                value={
                  result
                    ? formatResolution(result.outputResolution.width, result.outputResolution.height)
                    : "Waiting"
                }
                accent="ember"
              />
              <MetricCard
                label={text.latency}
                value={
                  result ? <AnimatedCounter value={result.latencyMs} format={formatLatency} /> : "Waiting"
                }
                accent="moss"
              />
              <MetricCard
                label={text.flops}
                value={
                  result ? <AnimatedCounter value={result.estimatedFlops} format={formatCount} /> : "Waiting"
                }
              />
              <MetricCard
                label={text.cpuTime}
                value={
                  result ? <AnimatedCounter value={result.cpuTimeMs} format={formatLatency} /> : "Waiting"
                }
              />
              <MetricCard
                label={text.multiadds}
                value={
                  result ? <AnimatedCounter value={result.estimatedMacs} format={formatCount} /> : "Waiting"
                }
                accent="ember"
              />
              <MetricCard
                label={text.parameters}
                value={
                  result ? <AnimatedCounter value={result.parameterCount} format={formatCount} /> : "Waiting"
                }
              />
              <MetricCard
                label={text.memoryUsage}
                value={
                  result && model ? (
                    <AnimatedCounter value={model.estimated_memory_bytes} format={formatMemoryBytes} />
                  ) : (
                    "Waiting"
                  )
                }
                accent="moss"
              />
            </div>

            {result ? (
              <div className="mt-4 flex flex-wrap gap-3">
                <a
                  href={activeVariant?.imageUrl}
                  download={`${result.requestId}.${result.outputFormat.toLowerCase()}`}
                  className="mono inline-flex rounded-full bg-dusk px-4 py-3 text-xs uppercase tracking-[0.24em] text-white"
                >
                  {text.download}
                </a>
                <button
                  type="button"
                  onClick={() => void copyShareLink(result.requestId)}
                  className="mono rounded-full border border-dusk/15 px-4 py-3 text-xs uppercase tracking-[0.24em] text-ink/70"
                >
                  {text.share}
                </button>
              </div>
            ) : null}

            <div className="mt-6 rounded-[24px] bg-sand p-4">
              <p className="mono text-xs uppercase tracking-[0.28em] text-ink/55">{text.oneClickRepro}</p>
              <p className="mt-2 text-sm text-ink/62">{text.oneClickReproBody}</p>
              <div className="mt-4 grid gap-3">
                {[
                  { label: text.inferCommand, value: commandSet.infer },
                  { label: text.evalCommand, value: commandSet.eval },
                  { label: text.trainCommand, value: commandSet.train }
                ].map((command) => (
                  <div
                    key={command.label}
                    className="rounded-[20px] border border-dusk/10 bg-white/75 p-4"
                  >
                    <div className="flex items-center justify-between gap-4">
                      <p className="mono text-xs uppercase tracking-[0.24em] text-ink/55">
                        {command.label}
                      </p>
                      <button
                        type="button"
                        onClick={() => void copyCommand(command.value)}
                        className="mono rounded-full border border-dusk/15 px-3 py-2 text-[11px] uppercase tracking-[0.24em] text-ink/70"
                      >
                        {text.copyCommand}
                      </button>
                    </div>
                    <pre className="mt-3 overflow-x-auto whitespace-pre-wrap rounded-[16px] bg-ink px-4 py-3 text-xs text-white">
                      {command.value}
                    </pre>
                  </div>
                ))}
              </div>
            </div>
          </motion.div>

          <motion.div
            {...fadeUp}
            transition={{ duration: 0.45, delay: 0.12 }}
            className="space-y-6"
          >
            <div className="panel p-5 md:p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="mono text-xs uppercase tracking-[0.28em] text-dusk/60">{text.analyticsEyebrow}</p>
                  <h2 className="mt-2 text-2xl font-semibold">{text.analyticsTitle}</h2>
                </div>
                <button
                  type="button"
                  onClick={() => void refreshAnalytics()}
                  className="mono rounded-full bg-ink px-4 py-2 text-xs uppercase tracking-[0.26em] text-white"
                >
                  Refresh
                </button>
              </div>

              <div className="mt-6 grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
                <MetricCard
                  label={text.requests}
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
                  label={text.sessions}
                  value={
                    analytics ? (
                      <AnimatedCounter value={analytics.session_count} format={formatCount} />
                    ) : (
                      "0"
                    )
                  }
                />
                <MetricCard
                  label={text.avgLatency}
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
                  label={text.p50Latency}
                  value={
                    analytics ? (
                      <AnimatedCounter value={analytics.p50_latency_ms} format={formatLatency} />
                    ) : (
                      "0 ms"
                    )
                  }
                />
                <MetricCard
                  label={text.p95Latency}
                  value={
                    analytics ? (
                      <AnimatedCounter value={analytics.p95_latency_ms} format={formatLatency} />
                    ) : (
                      "0 ms"
                    )
                  }
                />
                <MetricCard
                  label={text.avgMegapixels}
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
                  <p className="mono text-xs uppercase tracking-[0.28em] text-ink/55">
                    {text.latencySeries}
                  </p>
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
                    {text.upscaleDistribution}
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
              <p className="mono text-xs uppercase tracking-[0.28em] text-dusk/60">{text.recentUsage}</p>
              <h2 className="mt-2 text-2xl font-semibold">{text.inferenceActivity}</h2>
              <div className="mt-6 space-y-3">
                {(analytics?.recent_events ?? []).length > 0 ? (
                  analytics?.recent_events.map(renderRecentEvent)
                ) : (
                  <div className="rounded-[24px] bg-sand p-5 text-sm text-ink/60">
                    {text.recentEmpty}
                  </div>
                )}
              </div>
            </div>
          </motion.div>
        </section>

        <motion.section
          {...fadeUp}
          transition={{ duration: 0.45, delay: 0.14 }}
          className="panel p-5 md:p-6"
        >
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div className="max-w-3xl">
              <p className="mono text-xs uppercase tracking-[0.28em] text-ink/55">
                {text.compareGallery}
              </p>
              <h2 className="mt-2 text-2xl font-semibold">{text.compareGallery}</h2>
              <p className="mt-3 text-sm text-ink/62">{text.compareGalleryBody}</p>
            </div>
            {result ? (
              <div className="mono rounded-full bg-dusk px-3 py-2 text-[11px] uppercase tracking-[0.24em] text-white">
                {result.variants.length} variants
              </div>
            ) : null}
          </div>

          <div className="mt-6 grid gap-4 md:grid-cols-2 xl:grid-cols-3">
            {(result?.variants ?? []).map((variant) => {
              const active = selectedVariantMethod === variant.method;
              return (
                <button
                  key={variant.method}
                  type="button"
                  onClick={() => setSelectedVariantMethod(variant.method)}
                  className={`min-w-0 overflow-hidden rounded-[28px] border p-4 text-left transition ${
                    active
                      ? "border-dusk bg-white shadow-panel"
                      : "border-dusk/10 bg-white/70 hover:border-dusk/30"
                  }`}
                >
                  <div className="aspect-[16/10] overflow-hidden rounded-[22px] bg-sand">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={variant.imageUrl} alt={variant.label} className="h-full w-full object-cover" />
                  </div>
                  <div className="mt-4 flex min-w-0 items-center justify-between gap-3">
                    <div className="min-w-0">
                      <p className="break-words text-base font-medium">{variant.label}</p>
                      <p className="break-words text-xs text-ink/50">
                        {variant.format} · {formatCount(variant.bytes)}B
                      </p>
                    </div>
                    {variant.isPrimary ? (
                      <span className="shrink-0 mono rounded-full bg-ember/10 px-3 py-1 text-[10px] uppercase tracking-[0.24em] text-ember">
                        Primary
                      </span>
                    ) : null}
                  </div>
                </button>
              );
            })}

            {!result ? (
              <div className="rounded-[28px] border border-dusk/10 bg-white/70 px-5 py-12 text-center text-sm text-ink/55 md:col-span-2 xl:col-span-3">
                {text.noResult}
              </div>
            ) : null}
          </div>
        </motion.section>

        <motion.section
          {...fadeUp}
          transition={{ duration: 0.45, delay: 0.16 }}
          className="panel p-5 md:p-6"
        >
          <ModelExplainer model={model} compact />
        </motion.section>
      </div>
    </main>
  );
}
