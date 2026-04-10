"use client";

import { motion, useInView } from "framer-motion";
import { CSSProperties, useEffect, useMemo, useRef, useState } from "react";

type ViewMode = "product" | "technical";

type InteractiveKey =
  | "input"
  | "stem"
  | "pfem1"
  | "pfem2"
  | "pfem3"
  | "pfem4"
  | "concat"
  | "fusion"
  | "pixel"
  | "output"
  | "espm"
  | "dcab"
  | "pfemInset"
  | "legend";

type RegionFrame = {
  x: number;
  y: number;
  w: number;
  h: number;
  delay: number;
};

type DetailContent = {
  productTitle: string;
  technicalTitle: string;
  body: string;
  accent: string;
};

const VIEWBOX = { width: 1360, height: 900 };

const regionFrames: Record<InteractiveKey, RegionFrame> = {
  espm: { x: 34, y: 46, w: 456, h: 216, delay: 0.12 },
  dcab: { x: 968, y: 46, w: 344, h: 214, delay: 0.2 },
  input: { x: 28, y: 312, w: 126, h: 138, delay: 0.28 },
  stem: { x: 176, y: 344, w: 126, h: 82, delay: 0.33 },
  pfem1: { x: 328, y: 336, w: 130, h: 104, delay: 0.38 },
  pfem2: { x: 472, y: 336, w: 130, h: 104, delay: 0.43 },
  pfem3: { x: 616, y: 336, w: 130, h: 104, delay: 0.48 },
  pfem4: { x: 760, y: 336, w: 130, h: 104, delay: 0.53 },
  concat: { x: 914, y: 346, w: 104, h: 82, delay: 0.58 },
  fusion: { x: 1040, y: 336, w: 126, h: 104, delay: 0.63 },
  pixel: { x: 1188, y: 336, w: 124, h: 104, delay: 0.68 },
  output: { x: 1218, y: 306, w: 118, h: 142, delay: 0.73 },
  pfemInset: { x: 70, y: 560, w: 548, h: 246, delay: 0.24 },
  legend: { x: 886, y: 572, w: 426, h: 222, delay: 0.28 }
};

const connectorPaths = [
  { key: "main-1", d: "M 154 384 C 168 384, 174 384, 176 384" },
  { key: "main-2", d: "M 302 386 C 314 386, 320 388, 328 388" },
  { key: "main-3", d: "M 458 388 C 464 388, 468 388, 472 388" },
  { key: "main-4", d: "M 602 388 C 608 388, 612 388, 616 388" },
  { key: "main-5", d: "M 746 388 C 752 388, 756 388, 760 388" },
  { key: "main-6", d: "M 890 388 C 900 388, 908 388, 914 388" },
  { key: "main-7", d: "M 1018 388 C 1026 388, 1032 388, 1040 388" },
  { key: "main-8", d: "M 1166 388 C 1174 388, 1180 388, 1188 388" },
  { key: "main-9", d: "M 1312 388 C 1320 388, 1328 388, 1336 388" },
  { key: "espm-up", d: "M 238 344 C 238 292, 228 256, 228 244" },
  { key: "espm-forward", d: "M 490 154 C 650 154, 814 154, 968 154" },
  { key: "espm-down", d: "M 378 262 C 378 304, 478 328, 1104 328" },
  { key: "fusion-join", d: "M 1104 328 C 1104 336, 1104 340, 1104 344" },
  { key: "dcab-callout", d: "M 490 160 C 676 124, 834 116, 968 136" },
  { key: "pfem-callout", d: "M 538 440 C 512 506, 436 542, 332 560" }
];

const pulseWaypoints: Record<InteractiveKey, { x: number; y: number }> = {
  input: { x: 154, y: 384 },
  stem: { x: 238, y: 386 },
  pfem1: { x: 392, y: 388 },
  pfem2: { x: 536, y: 388 },
  pfem3: { x: 680, y: 388 },
  pfem4: { x: 824, y: 388 },
  concat: { x: 966, y: 388 },
  espm: { x: 238, y: 154 },
  fusion: { x: 1104, y: 388 },
  pixel: { x: 1250, y: 388 },
  output: { x: 1336, y: 388 },
  dcab: { x: 1138, y: 154 },
  pfemInset: { x: 344, y: 684 },
  legend: { x: 1098, y: 682 }
};

const flowSequence: InteractiveKey[] = [
  "input",
  "stem",
  "pfem1",
  "pfem2",
  "pfem3",
  "pfem4",
  "concat",
  "espm",
  "fusion",
  "pixel",
  "output"
];

const detailMap: Record<InteractiveKey, DetailContent> = {
  input: {
    productTitle: "Input image",
    technicalTitle: "Low-resolution input",
    body: "The architecture starts from a low-resolution image and immediately projects it into a learned feature space.",
    accent: "from-dusk/10 to-dusk/5"
  },
  stem: {
    productTitle: "Feature stem",
    technicalTitle: "Shallow Conv 3x3",
    body: "A lightweight 3x3 convolution creates the base feature tensor that feeds both the PFEM pipeline and the ESPM context branch.",
    accent: "from-sky-100/70 to-white"
  },
  pfem1: {
    productTitle: "Detail enhancement stage",
    technicalTitle: "PFEM-1",
    body: "PFEM progressively enriches image features across multiple stages before fusion and reconstruction.",
    accent: "from-amber-100/70 to-white"
  },
  pfem2: {
    productTitle: "Detail enhancement stage",
    technicalTitle: "PFEM-2",
    body: "Each PFEM stage blends local feature extraction, transformer-based context modeling, and spatial attention.",
    accent: "from-amber-100/70 to-white"
  },
  pfem3: {
    productTitle: "Detail enhancement stage",
    technicalTitle: "PFEM-3",
    body: "The middle PFEM stages deepen contextual refinement while preserving the same repeatable submodule structure from the paper diagram.",
    accent: "from-amber-100/70 to-white"
  },
  pfem4: {
    productTitle: "Detail enhancement stage",
    technicalTitle: "PFEM-4",
    body: "The final PFEM stage completes the panoramic feature pathway before concatenation and fusion.",
    accent: "from-amber-100/70 to-white"
  },
  concat: {
    productTitle: "Multi-stage feature fusion",
    technicalTitle: "Concat",
    body: "Outputs from the PFEM stages are gathered and aligned before the final reconstruction head.",
    accent: "from-cyan-100/70 to-white"
  },
  fusion: {
    productTitle: "Reconstruction head",
    technicalTitle: "Conv 3x3 fusion",
    body: "The PFEM stream merges with the ESPM branch and projects into the representation used for upsampling.",
    accent: "from-rose-100/70 to-white"
  },
  pixel: {
    productTitle: "Resolution lift",
    technicalTitle: "Pixel Shuffle",
    body: "Pixel Shuffle converts the fused representation into a larger spatial grid with low overhead.",
    accent: "from-emerald-100/70 to-white"
  },
  output: {
    productTitle: "Super-resolved output",
    technicalTitle: "High-resolution output",
    body: "The final fused representation is converted into the super-resolved output image.",
    accent: "from-moss/20 to-white"
  },
  espm: {
    productTitle: "Pyramid context branch",
    technicalTitle: "ESPM",
    body: "This branch emphasizes structured multi-scale context and edge-aware refinement.",
    accent: "from-violet-100/80 to-white"
  },
  dcab: {
    productTitle: "Channel split and recombine unit",
    technicalTitle: "DCAB inset",
    body: "This unit dynamically separates and recombines channels to preserve important visual signals efficiently.",
    accent: "from-fuchsia-100/75 to-white"
  },
  pfemInset: {
    productTitle: "PFEM submodule",
    technicalTitle: "LFEB + Transformer + ESAB",
    body: "Each PFEM stage blends local feature extraction, transformer-based context modeling, and spatial attention.",
    accent: "from-orange-100/75 to-white"
  },
  legend: {
    productTitle: "Abbreviation guide",
    technicalTitle: "Legend",
    body: "The legend keeps the technical shorthand readable during a live demo without forcing the audience to decode paper terminology in real time.",
    accent: "from-slate-100 to-white"
  }
};

const legendItems = [
  { short: "PFEM", long: "Panoramic feature extraction module" },
  { short: "ESPM", long: "Efficient spatial pyramid module" },
  { short: "LFEB", long: "Local feature extraction block" },
  { short: "ESAB", long: "Enhanced spatial attention block" },
  { short: "CA", long: "Channel attention" },
  { short: "AF", long: "Adaptive fusion" }
];

function toPercent(value: number, total: number): string {
  return `${(value / total) * 100}%`;
}

function frameStyle(frame: RegionFrame): CSSProperties {
  return {
    left: toPercent(frame.x, VIEWBOX.width),
    top: toPercent(frame.y, VIEWBOX.height),
    width: toPercent(frame.w, VIEWBOX.width),
    height: toPercent(frame.h, VIEWBOX.height)
  };
}

function regionTitle(key: InteractiveKey, viewMode: ViewMode): string {
  const productTitles: Record<InteractiveKey, string> = {
    input: "Input image",
    stem: "Feature stem",
    pfem1: "Detail stage 1",
    pfem2: "Detail stage 2",
    pfem3: "Detail stage 3",
    pfem4: "Detail stage 4",
    concat: "Stage merge",
    fusion: "Fusion conv",
    pixel: "Resolution lift",
    output: "Output image",
    espm: "Pyramid context branch",
    dcab: "Split and attend inset",
    pfemInset: "PFEM submodule",
    legend: "Legend"
  };
  const technicalTitles: Record<InteractiveKey, string> = {
    input: "Input",
    stem: "Conv 3x3",
    pfem1: "PFEM-1",
    pfem2: "PFEM-2",
    pfem3: "PFEM-3",
    pfem4: "PFEM-4",
    concat: "Concat",
    fusion: "Conv 3x3",
    pixel: "Pixel Shuffle",
    output: "Output",
    espm: "ESPM",
    dcab: "DCAB",
    pfemInset: "PFEM inset",
    legend: "Legend"
  };
  return viewMode === "product" ? productTitles[key] : technicalTitles[key];
}

function regionSubtitle(key: InteractiveKey, viewMode: ViewMode): string {
  const productSubtitles: Partial<Record<InteractiveKey, string>> = {
    stem: "base projection",
    concat: "four-stage gather",
    fusion: "multi-branch fusion",
    pixel: "efficient upsampling",
    espm: "structured multiscale refinement",
    dcab: "split -> attend -> fuse",
    pfemInset: "local + global + spatial attention"
  };
  const technicalSubtitles: Partial<Record<InteractiveKey, string>> = {
    stem: "shallow feature extraction",
    concat: "PFEM outputs",
    fusion: "fusion projection",
    pixel: "sub-pixel rearrangement",
    espm: "DCAB + AF flow",
    dcab: "split / conv / CA / concat / conv",
    pfemInset: "LFEB / Transformer / ESAB"
  };
  return (viewMode === "product" ? productSubtitles[key] : technicalSubtitles[key]) ?? "";
}

function regionToneLabel(key: InteractiveKey, viewMode: ViewMode): string {
  if (!key.startsWith("pfem") || key === "pfemInset") {
    return "";
  }
  if (viewMode === "product") {
    return "Detail branch";
  }
  return "PFEM stage";
}

function getActiveDetailKey(key: InteractiveKey): InteractiveKey {
  if (key.startsWith("pfem")) {
    return key === "pfemInset" ? "pfemInset" : key;
  }
  return key;
}

function isHighlighted(
  key: InteractiveKey,
  focused: InteractiveKey | null,
  flowKey: InteractiveKey | null
): boolean {
  return key === focused || key === flowKey;
}

function DiagramBox({
  regionKey,
  viewMode,
  focused,
  flowKey,
  isInView,
  onFocus,
  onHover
}: {
  regionKey: InteractiveKey;
  viewMode: ViewMode;
  focused: InteractiveKey | null;
  flowKey: InteractiveKey | null;
  isInView: boolean;
  onFocus: (key: InteractiveKey) => void;
  onHover: (key: InteractiveKey | null) => void;
}) {
  const frame = regionFrames[regionKey];
  const highlighted = isHighlighted(regionKey, focused, flowKey);
  const dimmed = focused !== null && !highlighted;

  const baseClasses =
    "absolute overflow-hidden rounded-[26px] border text-left shadow-[0_20px_60px_rgba(19,32,47,0.10)] backdrop-blur-sm transition duration-300 focus:outline-none focus:ring-2 focus:ring-dusk/45";

  const panelClasses: Record<InteractiveKey, string> = {
    input: "border-dusk/12 bg-gradient-to-br from-white via-slate-50 to-slate-100",
    stem: "border-sky-200/70 bg-gradient-to-br from-sky-50 via-white to-sky-100/70",
    pfem1: "border-amber-200/80 bg-gradient-to-br from-amber-50 via-white to-amber-100/70",
    pfem2: "border-amber-200/80 bg-gradient-to-br from-amber-50 via-white to-amber-100/70",
    pfem3: "border-amber-200/80 bg-gradient-to-br from-amber-50 via-white to-amber-100/70",
    pfem4: "border-amber-200/80 bg-gradient-to-br from-amber-50 via-white to-amber-100/70",
    concat: "border-cyan-200/80 bg-gradient-to-br from-cyan-50 via-white to-cyan-100/70",
    fusion: "border-rose-200/70 bg-gradient-to-br from-rose-50 via-white to-rose-100/70",
    pixel: "border-emerald-200/70 bg-gradient-to-br from-emerald-50 via-white to-emerald-100/70",
    output: "border-moss/25 bg-gradient-to-br from-white via-moss/5 to-moss/15",
    espm: "border-violet-200/80 bg-gradient-to-br from-violet-100/90 via-fuchsia-50 to-white",
    dcab: "border-fuchsia-200/80 bg-gradient-to-br from-fuchsia-50 via-white to-purple-50",
    pfemInset: "border-orange-200/80 bg-gradient-to-br from-orange-50 via-white to-amber-50",
    legend: "border-slate-200/80 bg-gradient-to-br from-slate-50 via-white to-white"
  };

  return (
    <motion.button
      type="button"
      initial={{ opacity: 0, y: 22 }}
      animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 22 }}
      transition={{ duration: 0.45, delay: frame.delay }}
      style={frameStyle(frame)}
      onFocus={() => onFocus(regionKey)}
      onMouseEnter={() => onHover(regionKey)}
      onMouseLeave={() => onHover(null)}
      onClick={() => onFocus(regionKey)}
      className={`${baseClasses} ${panelClasses[regionKey]} ${
        highlighted ? "scale-[1.01] shadow-[0_24px_85px_rgba(34,65,93,0.16)]" : ""
      } ${dimmed ? "opacity-45" : "opacity-100"}`}
      aria-pressed={focused === regionKey}
    >
      {regionKey === "input" || regionKey === "output" ? (
        <div className="relative flex h-full flex-col justify-between p-3">
          <div className="mono text-[10px] uppercase tracking-[0.18em] text-ink/48">
            {regionTitle(regionKey, viewMode)}
          </div>
          <div className="mt-3 h-full rounded-[20px] border border-white/70 bg-[radial-gradient(circle_at_top_left,rgba(34,65,93,0.16),transparent_36%),linear-gradient(180deg,#ffffff_0%,#edf2f5_100%)] p-3">
            <div className="grid h-full place-items-center rounded-[16px] border border-dusk/8 bg-white/80">
              <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-br from-dusk to-ember text-sm font-semibold text-white">
                {regionKey === "input" ? "LR" : "HR"}
              </div>
            </div>
          </div>
        </div>
      ) : null}

      {regionKey === "stem" || regionKey === "concat" || regionKey === "fusion" || regionKey === "pixel" ? (
        <div className="flex h-full flex-col justify-center px-3.5 text-center">
          <p className="text-[13px] font-semibold leading-tight text-ink">{regionTitle(regionKey, viewMode)}</p>
          {regionSubtitle(regionKey, viewMode) ? (
            <p className="mt-1 text-[10px] uppercase tracking-[0.14em] leading-tight text-ink/48">
              {regionSubtitle(regionKey, viewMode)}
            </p>
          ) : null}
        </div>
      ) : null}

      {regionKey.startsWith("pfem") && regionKey !== "pfemInset" ? (
        <div className="flex h-full flex-col justify-center px-3.5 text-center">
          <div className="mono text-[9px] uppercase tracking-[0.16em] leading-tight text-amber-700/70">
            {regionToneLabel(regionKey, viewMode)}
          </div>
          <p className="mt-2 text-[15px] font-semibold leading-snug text-ink">{regionTitle(regionKey, viewMode)}</p>
          <p className="mt-1 text-[10px] uppercase tracking-[0.08em] leading-snug text-ink/46">
            {viewMode === "product" ? "repeatable stage" : "LFEB + context + ESAB"}
          </p>
        </div>
      ) : null}

      {regionKey === "espm" ? (
        <div className="flex h-full flex-col p-4">
          <div className="flex items-start justify-between gap-3">
            <div>
              <p className="mono text-[10px] uppercase tracking-[0.2em] text-violet-700/70">
                Upper branch
              </p>
              <h3 className="mt-2 text-[17px] font-semibold leading-tight text-ink">{regionTitle(regionKey, viewMode)}</h3>
              <p className="mt-2 text-[11px] leading-5 text-ink/62">
                {viewMode === "product"
                  ? "Structured multi-scale context and edge-aware refinement."
                  : "Split features into a lightweight DCAB path and adaptive fusion output."}
              </p>
            </div>
            <div className="rounded-full bg-white/75 px-3 py-1 text-[10px] uppercase tracking-[0.18em] text-violet-700">
              {viewMode === "product" ? "context" : "ESPM"}
            </div>
          </div>

          <div className="mt-4 grid flex-1 gap-3 md:grid-cols-[0.95fr_1.2fr_0.9fr]">
            <div className="rounded-[18px] border border-white/60 bg-white/75 p-3 text-center">
              <p className="text-[13px] font-semibold leading-tight">{viewMode === "product" ? "Split channels" : "Split"}</p>
              <p className="mt-2 text-[11px] leading-4 text-ink/55">
                {viewMode === "product" ? "Separate structured signals" : "channel partition"}
              </p>
            </div>
            <div className="rounded-[18px] border border-white/60 bg-white/78 p-3">
              <div className="grid gap-2">
                <div className="rounded-[14px] bg-violet-50 px-3 py-2 text-left text-[13px] font-semibold leading-tight text-ink">
                  {viewMode === "product" ? "Edge-aware refinement cell" : "DCAB"}
                </div>
                <div className="grid grid-cols-2 gap-2 text-[10px] uppercase tracking-[0.12em] text-ink/55">
                  <div className="rounded-[12px] bg-white px-3 py-2 text-center">Conv</div>
                  <div className="rounded-[12px] bg-white px-3 py-2 text-center">CA</div>
                </div>
              </div>
            </div>
            <div className="rounded-[18px] border border-white/60 bg-white/75 p-3 text-center">
              <p className="text-[13px] font-semibold leading-tight">{viewMode === "product" ? "Adaptive merge" : "AF"}</p>
              <p className="mt-2 text-[11px] leading-4 text-ink/55">
                {viewMode === "product" ? "Feed fusion-ready context" : "adaptive fusion"}
              </p>
            </div>
          </div>
        </div>
      ) : null}

      {regionKey === "dcab" ? (
        <div className="flex h-full flex-col p-4">
          <div className="flex items-start justify-between gap-3">
            <div>
              <p className="mono text-[10px] uppercase tracking-[0.2em] text-fuchsia-700/70">
                Zoomed inset
              </p>
              <h3 className="mt-2 text-[17px] font-semibold leading-tight text-ink">{regionTitle(regionKey, viewMode)}</h3>
            </div>
            <div className="rounded-full bg-white/80 px-3 py-1 text-[10px] uppercase tracking-[0.16em] text-fuchsia-700">
              {viewMode === "product" ? "efficient channel flow" : "split -> fuse"}
            </div>
          </div>
          <div className="mt-4 flex flex-1 items-center gap-1.5">
            {[
              viewMode === "product" ? "Split" : "Split",
              "Conv",
              viewMode === "product" ? "Attention" : "CA",
              "Concat",
              "Conv"
            ].map((label, index) => (
              <div key={label} className="flex min-w-0 flex-1 items-center gap-1.5">
                <div className="flex-1 rounded-[16px] border border-fuchsia-100 bg-white/80 px-2 py-3 text-center text-[11px] font-semibold leading-tight text-ink">
                  {label}
                </div>
                {index < 4 ? <div className="h-0.5 w-2.5 shrink-0 bg-fuchsia-300" /> : null}
              </div>
            ))}
          </div>
        </div>
      ) : null}

      {regionKey === "pfemInset" ? (
        <div className="flex h-full flex-col p-4">
          <div className="flex items-start justify-between gap-3">
            <div>
              <p className="mono text-[10px] uppercase tracking-[0.2em] text-orange-700/70">
                Submodule view
              </p>
              <h3 className="mt-2 text-[17px] font-semibold leading-tight text-ink">{regionTitle(regionKey, viewMode)}</h3>
              <p className="mt-2 text-[11px] leading-5 text-ink/62">
                {viewMode === "product"
                  ? "Local textures, global context, and spatial attention are blended inside every PFEM stage."
                  : "PFEM repeats LFEB, a transformer-style context block, and ESAB."}
              </p>
            </div>
            <div className="rounded-full bg-white/75 px-3 py-1 text-[10px] uppercase tracking-[0.16em] text-orange-700">
              {viewMode === "product" ? "detail cell" : "PFEM"}
            </div>
          </div>

          <div className="mt-4 grid flex-1 gap-3 md:grid-cols-3">
            {[
              viewMode === "product" ? "Local detail" : "LFEB",
              viewMode === "product" ? "Global context" : "Transformer",
              viewMode === "product" ? "Spatial emphasis" : "ESAB"
            ].map((label) => (
              <div
                key={label}
                className="rounded-[18px] border border-orange-100 bg-white/80 px-4 py-5 text-center"
              >
                <p className="text-[13px] font-semibold leading-tight text-ink">{label}</p>
              </div>
            ))}
          </div>
        </div>
      ) : null}

      {regionKey === "legend" ? (
        <div className="flex h-full flex-col p-4">
          <div className="flex items-start justify-between gap-3">
            <div>
              <p className="mono text-[10px] uppercase tracking-[0.2em] text-slate-500">
                Reference guide
              </p>
              <h3 className="mt-2 text-[17px] font-semibold leading-tight text-ink">{regionTitle(regionKey, viewMode)}</h3>
            </div>
            <div className="rounded-full bg-white/80 px-3 py-1 text-[10px] uppercase tracking-[0.16em] text-slate-600">
              abbreviations
            </div>
          </div>
          <div className="mt-4 grid gap-2 sm:grid-cols-2">
            {legendItems.map((item) => (
              <div key={item.short} className="rounded-[16px] border border-slate-200 bg-white/80 px-3 py-3">
                <p className="mono text-[10px] uppercase tracking-[0.16em] text-slate-500">{item.short}</p>
                <p className="mt-1 text-[13px] leading-5 text-ink/72">{item.long}</p>
              </div>
            ))}
          </div>
        </div>
      ) : null}
    </motion.button>
  );
}

export function EPNetArchitectureDiagram() {
  const [viewMode, setViewMode] = useState<ViewMode>("product");
  const [selectedRegion, setSelectedRegion] = useState<InteractiveKey>("espm");
  const [hoveredRegion, setHoveredRegion] = useState<InteractiveKey | null>(null);
  const [playFlow, setPlayFlow] = useState(false);
  const [flowIndex, setFlowIndex] = useState(0);
  const ref = useRef<HTMLDivElement | null>(null);
  const isInView = useInView(ref, { once: true, amount: 0.22 });

  useEffect(() => {
    if (!playFlow) {
      return;
    }
    const interval = window.setInterval(() => {
      setFlowIndex((current) => (current + 1) % flowSequence.length);
    }, 900);
    return () => window.clearInterval(interval);
  }, [playFlow]);

  const flowKey = playFlow ? flowSequence[flowIndex] : null;
  const focusedKey = hoveredRegion ?? flowKey ?? selectedRegion;
  const detailContent = useMemo(
    () => detailMap[getActiveDetailKey(focusedKey)],
    [focusedKey]
  );
  const pulsePoint = flowKey ? pulseWaypoints[flowKey] : null;

  return (
    <section className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div className="max-w-4xl">
          <p className="mono text-xs uppercase tracking-[0.28em] text-dusk/60">Model explainer</p>
          <h2 className="mt-2 text-3xl font-semibold">EPNet architecture map</h2>
          <p className="mt-3 text-sm leading-6 text-ink/68">
            EPNet combines a detail-focused enhancement pathway, a lightweight pyramid context
            pathway, and a reconstruction stage to turn low-resolution images into sharper
            high-resolution outputs.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <div className="rounded-full border border-dusk/10 bg-white/80 p-1">
            {(["product", "technical"] as const).map((mode) => (
              <button
                key={mode}
                type="button"
                onClick={() => setViewMode(mode)}
                className={`rounded-full px-4 py-2 text-sm transition ${
                  viewMode === mode
                    ? "bg-dusk text-white shadow-panel"
                    : "text-ink/65 hover:bg-sand"
                }`}
                aria-pressed={viewMode === mode}
              >
                {mode === "product" ? "Product view" : "Technical view"}
              </button>
            ))}
          </div>

          <button
            type="button"
            onClick={() => setPlayFlow((value) => !value)}
            className={`mono rounded-full px-4 py-2 text-xs uppercase tracking-[0.24em] transition ${
              playFlow
                ? "bg-ember text-white shadow-panel"
                : "border border-dusk/15 bg-white/80 text-ink/70"
            }`}
            aria-pressed={playFlow}
          >
            {playFlow ? "Pause flow" : "Play flow"}
          </button>
        </div>
      </div>

      <div
        ref={ref}
        className="rounded-[34px] border border-white/55 bg-white/78 p-4 shadow-panel backdrop-blur sm:p-5"
      >
        <div className="overflow-x-auto pb-2">
          <div
            className="relative min-w-[1240px] rounded-[28px] border border-dusk/8 bg-[radial-gradient(circle_at_top_left,rgba(98,80,187,0.08),transparent_28%),radial-gradient(circle_at_bottom_left,rgba(194,84,47,0.07),transparent_24%),linear-gradient(180deg,rgba(255,255,255,0.95)_0%,rgba(247,242,236,0.9)_100%)]"
            style={{ aspectRatio: `${VIEWBOX.width} / ${VIEWBOX.height}` }}
          >
            <svg
              viewBox={`0 0 ${VIEWBOX.width} ${VIEWBOX.height}`}
              className="pointer-events-none absolute inset-0 h-full w-full"
              aria-hidden="true"
            >
              <defs>
                <linearGradient id="flowStroke" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="#c2542f" />
                  <stop offset="100%" stopColor="#22415d" />
                </linearGradient>
                <marker
                  id="arrowHead"
                  markerWidth="8"
                  markerHeight="8"
                  refX="6"
                  refY="3"
                  orient="auto"
                >
                  <path d="M0,0 L6,3 L0,6 Z" fill="#22415d" opacity="0.8" />
                </marker>
              </defs>

              {connectorPaths.map((path, index) => {
                const dashed = path.key.includes("callout");
                const highlighted = flowKey !== null && path.key.startsWith("main");
                return (
                  <motion.path
                    key={path.key}
                    d={path.d}
                    fill="none"
                    stroke={highlighted ? "url(#flowStroke)" : "#6f8190"}
                    strokeWidth={highlighted ? 3.5 : 2.2}
                    strokeDasharray={dashed ? "7 8" : undefined}
                    markerEnd="url(#arrowHead)"
                    initial={{ pathLength: 0, opacity: 0 }}
                    animate={isInView ? { pathLength: 1, opacity: 1 } : { pathLength: 0, opacity: 0 }}
                    transition={{ duration: 0.7, delay: 0.08 * index }}
                    opacity={focusedKey && !highlighted && dashed ? 0.5 : 0.85}
                  />
                );
              })}
            </svg>

            {pulsePoint ? (
              <motion.div
                className="pointer-events-none absolute h-4 w-4 rounded-full bg-gradient-to-r from-ember to-dusk shadow-[0_0_0_8px_rgba(194,84,47,0.12)]"
                animate={{
                  left: `calc(${toPercent(pulsePoint.x, VIEWBOX.width)} - 0.5rem)`,
                  top: `calc(${toPercent(pulsePoint.y, VIEWBOX.height)} - 0.5rem)`
                }}
                transition={{ duration: 0.45, ease: "easeInOut" }}
              />
            ) : null}

            {(Object.keys(regionFrames) as InteractiveKey[]).map((regionKey) => (
              <DiagramBox
                key={regionKey}
                regionKey={regionKey}
                viewMode={viewMode}
                focused={focusedKey}
                flowKey={flowKey}
                isInView={isInView}
                onFocus={setSelectedRegion}
                onHover={setHoveredRegion}
              />
            ))}
          </div>
        </div>
      </div>

      <div className="grid gap-4 xl:grid-cols-[1.15fr_0.85fr]">
        <motion.div
          initial={{ opacity: 0, y: 18 }}
          animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 18 }}
          transition={{ duration: 0.4, delay: 0.15 }}
          className={`rounded-[28px] border border-white/55 bg-gradient-to-br ${detailContent.accent} p-6 shadow-panel`}
        >
          <p className="mono text-xs uppercase tracking-[0.28em] text-dusk/60">
            {viewMode === "product" ? "Selected subsystem" : "Focused region"}
          </p>
          <h3 className="mt-3 text-2xl font-semibold">
            {viewMode === "product" ? detailContent.productTitle : detailContent.technicalTitle}
          </h3>
          <p className="mt-3 text-sm leading-6 text-ink/72">{detailContent.body}</p>
          <div className="mt-5 flex flex-wrap gap-2">
            <span className="rounded-full bg-white/75 px-3 py-2 text-xs text-ink/60">
              Hover a region to spotlight it
            </span>
            <span className="rounded-full bg-white/75 px-3 py-2 text-xs text-ink/60">
              Click to pin the explanation panel
            </span>
            <span className="rounded-full bg-white/75 px-3 py-2 text-xs text-ink/60">
              Play mode animates the end-to-end signal path
            </span>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 18 }}
          animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 18 }}
          transition={{ duration: 0.4, delay: 0.22 }}
          className="rounded-[28px] border border-white/55 bg-white/82 p-6 shadow-panel"
        >
          <p className="mono text-xs uppercase tracking-[0.28em] text-dusk/60">Why this matters</p>
          <div className="mt-4 space-y-3 text-sm leading-6 text-ink/72">
            <div className="rounded-[20px] bg-sand px-4 py-4">
              Balances image quality and efficiency by combining detail enhancement with a compact
              pyramid context branch.
            </div>
            <div className="rounded-[20px] bg-sand px-4 py-4">
              Designed for deployable image enhancement rather than a lab-only research showcase.
            </div>
            <div className="rounded-[20px] bg-sand px-4 py-4">
              Combines local detail, context, and reconstruction in one pipeline that is easy to
              explain during a live presentation.
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
