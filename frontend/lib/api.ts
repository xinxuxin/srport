export type Resolution = {
  width: number;
  height: number;
};

export type CheckpointOption = {
  name: string;
  scale: number;
  checkpoint_path: string | null;
  weights_source: string;
  model_version: string;
};

export type DeploymentInfo = {
  model_version: string;
  checkpoint_source: string;
  build_time: string;
  git_commit: string;
  device_target: string;
  api_version: string;
};

export type ModelInfo = {
  name: string;
  paper_title: string;
  checkpoint_loaded: boolean;
  checkpoint_path: string | null;
  weights_source: string;
  upscale: number;
  parameter_count: number;
  estimated_macs: number;
  estimated_flops: number;
  reference_latency_ms: number;
  architecture: Record<string, unknown>;
  deployment: DeploymentInfo;
  available_checkpoints: CheckpointOption[];
  supported_methods: string[];
  supported_scales: number[];
};

export type RuntimeInfo = {
  latency_ms: number;
  parameter_count: number;
  estimated_macs: number;
  estimated_flops: number;
};

export type ImageInfo = {
  resolution: Resolution;
  format: string;
  bytes: number;
};

export type PipelineStage = {
  key: string;
  label: string;
  description: string;
  duration_ms: number;
};

export type InferenceOptions = {
  session_id: string;
  method: string;
  checkpoint_name: string;
  scale: number;
  output_format: string;
  tile_size: number;
};

export type ComparisonOutput = {
  method: string;
  label: string;
  image_url: string;
  format: string;
  bytes: number;
};

export type InferenceResponse = {
  request_id: string;
  created_at: string;
  output_image_url: string;
  model: ModelInfo;
  runtime: RuntimeInfo;
  input: ImageInfo;
  output: ImageInfo;
  pipeline: {
    total_duration_ms: number;
    stages: PipelineStage[];
  };
  options: InferenceOptions;
  input_image_url: string;
  comparisons: ComparisonOutput[];
};

export type AnalyticsPoint = {
  label: string;
  value: number;
};

export type EventPreview = {
  request_id: string;
  created_at: string;
  session_id: string;
  input_resolution: Resolution;
  output_resolution: Resolution;
  latency_ms: number;
  upscale: number;
  method: string;
  checkpoint_name: string;
  output_format: string;
  input_image_url: string;
  output_image_url: string;
};

export type AnalyticsSummary = {
  total_requests: number;
  session_count: number;
  average_latency_ms: number;
  p50_latency_ms: number;
  p95_latency_ms: number;
  latest_request_at: string | null;
  average_output_megapixels: number;
  total_processed_pixels: number;
  latency_series: AnalyticsPoint[];
  upscale_distribution: AnalyticsPoint[];
  recent_events: EventPreview[];
};

export type HistoryEvent = {
  request_id: string;
  created_at: string;
  session_id: string;
  method: string;
  checkpoint_name: string;
  output_format: string;
  tile_size: number;
  input_resolution: Resolution;
  output_resolution: Resolution;
  latency_ms: number;
  upscale: number;
  parameter_count: number;
  estimated_macs: number;
  estimated_flops: number;
  input_image_url: string;
  output_image_url: string;
};

export type BatchAggregate = {
  total_files: number;
  completed_files: number;
  average_latency_ms: number;
  p50_latency_ms: number;
  p95_latency_ms: number;
  total_output_megapixels: number;
};

export type BatchInferenceResponse = {
  session_id: string;
  results: InferenceResponse[];
  aggregate: BatchAggregate;
};

export type InferenceRequestOptions = {
  sessionId: string;
  method: string;
  scale: number;
  outputFormat: string;
  tileSize: number;
  checkpointName?: string;
};

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000/api/v1";

export function resolveApiAssetUrl(assetPath: string): string {
  if (assetPath.startsWith("http://") || assetPath.startsWith("https://")) {
    return assetPath;
  }
  const origin = new URL(API_BASE_URL).origin;
  return new URL(assetPath, origin).toString();
}

async function parseResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const body = await response.text();
    let detail: string | null = null;
    try {
      const parsed = JSON.parse(body) as { detail?: string };
      detail = parsed.detail ?? null;
    } catch {
      detail = null;
    }
    throw new Error(detail ?? (body || `Request failed with ${response.status}`));
  }

  return (await response.json()) as T;
}

export async function fetchModelInfo(checkpointName?: string): Promise<ModelInfo> {
  const url = new URL(`${API_BASE_URL}/model/info`);
  if (checkpointName) {
    url.searchParams.set("checkpoint_name", checkpointName);
  }
  const response = await fetch(url.toString(), { cache: "no-store" });
  return parseResponse<ModelInfo>(response);
}

export async function fetchAnalyticsSummary(): Promise<AnalyticsSummary> {
  const response = await fetch(`${API_BASE_URL}/usage/summary`, { cache: "no-store" });
  return parseResponse<AnalyticsSummary>(response);
}

export async function fetchHistoryEvent(requestId: string): Promise<HistoryEvent> {
  const response = await fetch(`${API_BASE_URL}/history/${requestId}`, { cache: "no-store" });
  return parseResponse<HistoryEvent>(response);
}

function appendInferenceOptions(formData: FormData, options: InferenceRequestOptions) {
  formData.append("session_id", options.sessionId);
  formData.append("method", options.method);
  formData.append("scale", String(options.scale));
  formData.append("output_format", options.outputFormat);
  formData.append("tile_size", String(options.tileSize));
  if (options.checkpointName) {
    formData.append("checkpoint_name", options.checkpointName);
  }
}

export async function superResolve(
  file: File,
  options: InferenceRequestOptions
): Promise<InferenceResponse> {
  const formData = new FormData();
  formData.append("file", file);
  appendInferenceOptions(formData, options);
  const response = await fetch(`${API_BASE_URL}/infer`, {
    method: "POST",
    body: formData
  });
  return parseResponse<InferenceResponse>(response);
}

export async function batchSuperResolve(
  files: File[],
  options: InferenceRequestOptions
): Promise<BatchInferenceResponse> {
  const formData = new FormData();
  files.forEach((file) => {
    formData.append("files", file);
  });
  appendInferenceOptions(formData, options);
  const response = await fetch(`${API_BASE_URL}/infer/batch`, {
    method: "POST",
    body: formData
  });
  return parseResponse<BatchInferenceResponse>(response);
}
