export type Resolution = {
  width: number;
  height: number;
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

export type InferenceResponse = {
  request_id: string;
  created_at: string;
  output_image_base64: string;
  model: ModelInfo;
  runtime: RuntimeInfo;
  input: ImageInfo;
  output: ImageInfo;
};

export type AnalyticsPoint = {
  label: string;
  value: number;
};

export type EventPreview = {
  request_id: string;
  created_at: string;
  input_resolution: Resolution;
  output_resolution: Resolution;
  latency_ms: number;
  upscale: number;
};

export type AnalyticsSummary = {
  total_requests: number;
  average_latency_ms: number;
  latest_request_at: string | null;
  average_output_megapixels: number;
  total_processed_pixels: number;
  latency_series: AnalyticsPoint[];
  upscale_distribution: AnalyticsPoint[];
  recent_events: EventPreview[];
};

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000/api/v1";

async function parseResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const body = await response.text();
    throw new Error(body || `Request failed with ${response.status}`);
  }

  return (await response.json()) as T;
}

export async function fetchModelInfo(): Promise<ModelInfo> {
  const response = await fetch(`${API_BASE_URL}/model/info`, { cache: "no-store" });
  return parseResponse<ModelInfo>(response);
}

export async function fetchAnalyticsSummary(): Promise<AnalyticsSummary> {
  const response = await fetch(`${API_BASE_URL}/analytics/summary`, { cache: "no-store" });
  return parseResponse<AnalyticsSummary>(response);
}

export async function superResolve(file: File): Promise<InferenceResponse> {
  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch(`${API_BASE_URL}/super-resolve`, {
    method: "POST",
    body: formData
  });
  return parseResponse<InferenceResponse>(response);
}
