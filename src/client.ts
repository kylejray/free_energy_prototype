const API_BASE = '/api';

export interface HealthPayload {
  message: string;
  uptime_seconds: number;
}

export interface SampleRequest {
  mean: number;
  variance: number;
  size: number;
  bins?: number;
}

export interface HistogramPayload {
  edges: number[];
  counts: number[];
}

export interface SampleStats {
  sample_mean: number;
  sample_variance: number;
  requested_mean: number;
  requested_variance: number;
}

export interface SampleResponse {
  samples: number[];
  histogram: HistogramPayload;
  stats: SampleStats;
}

export interface NotebookRequestPayload {
  xp: number[];
  fp: number[];
  ll: number;
  ul: number;
  section?: 'sampling' | 'free_energy' | 'standard' | 'all';
  sampleSize?: number;
  trials?: number;
}

interface NotebookResponseRaw {
  sampling_top_plot?: string | null;
  sampling_bottom_plot?: string | null;
  free_energy_top_plot?: string | null;
  free_energy_bottom_plot?: string | null;
  free_energy_standard_plot?: string | null;
  metadata?: Record<string, number>;
}

export interface NotebookResponse {
  samplingTopPlot?: string | null;
  samplingBottomPlot?: string | null;
  freeEnergyTopPlot?: string | null;
  freeEnergyBottomPlot?: string | null;
  freeEnergyStandardPlot?: string | null;
  metadata?: Record<string, number>;
}

class HttpError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const normalizedPath = path.startsWith('/') ? path : `/${path}`;
  const headers = {
    'Content-Type': 'application/json',
    ...(options.headers ?? {})
  };

  const response = await fetch(`${API_BASE}${normalizedPath}`, {
    ...options,
    headers
  });

  if (!response.ok) {
    let message = 'Request failed';
    try {
      const problem = await response.json();
      if (typeof problem === 'object' && problem !== null) {
        if (typeof (problem as { detail?: unknown }).detail === 'string') {
          message = (problem as { detail: string }).detail;
        } else {
          message = JSON.stringify(problem);
        }
      }
    } catch {
      const fallback = await response.text();
      if (fallback) {
        message = fallback;
      }
    }

    throw new HttpError(response.status, message);
  }

  try {
    return (await response.json()) as T;
  } catch (error) {
    throw new Error('Invalid JSON response from server');
  }
}

export async function fetchHealth(): Promise<HealthPayload> {
  return request<HealthPayload>('/');
}

export async function generateSamples(payload: SampleRequest): Promise<SampleResponse> {
  return request<SampleResponse>('/sample', {
    method: 'POST',
    body: JSON.stringify(payload)
  });
}

export async function runNotebookAnalysis(
  payload: NotebookRequestPayload
): Promise<NotebookResponse> {
  const bodyPayload: Record<string, unknown> = {
    xp: payload.xp,
    fp: payload.fp,
    ll: payload.ll,
    ul: payload.ul,
    section: payload.section,
    sample_size: payload.sampleSize,
    trials: payload.trials
  };

  if (bodyPayload.sample_size === undefined) {
    delete bodyPayload.sample_size;
  }
  if (bodyPayload.trials === undefined) {
    delete bodyPayload.trials;
  }

  const response = await request<NotebookResponseRaw>('/analysis', {
    method: 'POST',
    body: JSON.stringify(bodyPayload)
  });

  return {
    samplingTopPlot: response.sampling_top_plot,
    samplingBottomPlot: response.sampling_bottom_plot,
    freeEnergyTopPlot: response.free_energy_top_plot,
    freeEnergyBottomPlot: response.free_energy_bottom_plot,
    freeEnergyStandardPlot: response.free_energy_standard_plot,
    metadata: response.metadata
  };
}
