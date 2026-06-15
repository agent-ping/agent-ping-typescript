import { HttpClient } from "./client.js";
import { BoundedQueue } from "./queue.js";
import { Worker } from "./worker.js";
import { warnOnce } from "./warnings.js";
import { registerExitHandler } from "./exit.js";

export type EnvelopeKind =
  | "run_start"
  | "run_finish"
  | "run_events"
  | "heartbeat";

export interface RunStartEnvelope {
  kind: "run_start";
  body: Record<string, unknown>;
}

export interface RunFinishEnvelope {
  kind: "run_finish";
  runId: string;
  body: Record<string, unknown>;
}

export interface RunEventEnvelope {
  kind: "run_events";
  runId: string;
  event: Record<string, unknown>;
}

export interface HeartbeatEnvelope {
  kind: "heartbeat";
  body: Record<string, unknown>;
}

export type Envelope =
  | RunStartEnvelope
  | RunFinishEnvelope
  | RunEventEnvelope
  | HeartbeatEnvelope;

export interface InitOptions {
  apiKey?: string;
  baseUrl?: string;
  controlUrl?: string;
  queueSize?: number;
  flushIntervalMs?: number;
  batchSize?: number;
  fetchImpl?: typeof fetch;
}

export interface StatusSnapshot {
  queueSize: number;
  droppedCount: number;
  lastFlushTs: number | null;
  lastError: string | null;
}

export interface SdkState {
  apiKey: string;
  baseUrl: string;
  controlUrl: string;
  region: string;
  queue: BoundedQueue<Envelope>;
  worker: Worker;
  client: HttpClient;
  controlClient: HttpClient;
  flushIntervalMs: number;
  batchSize: number;
}

const KNOWN_REGIONS = new Set(["eu", "us"]);
const DEFAULT_REGION = "eu";
const DEFAULT_QUEUE_SIZE = 1000;
const DEFAULT_FLUSH_INTERVAL_MS = 2_000;
const DEFAULT_BATCH_SIZE = 50;

/**
 * Regional ingest hostname for the SDK to call (spec §4.1, §2.9). Falls
 * back to the EU host if the credential's region is unknown; the edge
 * returns 401 region_mismatch in that case, which surfaces the
 * misconfiguration clearly.
 */
export function defaultBaseUrlForRegion(region: string): string {
  const r = KNOWN_REGIONS.has(region) ? region : DEFAULT_REGION;
  return `https://${r}.ingest.agentping.io`;
}

/**
 * Control-plane hostname for the SDK to call (guard-checks-spec). The guard
 * gate is served by the control plane, not the ingest edge, so it has its own
 * base URL. The EU control plane is served at the apex; other regions are
 * subdomained.
 */
export function defaultControlUrlForRegion(region: string): string {
  const r = KNOWN_REGIONS.has(region) ? region : DEFAULT_REGION;
  return r === "eu" ? "https://agentping.io" : `https://${r}.agentping.io`;
}

let current: SdkState | null = null;

export function getState(): SdkState | null {
  return current;
}

export function requireState(): SdkState {
  if (!current) {
    throw new Error("agentping.init() has not been called");
  }
  return current;
}

export function initState(options: InitOptions = {}): SdkState {
  const apiKey =
    options.apiKey ??
    (typeof process !== "undefined"
      ? process.env["AGENTPING_API_KEY"]
      : undefined) ??
    "";
  const region = extractRegionLocal(apiKey);
  const baseUrl =
    options.baseUrl ??
    (typeof process !== "undefined"
      ? process.env["AGENTPING_BASE_URL"]
      : undefined) ??
    defaultBaseUrlForRegion(region);
  const controlUrl =
    options.controlUrl ??
    (typeof process !== "undefined"
      ? process.env["AGENTPING_CONTROL_URL"]
      : undefined) ??
    defaultControlUrlForRegion(region);

  if (!apiKey) {
    warnOnce(
      "auth_error",
      "no AGENTPING_API_KEY supplied; events will be dropped client-side.",
    );
  }

  if (current) {
    current.worker.stop();
  }
  const queue = new BoundedQueue<Envelope>(
    options.queueSize ?? DEFAULT_QUEUE_SIZE,
  );
  const client = new HttpClient({
    apiKey,
    baseUrl,
    fetchImpl: options.fetchImpl,
  });
  const controlClient = new HttpClient({
    apiKey,
    baseUrl: controlUrl,
    fetchImpl: options.fetchImpl,
  });
  const flushIntervalMs = options.flushIntervalMs ?? DEFAULT_FLUSH_INTERVAL_MS;
  const batchSize = options.batchSize ?? DEFAULT_BATCH_SIZE;
  const worker = new Worker({ queue, client, flushIntervalMs, batchSize });

  current = {
    apiKey,
    baseUrl,
    controlUrl,
    region,
    queue,
    worker,
    client,
    controlClient,
    flushIntervalMs,
    batchSize,
  };

  worker.start();
  registerExitHandler();
  return current;
}

export function shutdown(): void {
  if (!current) return;
  current.worker.stop();
  current = null;
}

export function status(): StatusSnapshot {
  if (!current) {
    return {
      queueSize: 0,
      droppedCount: 0,
      lastFlushTs: null,
      lastError: null,
    };
  }
  return {
    queueSize: current.queue.size,
    droppedCount: current.queue.droppedCount,
    lastFlushTs: current.worker.lastFlushTs,
    lastError: current.worker.lastError,
  };
}

function extractRegionLocal(apiKey: string): string {
  const m = /^apk_([a-z]{2})_[0-9a-f]{32}$/.exec(apiKey);
  if (!m || !m[1]) return "eu";
  return m[1];
}
