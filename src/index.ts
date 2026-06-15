import { Run, type RunStartOptions } from "./run.js";
import { uuid7Hex } from "./ids.js";
import {
  initState,
  requireState,
  shutdown as shutdownState,
  status as statusSnapshot,
  type InitOptions,
  type StatusSnapshot,
} from "./state.js";

export { Run } from "./run.js";
export type { RunStartOptions, RunFinishOptions } from "./run.js";
export type { InitOptions, StatusSnapshot } from "./state.js";

// The guard gate (guard-checks-spec): agentping.guard.check(...).
export * as guard from "./guard.js";
export { Paused } from "./guard.js";
export type { GuardVerdict, GuardCheckOptions } from "./guard.js";

export function init(options: InitOptions = {}): void {
  initState(options);
}

export function run(agent: string, options: RunStartOptions = {}): Run {
  const state = requireState();
  return new Run(state, agent, options);
}

export interface HeartbeatOptions {
  status?: "ok" | "failed" | "timeout";
  costUsd?: number;
  durationMs?: number;
  metadata?: Record<string, unknown>;
  customerId?: string;
  feature?: string;
  inputTokens?: number;
  outputTokens?: number;
}

export function heartbeat(
  agent: string,
  options: HeartbeatOptions = {},
): string {
  const state = requireState();
  const now = Date.now();
  const finishedAt = new Date(now).toISOString();
  const durationMs = options.durationMs ?? 0;
  const startedAt = new Date(now - durationMs).toISOString();

  const id = `run_${state.region}_${uuid7Hex()}`;
  const body: Record<string, unknown> = {
    id,
    agent,
    status: options.status ?? "ok",
    started_at: startedAt,
    finished_at: finishedAt,
  };
  if (options.costUsd !== undefined) body["cost_usd"] = options.costUsd;
  if (options.metadata) body["metadata"] = options.metadata;
  if (options.customerId) body["customer_id"] = options.customerId;
  if (options.feature) body["feature"] = options.feature;
  if (options.inputTokens !== undefined)
    body["input_tokens"] = options.inputTokens;
  if (options.outputTokens !== undefined)
    body["output_tokens"] = options.outputTokens;

  state.queue.push({ kind: "heartbeat", body });
  state.worker.notifyEnqueued();
  return id;
}

export function status(): StatusSnapshot {
  return statusSnapshot();
}

export interface FlushOptions {
  timeoutMs?: number;
}

export async function flush(options: FlushOptions = {}): Promise<void> {
  const state = requireState();
  try {
    await state.worker.flushAll(options.timeoutMs ?? 5_000);
  } catch {
    // never throw from flush
  }
}

export function shutdown(): void {
  shutdownState();
}

export { instrumentAnthropic } from "./instrumentation/anthropic.js";
export { instrumentOpenAI } from "./instrumentation/openai.js";
export { instrumentGemini } from "./instrumentation/gemini.js";
export { instrumentMistral } from "./instrumentation/mistral.js";
export { instrumentCohere } from "./instrumentation/cohere.js";
export { instrumentBedrock } from "./instrumentation/bedrock.js";
export {
  withAgentPing,
  agentPingOnFinish,
} from "./instrumentation/vercel-ai.js";
export { AgentPingHooks } from "./instrumentation/openai-agents.js";
export { AgentPingLangChainCallbackHandler } from "./instrumentation/langchain.js";
export { runScope, runScopeAsync, getActiveRun } from "./context.js";
