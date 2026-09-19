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

export interface ConversationOptions extends Omit<RunStartOptions, "resume"> {}

/**
 * One chat session as one run, with every turn appended to it.
 *
 * A conversation has no reliable end -- the user closes the tab, or simply stops --
 * so pass the same `sessionId` on every turn and the turns collect on a single
 * record instead of becoming a run per message. The run id and its start time are
 * both derived from the session id, so a stateless backend can rebuild this on any
 * request without storing a mapping; re-opening is idempotent server-side.
 *
 *   const chat = agentping.conversation("support-chat", sessionId);
 *   chat.event("llm_call", { provider: "anthropic", model, input_tokens, output_tokens });
 *   // ...and when the user closes the chat:
 *   chat.finish({ status: "success" });
 *
 * If finish is never called, the idle timeout closes it at its last activity.
 *
 * @param sessionId 32 hex characters whose first 12 are the session's start time in
 *   milliseconds. Use newSessionId().
 */
export function conversation(
  agent: string,
  sessionId: string,
  options: ConversationOptions = {},
): Run {
  const state = requireState();
  if (!/^[0-9a-f]{32}$/.test(sessionId)) {
    throw new Error(
      "agentping.conversation: sessionId must be 32 hex characters (see newSessionId())",
    );
  }
  const startedMs = parseInt(sessionId.slice(0, 12), 16);
  if (!Number.isFinite(startedMs) || startedMs <= 0) {
    throw new Error("agentping.conversation: sessionId does not carry a start time");
  }

  return new Run(state, agent, {
    ...options,
    idleTimeoutSeconds: options.idleTimeoutSeconds ?? 900,
    idleTimeoutStatus: options.idleTimeoutStatus ?? "success",
    resume: {
      id: `run_${state.region}_${sessionId}`,
      startedAt: new Date(startedMs).toISOString(),
    },
  });
}

/** A session id for conversation(). Hold on to it for the life of the chat. */
export function newSessionId(): string {
  return uuid7Hex();
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
