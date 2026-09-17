/**
 * Vercel AI SDK helper.
 *
 * The AI SDK doesn't expose a single client object to wrap, but every
 * `generateText` and `streamText` accepts an `onFinish` callback (and
 * `onStepFinish` for tool-using agents). This module exposes two helpers:
 *
 * - `withAgentPing(run, options?)` - returns a partial options object you
 *   can spread into `generateText` / `streamText` calls. It populates
 *   `onFinish` and `onStepFinish` with handlers that emit events into
 *   the run. Provider + model are read from the event payload (AI SDK v5
 *   exposes `response.modelId` and the model object).
 *
 * - `agentPingOnFinish(run, options?)` - returns just the `onFinish`
 *   callback for cases where you want to compose with your own handlers.
 *
 * By default one `llm_call` is emitted per `generateText` / `streamText`
 * call with the usage summed across steps, followed by one `tool_call`
 * per tool the model used. With `perStep: true` each model step becomes
 * its own `llm_call` (with its tool calls after it) and `onFinish` emits
 * nothing, so tokens are never priced twice.
 */

import { getActiveRun } from "../context.js";
import {
  DEFAULT_TOOL_PAYLOAD_MAX_CHARS,
  errorFields,
  putIfPositive,
  putIfString,
  stringifyPayload,
  type RunLike,
  type ToolPayloadOptions,
} from "./shared.js";

interface AISDKUsage {
  promptTokens?: number;
  completionTokens?: number;
  totalTokens?: number;
  // AI SDK v5 uses `inputTokens` / `outputTokens`.
  inputTokens?: number;
  outputTokens?: number;
  cachedInputTokens?: number;
  reasoningTokens?: number;
}

interface AISDKToolCall {
  toolName?: string;
  toolCallId?: string;
  /** AI SDK v5. */
  input?: unknown;
  /** AI SDK v4. */
  args?: unknown;
}

interface AISDKToolResult extends AISDKToolCall {
  /** AI SDK v5. */
  output?: unknown;
  /** AI SDK v4. */
  result?: unknown;
}

/** AI SDK v5 step content part; only tool errors are read here. */
interface AISDKContentPart {
  type?: string;
  toolCallId?: string;
  toolName?: string;
  error?: unknown;
}

/** Shape exposed by AI SDK v5 onFinish responses. */
interface AISDKResponseMetadata {
  modelId?: string;
  providerMetadata?: Record<string, unknown>;
}

interface StepLike {
  usage?: AISDKUsage;
  finishReason?: string;
  toolCalls?: AISDKToolCall[];
  toolResults?: AISDKToolResult[];
  content?: AISDKContentPart[];
  response?: AISDKResponseMetadata;
  model?: unknown;
}

interface OnFinishPayload extends StepLike {
  text?: string;
  /** AI SDK v5: usage summed over every step. */
  totalUsage?: AISDKUsage;
  /** AI SDK v5: one entry per model step. */
  steps?: StepLike[];
}

type OnStepFinishPayload = StepLike;

interface WithAgentPingOptions extends ToolPayloadOptions {
  /** Optional provider override. Auto-detected from response.modelId or the model object when omitted. */
  provider?: string;
  /** Optional model override. Auto-detected from response.modelId when omitted. */
  model?: string;
  /** Emit one `llm_call` per model step instead of one per call. Defaults to false. */
  perStep?: boolean;
}

interface AgentPingAISDKOptions {
  onFinish: (event: OnFinishPayload) => void;
  onStepFinish?: (event: OnStepFinishPayload) => void;
}

export function withAgentPing(
  runOrOptions?: RunLike | WithAgentPingOptions,
  maybeOptions: WithAgentPingOptions = {},
): AgentPingAISDKOptions {
  // Support both `withAgentPing(run, options?)` and `withAgentPing(options?)`;
  // the latter resolves the run from the active AsyncLocalStorage scope.
  const explicitRun: RunLike | undefined =
    runOrOptions && typeof (runOrOptions as RunLike).event === "function"
      ? (runOrOptions as RunLike)
      : undefined;
  const options: WithAgentPingOptions = explicitRun
    ? maybeOptions
    : ((runOrOptions as WithAgentPingOptions) ?? {});

  const start = Date.now();

  const resolveRun = (): RunLike | undefined => explicitRun ?? getActiveRun();

  const onFinish = (event: OnFinishPayload): void => {
    // Per-step mode already reported every step as it finished.
    if (options.perStep) return;
    const run = resolveRun();
    if (!run) return;
    const { provider, model } = resolveProviderModel(event, options);
    const steps = event.steps ?? [];
    emitLlmCall(run, provider, model, start, event.totalUsage ?? event.usage, {
      finishReason: event.finishReason,
      toolCalls: steps.length > 0 ? sum(steps, (s) => s.toolCalls?.length ?? 0) : event.toolCalls?.length ?? 0,
    });
    if (steps.length > 0) {
      for (const step of steps) emitToolCalls(run, step, options);
    } else {
      emitToolCalls(run, event, options);
    }
  };

  const out: AgentPingAISDKOptions = { onFinish };

  if (options.perStep) {
    let stepStart = Date.now();
    out.onStepFinish = (event: OnStepFinishPayload): void => {
      const run = resolveRun();
      if (!run) return;
      const { provider, model } = resolveProviderModel(event, options);
      emitLlmCall(run, provider, model, stepStart, event.usage, {
        finishReason: event.finishReason,
        toolCalls: event.toolCalls?.length ?? 0,
      });
      emitToolCalls(run, event, options);
      stepStart = Date.now();
    };
  }

  return out;
}

export function agentPingOnFinish(
  runOrOptions?: RunLike | WithAgentPingOptions,
  maybeOptions: WithAgentPingOptions = {},
): (event: OnFinishPayload) => void {
  return withAgentPing(runOrOptions, maybeOptions).onFinish;
}

/**
 * Pull provider + model from the event payload.
 *
 * AI SDK v5 onFinish exposes `response.modelId` like "openai/gpt-4o-mini"
 * or "anthropic/claude-sonnet-4-5". We split on the first slash; the prefix
 * is the provider, the suffix is the model. Falls back to event.model
 * object's provider/modelId if present (older AI SDK shapes), then to
 * explicit user overrides.
 */
function resolveProviderModel(
  event: StepLike,
  overrides: WithAgentPingOptions,
): { provider: string; model: string } {
  if (overrides.provider && overrides.model) {
    return { provider: overrides.provider, model: overrides.model };
  }

  const modelId = event.response?.modelId;
  if (typeof modelId === "string" && modelId.length > 0) {
    const slash = modelId.indexOf("/");
    if (slash > 0) {
      return {
        provider: overrides.provider ?? modelId.slice(0, slash),
        model: overrides.model ?? modelId.slice(slash + 1),
      };
    }
    return {
      provider: overrides.provider ?? "vercel-ai",
      model: overrides.model ?? modelId,
    };
  }

  // Older AI SDK shapes pass the model object on event.model.
  const m = event.model as { provider?: string; modelId?: string } | undefined;
  if (m?.provider || m?.modelId) {
    return {
      provider: overrides.provider ?? m.provider ?? "vercel-ai",
      model: overrides.model ?? m.modelId ?? "unknown",
    };
  }

  return {
    provider: overrides.provider ?? "vercel-ai",
    model: overrides.model ?? "unknown",
  };
}

function sum<T>(items: T[], pick: (item: T) => number): number {
  return items.reduce((total, item) => total + pick(item), 0);
}

function emitLlmCall(
  run: RunLike,
  provider: string,
  model: string,
  start: number,
  usage: AISDKUsage | undefined,
  info: { finishReason?: string; toolCalls: number },
): void {
  try {
    const latencyMs = Date.now() - start;
    // AI SDK v5 reports inputTokens gross, with cachedInputTokens as the
    // cached subset; ingest prices from the gross figure.
    const data: Record<string, unknown> = {
      provider,
      model,
      input_tokens: usage?.inputTokens ?? usage?.promptTokens ?? 0,
      output_tokens: usage?.outputTokens ?? usage?.completionTokens ?? 0,
      latency_ms: latencyMs,
    };
    putIfPositive(data, "cached_input_tokens", usage?.cachedInputTokens);
    putIfPositive(data, "reasoning_tokens", usage?.reasoningTokens);
    putIfString(data, "finish_reason", info.finishReason);
    if (info.toolCalls > 0) data["tool_calls"] = info.toolCalls;
    run.event("llm_call", data);
  } catch {
    // swallow
  }
}

/**
 * One `tool_call` per tool the step invoked, joined to its result (or
 * error) by `toolCallId`. The AI SDK runs tools itself and does not expose
 * per-tool timing, so there is no `latency_ms`.
 */
function emitToolCalls(run: RunLike, step: StepLike, options: ToolPayloadOptions): void {
  const calls = step.toolCalls ?? [];
  if (calls.length === 0) return;
  const capture = options.captureToolPayloads ?? true;
  const maxChars = options.toolPayloadMaxChars ?? DEFAULT_TOOL_PAYLOAD_MAX_CHARS;
  const results = new Map<string, AISDKToolResult>();
  for (const result of step.toolResults ?? []) {
    if (result.toolCallId) results.set(result.toolCallId, result);
  }
  const errors = new Map<string, unknown>();
  for (const part of step.content ?? []) {
    if (part?.type === "tool-error" && part.toolCallId) errors.set(part.toolCallId, part.error);
  }

  for (const call of calls) {
    try {
      const id = call.toolCallId;
      const data: Record<string, unknown> = {
        tool: call.toolName ?? "unknown",
        status: "success",
      };
      putIfString(data, "tool_invocation_id", id);
      const input = call.input ?? call.args;
      if (capture && input !== undefined) data["input"] = stringifyPayload(input, maxChars);

      if (id && errors.has(id)) {
        data["status"] = "error";
        Object.assign(data, errorFields(errors.get(id)));
      } else {
        const result = id ? results.get(id) : undefined;
        const output = result?.output ?? result?.result;
        if (capture && output !== undefined) data["output"] = stringifyPayload(output, maxChars);
      }
      run.event("tool_call", data);
    } catch {
      // swallow
    }
  }
}
