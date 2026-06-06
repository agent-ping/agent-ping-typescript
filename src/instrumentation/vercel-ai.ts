/**
 * Vercel AI SDK helper.
 *
 * The AI SDK doesn't expose a single client object to wrap, but every
 * `generateText` and `streamText` accepts an `onFinish` callback (and
 * `onStepFinish` for tool-using agents) that fires once with the final
 * usage block. This module exposes two helpers:
 *
 * - `withAgentPing(run, options?)` - returns a partial options object you
 *   can spread into `generateText` / `streamText` calls. It populates
 *   `onFinish` and `onStepFinish` with handlers that emit events into
 *   the run. Provider + model are read from the event payload (AI SDK v5
 *   exposes `response.modelId` and the model object).
 *
 * - `agentPingOnFinish(run, options?)` - returns just the `onFinish`
 *   callback for cases where you want to compose with your own handlers.
 */

interface AISDKUsage {
  promptTokens?: number;
  completionTokens?: number;
  totalTokens?: number;
  // AI SDK v5 uses `inputTokens` / `outputTokens`.
  inputTokens?: number;
  outputTokens?: number;
  cachedInputTokens?: number;
}

interface AISDKToolCall {
  toolName?: string;
  toolCallId?: string;
  args?: unknown;
}

/** Shape exposed by AI SDK v5 onFinish responses. */
interface AISDKResponseMetadata {
  modelId?: string;
  providerMetadata?: Record<string, unknown>;
}

interface OnFinishPayload {
  usage?: AISDKUsage;
  finishReason?: string;
  text?: string;
  toolCalls?: AISDKToolCall[];
  // AI SDK v5+ surfaces response metadata here.
  response?: AISDKResponseMetadata;
  // Some flows also pass the model object directly.
  model?: unknown;
}

interface OnStepFinishPayload {
  usage?: AISDKUsage;
  finishReason?: string;
  toolCalls?: AISDKToolCall[];
  response?: AISDKResponseMetadata;
  model?: unknown;
}

import { getActiveRun } from "../context.js";

interface RunLike {
  event: (type: string, data: Record<string, unknown>) => void;
}

interface WithAgentPingOptions {
  /** Optional provider override. Auto-detected from response.modelId or the model object when omitted. */
  provider?: string;
  /** Optional model override. Auto-detected from response.modelId when omitted. */
  model?: string;
  /** Set to true to emit per-step events for tool-using agents. Defaults to false. */
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
  // Support both `withAgentPing(run, options?)` and `withAgentPing(options?)`
  // — the latter resolves the run from the active AsyncLocalStorage scope.
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
    const run = resolveRun();
    if (!run) return;
    const { provider, model } = resolveProviderModel(event, options);
    emitLlmCall(run, provider, model, start, event.usage, event.toolCalls);
    if (event.finishReason && event.finishReason !== "stop") {
      try {
        run.event("finish_reason", { provider, reason: event.finishReason });
      } catch {
        // swallow
      }
    }
  };

  const out: AgentPingAISDKOptions = { onFinish };

  if (options.perStep) {
    let stepStart = Date.now();
    out.onStepFinish = (event: OnStepFinishPayload): void => {
      const run = resolveRun();
      if (!run) return;
      const { provider, model } = resolveProviderModel(event, options);
      emitLlmCall(run, provider, model, stepStart, event.usage, event.toolCalls);
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
  event: OnFinishPayload | OnStepFinishPayload,
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

function emitLlmCall(
  run: RunLike,
  provider: string,
  model: string,
  start: number,
  usage: AISDKUsage | undefined,
  toolCalls: AISDKToolCall[] | undefined,
): void {
  if (!run) return;
  try {
    const latencyMs = Date.now() - start;
    const input = usage?.inputTokens ?? usage?.promptTokens ?? 0;
    const output = usage?.outputTokens ?? usage?.completionTokens ?? 0;
    const cached = usage?.cachedInputTokens ?? 0;

    const data: Record<string, unknown> = {
      provider,
      model,
      input_tokens: cached > 0 ? Math.max(0, input - cached) : input,
      output_tokens: output,
      latency_ms: latencyMs,
    };
    if (cached > 0) {
      data["cached_input_tokens"] = cached;
    }
    run.event("llm_call", data);

    if (toolCalls && toolCalls.length > 0) {
      for (const call of toolCalls) {
        run.event("tool_call", {
          tool: call.toolName ?? "unknown",
          tool_call_id: call.toolCallId,
        });
      }
    }
  } catch {
    // swallow
  }
}
