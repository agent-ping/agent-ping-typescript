/**
 * Helpers shared by the provider and framework instrumentations.
 *
 * The wire format they produce matches the Laravel SDK so a run looks the
 * same on the timeline whichever SDK reported it:
 *
 *  - a successful model call is `llm_call` with `provider`, `model`,
 *    `input_tokens` (gross, cached tokens included), `output_tokens`,
 *    `latency_ms`, plus `cached_input_tokens`,
 *    `cache_creation_input_tokens` and `reasoning_tokens` when non-zero,
 *    `finish_reason` and `tool_calls` when known, and `stream: true` for
 *    streamed responses;
 *  - a failed model call is `llm_call` with `status: "error"`, `error`
 *    (the message) and `exception` (the error class), so the run is
 *    flagged as errored by ingest;
 *  - a tool invocation is `tool_call` with `tool`, `status`,
 *    `tool_invocation_id`, `latency_ms`, `input`, `output`, and the same
 *    `error` / `exception` pair when the tool threw.
 */

import { getActiveRun } from "../context.js";

export interface RunLike {
  event: (type: string, data: Record<string, unknown>) => void;
}

export interface InstrumentOptions {
  /** Explicit run to report to. Defaults to the active run context. */
  run?: RunLike;
}

export interface ToolPayloadOptions {
  /**
   * Send tool arguments and results as `tool_call` input and output.
   * Defaults to true. Turn it off if your tools handle sensitive data.
   */
  captureToolPayloads?: boolean;
  /** Truncate each tool argument list and result at this length. Default 4000. */
  toolPayloadMaxChars?: number;
}

export const DEFAULT_TOOL_PAYLOAD_MAX_CHARS = 4000;

export function resolveRun(options: InstrumentOptions): RunLike | undefined {
  return options.run ?? getActiveRun();
}

export function isPromiseLike<T = unknown>(value: unknown): value is PromiseLike<T> {
  return (
    typeof value === "object" &&
    value !== null &&
    "then" in value &&
    typeof (value as { then: unknown }).then === "function"
  );
}

/** Message and class name of a thrown value, as `error` and `exception`. */
export function errorFields(err: unknown): { error: string; exception: string } {
  if (err instanceof Error) {
    return {
      error: err.message,
      exception: err.constructor?.name || err.name || "Error",
    };
  }
  if (err && typeof err === "object") {
    const message = (err as { message?: unknown }).message;
    const name = (err as { name?: unknown }).name;
    return {
      error: typeof message === "string" ? message : String(err),
      exception: typeof name === "string" ? name : "Error",
    };
  }
  return { error: String(err), exception: "Error" };
}

/** Add a non-zero numeric field to an event payload. */
export function putIfPositive(
  data: Record<string, unknown>,
  key: string,
  value: number | undefined | null,
): void {
  if (typeof value === "number" && Number.isFinite(value) && value > 0) {
    data[key] = value;
  }
}

/** Add a string field to an event payload when it has a value. */
export function putIfString(
  data: Record<string, unknown>,
  key: string,
  value: unknown,
): void {
  if (typeof value === "string" && value.length > 0) {
    data[key] = value;
  }
}

/**
 * Stringify a tool argument list or result for the timeline, capped at
 * `maxChars`. Strings pass through; anything else is JSON encoded.
 */
export function stringifyPayload(
  value: unknown,
  maxChars: number = DEFAULT_TOOL_PAYLOAD_MAX_CHARS,
): string | undefined {
  if (value === undefined) return undefined;
  let text: string;
  if (typeof value === "string") {
    text = value;
  } else {
    try {
      text = JSON.stringify(value) ?? String(value);
    } catch {
      text = String(value);
    }
  }
  if (maxChars > 0 && text.length > maxChars) {
    return `${text.slice(0, maxChars)}...`;
  }
  return text;
}

/**
 * Emit an errored `llm_call`. Ingest flags the run as errored when it sees
 * `status: "error"` or an `error` field on a run event.
 */
export function emitErroredLlmCall(
  options: InstrumentOptions,
  provider: string,
  model: string | undefined,
  start: number,
  err: unknown,
  extra: Record<string, unknown> = {},
): void {
  const run = resolveRun(options);
  if (!run) return;
  try {
    run.event("llm_call", {
      provider,
      model: model ?? "unknown",
      latency_ms: Date.now() - start,
      status: "error",
      ...errorFields(err),
      ...extra,
    });
  } catch {
    // swallow
  }
}
