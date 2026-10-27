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
import { DEFAULT_LLM_CONTENT_MAX_CHARS, getState } from "../state.js";

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

/** A message on the wire: what the run page renders as a labelled bubble. */
export interface WireMessage {
  role: string;
  content: string;
}

/**
 * Records when a stream's first content chunk arrived. Collectors call
 * `mark()` when a chunk carries content (text or a tool call), not on
 * housekeeping events, and `ttftMs(start)` is the time to first token the
 * Pulse page charts.
 */
export class FirstChunkClock {
  private firstContentAt: number | undefined;

  mark(): void {
    if (this.firstContentAt === undefined) this.firstContentAt = Date.now();
  }

  ttftMs(start: number): number | undefined {
    if (this.firstContentAt === undefined) return undefined;
    return Math.max(0, this.firstContentAt - start);
  }

  put(data: Record<string, unknown>, start: number): void {
    const ttft = this.ttftMs(start);
    if (ttft !== undefined) data["ttft_ms"] = ttft;
  }
}

export function contentCaptureEnabled(): boolean {
  return getState()?.captureLlmContent ?? false;
}

export function clipText(text: string, maxChars?: number): string {
  const limit = maxChars ?? getState()?.llmContentMaxChars ?? DEFAULT_LLM_CONTENT_MAX_CHARS;
  if (limit > 0 && text.length > limit) return `${text.slice(0, limit)}...`;
  return text;
}

/**
 * Flatten a message's content: a string as is, a list of parts joined, with
 * non-text parts (images, tool calls, tool results) named in brackets.
 */
export function messageText(content: unknown): string {
  if (typeof content === "string") return content;
  if (content === null || content === undefined) return "";
  if (Array.isArray(content)) {
    return content
      .map((part) => {
        if (typeof part === "string") return part;
        const record = asRecord(part);
        if (!record) return stringifyPayload(part, 200) ?? "";
        if (typeof record["text"] === "string") return record["text"];
        const kind = record["type"];
        const name = record["name"] ?? record["tool_name"] ?? record["id"];
        if (typeof kind === "string") {
          return `[${kind}${typeof name === "string" ? ` ${name}` : ""}]`;
        }
        return stringifyPayload(part, 200) ?? "";
      })
      .join("\n");
  }
  return stringifyPayload(content, 200) ?? "";
}

/** A provider's message list as the wire's [{role, content}], each clipped. */
export function renderMessages(messages: unknown, defaultRole = "user"): WireMessage[] {
  if (typeof messages === "string") return [{ role: defaultRole, content: clipText(messages) }];
  if (!Array.isArray(messages)) return [];
  return messages.map((message) => {
    if (typeof message === "string") return { role: defaultRole, content: clipText(message) };
    const record = asRecord(message) ?? {};
    const role = typeof record["role"] === "string" && record["role"] ? record["role"] : defaultRole;
    return { role, content: clipText(messageText(record["content"] ?? record["parts"])) };
  });
}

/** Add captured prompt and completion text when the SDK is configured for it. */
export function putContent(
  data: Record<string, unknown>,
  parts: { input?: unknown; output?: unknown; system?: unknown },
): void {
  if (!contentCaptureEnabled()) return;
  if (parts.input !== undefined && parts.input !== null) {
    const rendered = renderMessages(parts.input);
    if (rendered.length > 0) data["input"] = rendered;
  }
  if (parts.system !== undefined && parts.system !== null) {
    const text = typeof parts.system === "string" ? parts.system : messageText(parts.system);
    if (text) data["system_instructions"] = clipText(text);
  }
  if (parts.output !== undefined && parts.output !== null) {
    const text = typeof parts.output === "string" ? parts.output : messageText(parts.output);
    if (text) data["output"] = clipText(text);
  }
}

function asRecord(value: unknown): Record<string, unknown> | undefined {
  return typeof value === "object" && value !== null ? (value as Record<string, unknown>) : undefined;
}

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
