import { wrapAsyncIterable } from "./streaming.js";
import {
  emitErroredLlmCall,
  isPromiseLike,
  putIfPositive,
  putIfString,
  resolveRun,
  type InstrumentOptions,
} from "./shared.js";

interface AnthropicUsage {
  input_tokens?: number;
  output_tokens?: number;
  cache_read_input_tokens?: number;
  cache_creation_input_tokens?: number;
}

interface AnthropicContentBlock {
  type?: string;
}

interface AnthropicResponse {
  model?: string;
  usage?: AnthropicUsage;
  stop_reason?: string | null;
  content?: AnthropicContentBlock[];
}

interface AnthropicStreamChunk {
  type?: string;
  message?: { usage?: AnthropicUsage; model?: string };
  usage?: AnthropicUsage;
  delta?: { stop_reason?: string | null };
  content_block?: AnthropicContentBlock;
}

interface MessagesNamespace {
  create: (...args: unknown[]) => unknown;
  stream?: (...args: unknown[]) => unknown;
}

interface AnthropicClient {
  messages: MessagesNamespace;
}

/** What a stream has told us so far; folded into one `llm_call` at the end. */
interface StreamState {
  model?: string;
  usage: AnthropicUsage;
  stopReason?: string;
  toolCalls: number;
}

export function instrumentAnthropic<T extends AnthropicClient>(
  client: T,
  options: InstrumentOptions = {},
): T {
  if (!client || typeof client !== "object" || !client.messages) {
    return client;
  }

  const originalCreate = client.messages.create.bind(client.messages);
  const originalStream = client.messages.stream?.bind(client.messages);

  const wrappedCreate = (...args: unknown[]): unknown => {
    const start = Date.now();
    const firstArg = args[0] as { stream?: boolean; model?: string } | undefined;
    const isStream = Boolean(firstArg?.stream);
    let result: unknown;
    try {
      result = originalCreate(...args);
    } catch (err) {
      // Synchronous throw: never our fault. Just propagate.
      throw err;
    }
    if (isStream) {
      return wrapStreamPromise(result, options, start, firstArg?.model);
    }
    return wrapResponsePromise(result, options, start, firstArg?.model);
  };

  const wrappedStream = originalStream
    ? (...args: unknown[]): unknown => {
        const start = Date.now();
        const firstArg = args[0] as { model?: string } | undefined;
        const result = originalStream(...args);
        return wrapStreamResult(result, options, start, firstArg?.model);
      }
    : undefined;

  const proxy = new Proxy(client, {
    get(target, prop, receiver) {
      if (prop === "messages") {
        return new Proxy(target.messages, {
          get(t, p, r) {
            if (p === "create") return wrappedCreate;
            if (p === "stream" && wrappedStream) return wrappedStream;
            return Reflect.get(t, p, r);
          },
        });
      }
      return Reflect.get(target, prop, receiver);
    },
  });

  return proxy;
}

function wrapResponsePromise(
  result: unknown,
  options: InstrumentOptions,
  start: number,
  requestedModel: string | undefined,
): unknown {
  if (!isPromiseLike(result)) return result;
  return result.then(
    (response: unknown) => {
      emitLlmCall(options, start, requestedModel, response as AnthropicResponse, false);
      return response;
    },
    (err: unknown) => {
      emitErroredLlmCall(options, "anthropic", requestedModel, start, err);
      throw err;
    },
  );
}

function wrapStreamPromise(
  result: unknown,
  options: InstrumentOptions,
  start: number,
  requestedModel: string | undefined,
): unknown {
  if (!isPromiseLike(result)) {
    return wrapStreamResult(result, options, start, requestedModel);
  }
  return result.then(
    (stream: unknown) => wrapStreamResult(stream, options, start, requestedModel),
    (err: unknown) => {
      emitErroredLlmCall(options, "anthropic", requestedModel, start, err, { stream: true });
      throw err;
    },
  );
}

function wrapStreamResult(
  stream: unknown,
  options: InstrumentOptions,
  start: number,
  requestedModel: string | undefined,
): unknown {
  if (!stream || typeof stream !== "object") return stream;
  if (!(Symbol.asyncIterator in stream)) return stream;

  const state: StreamState = { model: requestedModel, usage: {}, toolCalls: 0 };

  const observer = {
    onChunk(chunk: AnthropicStreamChunk): void {
      try {
        observeChunk(state, chunk);
      } catch {
        // swallow
      }
    },
    onDone(): void {
      emitLlmCall(
        options,
        start,
        state.model,
        {
          model: state.model,
          usage: state.usage,
          stop_reason: state.stopReason,
          content: Array.from({ length: state.toolCalls }, () => ({ type: "tool_use" })),
        },
        true,
      );
    },
    onError(err: unknown): void {
      emitErroredLlmCall(options, "anthropic", state.model, start, err, { stream: true });
    },
  };

  const wrappedIterable = wrapAsyncIterable(
    stream as AsyncIterable<AnthropicStreamChunk>,
    observer,
  );

  // Preserve other stream methods (finalMessage, on, etc.) by proxying.
  return new Proxy(stream as object, {
    get(target, prop, receiver) {
      if (prop === Symbol.asyncIterator) {
        return wrappedIterable[Symbol.asyncIterator].bind(wrappedIterable);
      }
      return Reflect.get(target, prop, receiver);
    },
  });
}

/**
 * Fold one streaming event into the state. `message_start` carries the
 * model and the input side of usage; `message_delta` carries the stop
 * reason and a cumulative `output_tokens`, so the last value wins rather
 * than being summed.
 */
function observeChunk(state: StreamState, chunk: AnthropicStreamChunk): void {
  if (chunk?.message?.model) state.model = chunk.message.model;
  const u = chunk?.message?.usage ?? chunk?.usage;
  if (u) mergeUsage(state.usage, u);
  const stopReason = chunk?.delta?.stop_reason;
  if (typeof stopReason === "string") state.stopReason = stopReason;
  if (chunk?.type === "content_block_start" && chunk.content_block?.type === "tool_use") {
    state.toolCalls += 1;
  }
}

function mergeUsage(into: AnthropicUsage, from: AnthropicUsage): void {
  if (from.input_tokens !== undefined) into.input_tokens = from.input_tokens;
  if (from.output_tokens !== undefined) into.output_tokens = from.output_tokens;
  if (from.cache_read_input_tokens !== undefined) {
    into.cache_read_input_tokens = from.cache_read_input_tokens;
  }
  if (from.cache_creation_input_tokens !== undefined) {
    into.cache_creation_input_tokens = from.cache_creation_input_tokens;
  }
}

function emitLlmCall(
  options: InstrumentOptions,
  start: number,
  requestedModel: string | undefined,
  response: AnthropicResponse,
  stream: boolean,
): void {
  const run = resolveRun(options);
  if (!run) return;
  try {
    const latencyMs = Date.now() - start;
    const usage = response.usage ?? {};
    const model = response.model ?? requestedModel ?? "unknown";
    const cached = usage.cache_read_input_tokens ?? 0;
    const cacheCreation = usage.cache_creation_input_tokens ?? 0;
    // Anthropic reports input_tokens net of cache reads and writes. Ingest
    // prices from gross input, so add them back.
    const data: Record<string, unknown> = {
      provider: "anthropic",
      model,
      input_tokens: (usage.input_tokens ?? 0) + cached + cacheCreation,
      output_tokens: usage.output_tokens ?? 0,
      latency_ms: latencyMs,
    };
    putIfPositive(data, "cached_input_tokens", cached);
    putIfPositive(data, "cache_creation_input_tokens", cacheCreation);
    putIfString(data, "finish_reason", response.stop_reason);
    const toolCalls = (response.content ?? []).filter((b) => b?.type === "tool_use").length;
    if (toolCalls > 0) data["tool_calls"] = toolCalls;
    if (stream) data["stream"] = true;
    run.event("llm_call", data);
  } catch {
    // swallow
  }
}
