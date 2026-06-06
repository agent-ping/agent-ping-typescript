import { wrapAsyncIterable } from "./streaming.js";
import { getActiveRun } from "../context.js";

interface AnthropicUsage {
  input_tokens?: number;
  output_tokens?: number;
  cache_read_input_tokens?: number;
  cache_creation_input_tokens?: number;
}

interface AnthropicResponse {
  model?: string;
  usage?: AnthropicUsage;
}

interface AnthropicStreamChunk {
  type?: string;
  message?: { usage?: AnthropicUsage; model?: string };
  usage?: AnthropicUsage;
}

interface MessagesNamespace {
  create: (...args: unknown[]) => unknown;
  stream?: (...args: unknown[]) => unknown;
}

interface AnthropicClient {
  messages: MessagesNamespace;
}

interface RunLike {
  event: (type: string, data: Record<string, unknown>) => void;
}

interface InstrumentOptions {
  run?: RunLike;
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
      emitLlmCall(options, start, requestedModel, response as AnthropicResponse);
      return response;
    },
    (err: unknown) => {
      emitFailure(options, start, requestedModel, err);
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
      emitFailure(options, start, requestedModel, err);
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

  const usage: AnthropicUsage = {};
  let model: string | undefined = requestedModel;

  const observer = {
    onChunk(chunk: AnthropicStreamChunk): void {
      try {
        if (chunk?.message?.model) model = chunk.message.model;
        const u = chunk?.message?.usage ?? chunk?.usage;
        if (u) mergeUsage(usage, u);
      } catch {
        // swallow
      }
    },
    onDone(): void {
      emitLlmCall(options, start, model, { model, usage });
    },
    onError(err: unknown): void {
      emitFailure(options, start, model, err);
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

function mergeUsage(into: AnthropicUsage, from: AnthropicUsage): void {
  if (from.input_tokens !== undefined) into.input_tokens = from.input_tokens;
  if (from.output_tokens !== undefined) {
    into.output_tokens = (into.output_tokens ?? 0) + from.output_tokens;
  }
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
): void {
  const run = options.run ?? getActiveRun();
  if (!run) return;
  try {
    const latencyMs = Date.now() - start;
    const usage = response.usage ?? {};
    const model = response.model ?? requestedModel ?? "unknown";
    const data: Record<string, unknown> = {
      provider: "anthropic",
      model,
      input_tokens: usage.input_tokens ?? 0,
      output_tokens: usage.output_tokens ?? 0,
      latency_ms: latencyMs,
    };
    if (usage.cache_read_input_tokens !== undefined) {
      data["cached_input_tokens"] = usage.cache_read_input_tokens;
    }
    if (usage.cache_creation_input_tokens !== undefined) {
      data["cache_creation_input_tokens"] = usage.cache_creation_input_tokens;
    }
    run.event("llm_call", data);
  } catch {
    // swallow
  }
}

function emitFailure(
  options: InstrumentOptions,
  start: number,
  model: string | undefined,
  err: unknown,
): void {
  const run = options.run ?? getActiveRun();
  if (!run) return;
  try {
    run.event("llm_call_error", {
      provider: "anthropic",
      model: model ?? "unknown",
      latency_ms: Date.now() - start,
      error: err instanceof Error ? err.message : String(err),
    });
  } catch {
    // swallow
  }
}

function isPromiseLike<T = unknown>(value: unknown): value is PromiseLike<T> {
  return (
    typeof value === "object" &&
    value !== null &&
    "then" in value &&
    typeof (value as { then: unknown }).then === "function"
  );
}
