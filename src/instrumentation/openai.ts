import { getActiveRun } from "../context.js";
import { wrapAsyncIterable } from "./streaming.js";

interface OpenAIUsage {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
  prompt_tokens_details?: { cached_tokens?: number };
}

interface OpenAIResponse {
  model?: string;
  usage?: OpenAIUsage;
}

interface OpenAIStreamChunk {
  model?: string;
  usage?: OpenAIUsage;
  choices?: unknown[];
}

interface ChatCompletionsNamespace {
  create: (...args: unknown[]) => unknown;
}

interface ChatNamespace {
  completions: ChatCompletionsNamespace;
}

interface OpenAIClient {
  chat: ChatNamespace;
}

interface RunLike {
  event: (type: string, data: Record<string, unknown>) => void;
}

interface InstrumentOptions {
  run?: RunLike;
  mode?: "standard" | "batch";
}

export function instrumentOpenAI<T extends OpenAIClient>(
  client: T,
  options: InstrumentOptions = {},
): T {
  if (
    !client ||
    typeof client !== "object" ||
    !client.chat ||
    !client.chat.completions
  ) {
    return client;
  }

  const originalCreate = client.chat.completions.create.bind(client.chat.completions);

  const wrappedCreate = (...args: unknown[]): unknown => {
    const start = Date.now();
    const firstArg = args[0] as { stream?: boolean; model?: string } | undefined;
    const isStream = Boolean(firstArg?.stream);
    let result: unknown;
    try {
      result = originalCreate(...args);
    } catch (err) {
      throw err;
    }
    if (isStream) {
      return wrapStreamPromise(result, options, start, firstArg?.model);
    }
    return wrapResponsePromise(result, options, start, firstArg?.model);
  };

  const proxy = new Proxy(client, {
    get(target, prop, receiver) {
      if (prop === "chat") {
        return new Proxy(target.chat, {
          get(t1, p1, r1) {
            if (p1 === "completions") {
              return new Proxy(t1.completions, {
                get(t2, p2, r2) {
                  if (p2 === "create") return wrappedCreate;
                  return Reflect.get(t2, p2, r2);
                },
              });
            }
            return Reflect.get(t1, p1, r1);
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
      emitLlmCall(options, start, requestedModel, response as OpenAIResponse);
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

  let model: string | undefined = requestedModel;
  let usage: OpenAIUsage | undefined;

  const observer = {
    onChunk(chunk: OpenAIStreamChunk): void {
      try {
        if (chunk?.model) model = chunk.model;
        if (chunk?.usage) usage = chunk.usage;
      } catch {
        // swallow
      }
    },
    onDone(): void {
      emitLlmCall(options, start, model, { model, ...(usage ? { usage } : {}) });
    },
    onError(err: unknown): void {
      emitFailure(options, start, model, err);
    },
  };

  const wrappedIterable = wrapAsyncIterable(
    stream as AsyncIterable<OpenAIStreamChunk>,
    observer,
  );

  return new Proxy(stream as object, {
    get(target, prop, receiver) {
      if (prop === Symbol.asyncIterator) {
        return wrappedIterable[Symbol.asyncIterator].bind(wrappedIterable);
      }
      return Reflect.get(target, prop, receiver);
    },
  });
}

function emitLlmCall(
  options: InstrumentOptions,
  start: number,
  requestedModel: string | undefined,
  response: OpenAIResponse,
): void {
  const run = options.run ?? getActiveRun();
  if (!run) return;
  try {
    const latencyMs = Date.now() - start;
    const usage = response.usage ?? {};
    const cachedTokens = usage.prompt_tokens_details?.cached_tokens;
    const promptTokens = usage.prompt_tokens ?? 0;
    const uncached =
      cachedTokens !== undefined ? Math.max(0, promptTokens - cachedTokens) : promptTokens;
    const model = response.model ?? requestedModel ?? "unknown";
    const data: Record<string, unknown> = {
      provider: "openai",
      model,
      input_tokens: uncached,
      output_tokens: usage.completion_tokens ?? 0,
      latency_ms: latencyMs,
    };
    if (cachedTokens !== undefined && cachedTokens > 0) {
      data["cached_input_tokens"] = cachedTokens;
    }
    if (options.mode === "batch") {
      data["mode"] = "batch";
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
      provider: "openai",
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
