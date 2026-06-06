import { getActiveRun } from "../context.js";
import { wrapAsyncIterable } from "./streaming.js";

interface GeminiUsage {
  promptTokenCount?: number;
  candidatesTokenCount?: number;
  cachedContentTokenCount?: number;
  totalTokenCount?: number;
}

interface GeminiResponse {
  usageMetadata?: GeminiUsage;
  modelVersion?: string;
}

interface GeminiModelsNamespace {
  generateContent: (...args: unknown[]) => unknown;
  generateContentStream?: (...args: unknown[]) => unknown;
  embedContent?: (...args: unknown[]) => unknown;
}

interface GeminiStreamChunk {
  usageMetadata?: GeminiUsage;
  modelVersion?: string;
}

interface GeminiEmbedResponse {
  usageMetadata?: { totalTokenCount?: number; promptTokenCount?: number };
}

interface GeminiClient {
  models: GeminiModelsNamespace;
}

interface RunLike {
  event: (type: string, data: Record<string, unknown>) => void;
}

interface InstrumentOptions {
  run?: RunLike;
  mode?: "standard" | "batch";
}

export function instrumentGemini<T extends GeminiClient>(
  client: T,
  options: InstrumentOptions = {},
): T {
  if (!client || typeof client !== "object" || !client.models) {
    return client;
  }

  const originalGenerate = client.models.generateContent.bind(client.models);
  const originalStream = client.models.generateContentStream?.bind(client.models);
  const originalEmbed = client.models.embedContent?.bind(client.models);

  const wrappedGenerate = (...args: unknown[]): unknown => {
    const start = Date.now();
    const firstArg = args[0] as { model?: string } | undefined;
    const requestedModel = firstArg?.model;

    let result: unknown;
    try {
      result = originalGenerate(...args);
    } catch (err) {
      emitFailure(options, start, requestedModel, err);
      throw err;
    }

    return wrapResponsePromise(result, options, start, requestedModel);
  };

  const wrappedStream = originalStream
    ? (...args: unknown[]): unknown => {
        const start = Date.now();
        const firstArg = args[0] as { model?: string } | undefined;
        const requestedModel = firstArg?.model;
        try {
          const result = originalStream(...args);
          return wrapStreamPromise(result, options, start, requestedModel);
        } catch (err) {
          emitFailure(options, start, requestedModel, err);
          throw err;
        }
      }
    : undefined;

  const wrappedEmbed = originalEmbed
    ? (...args: unknown[]): unknown => {
        const start = Date.now();
        const firstArg = args[0] as { model?: string } | undefined;
        const requestedModel = firstArg?.model;
        try {
          const result = originalEmbed(...args);
          return wrapEmbedPromise(result, options, start, requestedModel);
        } catch (err) {
          emitFailure(options, start, requestedModel, err);
          throw err;
        }
      }
    : undefined;

  return new Proxy(client, {
    get(target, prop, receiver) {
      if (prop === "models") {
        return new Proxy(target.models, {
          get(t, p, r) {
            if (p === "generateContent") return wrappedGenerate;
            if (p === "generateContentStream" && wrappedStream) return wrappedStream;
            if (p === "embedContent" && wrappedEmbed) return wrappedEmbed;
            return Reflect.get(t, p, r);
          },
        });
      }
      return Reflect.get(target, prop, receiver);
    },
  });
}

function wrapStreamPromise(
  result: unknown,
  options: InstrumentOptions,
  start: number,
  requestedModel: string | undefined,
): unknown {
  if (isPromiseLike(result)) {
    return result.then(
      (stream: unknown) => wrapStreamResult(stream, options, start, requestedModel),
      (err: unknown) => {
        emitFailure(options, start, requestedModel, err);
        throw err;
      },
    );
  }
  return wrapStreamResult(result, options, start, requestedModel);
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
  let usage: GeminiUsage | undefined;

  return wrapAsyncIterable(stream as AsyncIterable<GeminiStreamChunk>, {
    onChunk(chunk: GeminiStreamChunk): void {
      if (chunk?.modelVersion) model = chunk.modelVersion;
      if (chunk?.usageMetadata) usage = chunk.usageMetadata;
    },
    onDone(): void {
      emitLlmCall(options, start, model, { modelVersion: model, usageMetadata: usage });
    },
    onError(err: unknown): void {
      emitFailure(options, start, model, err);
    },
  });
}

function wrapEmbedPromise(
  result: unknown,
  options: InstrumentOptions,
  start: number,
  requestedModel: string | undefined,
): unknown {
  if (!isPromiseLike(result)) {
    emitEmbedding(options, start, requestedModel, result as GeminiEmbedResponse);
    return result;
  }
  return result.then(
    (response: unknown) => {
      emitEmbedding(options, start, requestedModel, response as GeminiEmbedResponse);
      return response;
    },
    (err: unknown) => {
      emitFailure(options, start, requestedModel, err);
      throw err;
    },
  );
}

function emitEmbedding(
  options: InstrumentOptions,
  start: number,
  requestedModel: string | undefined,
  response: GeminiEmbedResponse,
): void {
  const run = options.run ?? getActiveRun();
  if (!run) return;
  try {
    const latencyMs = Date.now() - start;
    const total = response.usageMetadata?.totalTokenCount ?? response.usageMetadata?.promptTokenCount ?? 0;
    run.event("llm_call", {
      provider: "gemini",
      model: requestedModel ?? "unknown",
      kind: "embedding",
      input_tokens: total,
      output_tokens: 0,
      latency_ms: latencyMs,
    });
  } catch {
    // swallow
  }
}

function wrapResponsePromise(
  result: unknown,
  options: InstrumentOptions,
  start: number,
  requestedModel: string | undefined,
): unknown {
  if (!isPromiseLike(result)) {
    emitLlmCall(options, start, requestedModel, result as GeminiResponse);
    return result;
  }
  return result.then(
    (response: unknown) => {
      emitLlmCall(options, start, requestedModel, response as GeminiResponse);
      return response;
    },
    (err: unknown) => {
      emitFailure(options, start, requestedModel, err);
      throw err;
    },
  );
}

function emitLlmCall(
  options: InstrumentOptions,
  start: number,
  requestedModel: string | undefined,
  response: GeminiResponse,
): void {
  const run = options.run ?? getActiveRun();
  if (!run) return;
  try {
    const latencyMs = Date.now() - start;
    const usage = response.usageMetadata ?? {};
    const prompt = usage.promptTokenCount ?? 0;
    const cached = usage.cachedContentTokenCount ?? 0;
    const uncached = cached > 0 ? Math.max(0, prompt - cached) : prompt;

    const data: Record<string, unknown> = {
      provider: "gemini",
      model: response.modelVersion ?? requestedModel ?? "unknown",
      input_tokens: uncached,
      output_tokens: usage.candidatesTokenCount ?? 0,
      latency_ms: latencyMs,
    };
    if (cached > 0) {
      data["cached_input_tokens"] = cached;
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
      provider: "gemini",
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
