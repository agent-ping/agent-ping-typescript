import { wrapAsyncIterable } from "./streaming.js";
import {
  emitErroredLlmCall,
  isPromiseLike,
  putIfPositive,
  putIfString,
  resolveRun,
  type InstrumentOptions as BaseInstrumentOptions,
} from "./shared.js";

interface GeminiUsage {
  promptTokenCount?: number;
  candidatesTokenCount?: number;
  cachedContentTokenCount?: number;
  thoughtsTokenCount?: number;
  totalTokenCount?: number;
}

interface GeminiPart {
  functionCall?: unknown;
}

interface GeminiCandidate {
  finishReason?: string;
  content?: { parts?: GeminiPart[] };
}

interface GeminiResponse {
  usageMetadata?: GeminiUsage;
  modelVersion?: string;
  candidates?: GeminiCandidate[];
}

interface GeminiModelsNamespace {
  generateContent: (...args: unknown[]) => unknown;
  generateContentStream?: (...args: unknown[]) => unknown;
  embedContent?: (...args: unknown[]) => unknown;
}

type GeminiStreamChunk = GeminiResponse;

interface GeminiEmbedResponse {
  usageMetadata?: { totalTokenCount?: number; promptTokenCount?: number };
}

interface GeminiClient {
  models: GeminiModelsNamespace;
}

export interface InstrumentOptions extends BaseInstrumentOptions {
  mode?: "standard" | "batch";
}

const PROVIDER = "gemini";

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
      emitErroredLlmCall(options, PROVIDER, requestedModel, start, err);
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
          emitErroredLlmCall(options, PROVIDER, requestedModel, start, err, { stream: true });
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
          emitErroredLlmCall(options, PROVIDER, requestedModel, start, err, { kind: "embedding" });
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
        emitErroredLlmCall(options, PROVIDER, requestedModel, start, err, { stream: true });
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
  let finishReason: string | undefined;
  let toolCalls = 0;

  return wrapAsyncIterable(stream as AsyncIterable<GeminiStreamChunk>, {
    onChunk(chunk: GeminiStreamChunk): void {
      if (chunk?.modelVersion) model = chunk.modelVersion;
      // usageMetadata is cumulative, so the last chunk's value is the total.
      if (chunk?.usageMetadata) usage = chunk.usageMetadata;
      const candidate = chunk?.candidates?.[0];
      if (candidate?.finishReason) finishReason = candidate.finishReason;
      toolCalls += countFunctionCalls(candidate);
    },
    onDone(): void {
      emitLlmCall(
        options,
        start,
        model,
        { modelVersion: model, usageMetadata: usage },
        { stream: true, finishReason, toolCalls },
      );
    },
    onError(err: unknown): void {
      emitErroredLlmCall(options, PROVIDER, model, start, err, { stream: true });
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
      emitErroredLlmCall(options, PROVIDER, requestedModel, start, err, { kind: "embedding" });
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
  const run = resolveRun(options);
  if (!run) return;
  try {
    const latencyMs = Date.now() - start;
    const total = response.usageMetadata?.totalTokenCount ?? response.usageMetadata?.promptTokenCount ?? 0;
    run.event("llm_call", {
      provider: PROVIDER,
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
      emitErroredLlmCall(options, PROVIDER, requestedModel, start, err);
      throw err;
    },
  );
}

function countFunctionCalls(candidate: GeminiCandidate | undefined): number {
  return (candidate?.content?.parts ?? []).filter((p) => p?.functionCall !== undefined).length;
}

function emitLlmCall(
  options: InstrumentOptions,
  start: number,
  requestedModel: string | undefined,
  response: GeminiResponse,
  streamInfo?: { stream: true; finishReason?: string; toolCalls: number },
): void {
  const run = resolveRun(options);
  if (!run) return;
  try {
    const latencyMs = Date.now() - start;
    const usage = response.usageMetadata ?? {};
    const candidate = response.candidates?.[0];
    const thoughts = usage.thoughtsTokenCount ?? 0;

    // promptTokenCount is gross: it includes the cached content. Thinking
    // tokens are billed as output but reported apart from candidatesTokenCount.
    const data: Record<string, unknown> = {
      provider: PROVIDER,
      model: response.modelVersion ?? requestedModel ?? "unknown",
      input_tokens: usage.promptTokenCount ?? 0,
      output_tokens: (usage.candidatesTokenCount ?? 0) + thoughts,
      latency_ms: latencyMs,
    };
    putIfPositive(data, "cached_input_tokens", usage.cachedContentTokenCount);
    putIfPositive(data, "reasoning_tokens", thoughts);
    putIfString(data, "finish_reason", streamInfo?.finishReason ?? candidate?.finishReason);
    const toolCalls = streamInfo ? streamInfo.toolCalls : countFunctionCalls(candidate);
    if (toolCalls > 0) data["tool_calls"] = toolCalls;
    if (streamInfo) data["stream"] = true;
    if (options.mode === "batch") data["mode"] = "batch";
    run.event("llm_call", data);
  } catch {
    // swallow
  }
}
