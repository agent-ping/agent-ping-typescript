import { wrapAsyncIterable } from "./streaming.js";
import {
  emitErroredLlmCall,
  isPromiseLike,
  putIfString,
  resolveRun,
  type InstrumentOptions as BaseInstrumentOptions,
} from "./shared.js";

interface CohereBilledUnits {
  inputTokens?: number;
  outputTokens?: number;
  searchUnits?: number;
  classifications?: number;
}

interface CohereUsage {
  billedUnits?: CohereBilledUnits;
  tokens?: { inputTokens?: number; outputTokens?: number };
}

interface CohereResponse {
  usage?: CohereUsage;
  finishReason?: string;
  message?: { toolCalls?: unknown[] };
}

interface InstrumentableCohereClient {
  chat: (...args: unknown[]) => unknown;
  chatStream?: (...args: unknown[]) => unknown;
  embed?: (...args: unknown[]) => unknown;
}

interface CohereStreamEvent {
  type?: string;
  delta?: { usage?: CohereUsage; finishReason?: string };
}

interface CohereEmbedResponse {
  meta?: { billedUnits?: { inputTokens?: number } };
}

export interface InstrumentOptions extends BaseInstrumentOptions {
  mode?: "standard" | "batch";
}

const PROVIDER = "cohere";

export function instrumentCohere<T extends InstrumentableCohereClient>(
  client: T,
  options: InstrumentOptions = {},
): T {
  if (!client || typeof client !== "object" || typeof client.chat !== "function") {
    return client;
  }

  const originalChat = client.chat.bind(client);
  const originalChatStream = client.chatStream?.bind(client);
  const originalEmbed = client.embed?.bind(client);

  const wrappedChat = (...args: unknown[]): unknown => {
    const start = Date.now();
    const firstArg = args[0] as { model?: string } | undefined;
    const requestedModel = firstArg?.model;

    let result: unknown;
    try {
      result = originalChat(...args);
    } catch (err) {
      emitErroredLlmCall(options, PROVIDER, requestedModel, start, err);
      throw err;
    }

    return wrapResponsePromise(result, options, start, requestedModel);
  };

  const wrappedChatStream = originalChatStream
    ? (...args: unknown[]): unknown => {
        const start = Date.now();
        const firstArg = args[0] as { model?: string } | undefined;
        const requestedModel = firstArg?.model;
        try {
          const result = originalChatStream(...args);
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
      if (prop === "chat") return wrappedChat;
      if (prop === "chatStream" && wrappedChatStream) return wrappedChatStream;
      if (prop === "embed" && wrappedEmbed) return wrappedEmbed;
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

  let billed: CohereBilledUnits | undefined;
  let finishReason: string | undefined;
  let toolCalls = 0;

  return wrapAsyncIterable(stream as AsyncIterable<CohereStreamEvent>, {
    onChunk(event: CohereStreamEvent): void {
      if (event?.type === "tool-call-start") toolCalls += 1;
      if (event?.type === "message-end") {
        const b = event.delta?.usage?.billedUnits;
        if (b) billed = b;
        if (event.delta?.finishReason) finishReason = event.delta.finishReason;
      }
    },
    onDone(): void {
      emitLlmCall(
        options,
        start,
        requestedModel,
        billed ? { usage: { billedUnits: billed } } : {},
        { stream: true, finishReason, toolCalls },
      );
    },
    onError(err: unknown): void {
      emitErroredLlmCall(options, PROVIDER, requestedModel, start, err, { stream: true });
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
    emitEmbedding(options, start, requestedModel, result as CohereEmbedResponse);
    return result;
  }
  return result.then(
    (response: unknown) => {
      emitEmbedding(options, start, requestedModel, response as CohereEmbedResponse);
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
  response: CohereEmbedResponse,
): void {
  const run = resolveRun(options);
  if (!run) return;
  try {
    const latencyMs = Date.now() - start;
    const input = response.meta?.billedUnits?.inputTokens ?? 0;
    run.event("llm_call", {
      provider: PROVIDER,
      model: requestedModel ?? "unknown",
      kind: "embedding",
      input_tokens: input,
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
    emitLlmCall(options, start, requestedModel, result as CohereResponse);
    return result;
  }
  return result.then(
    (response: unknown) => {
      emitLlmCall(options, start, requestedModel, response as CohereResponse);
      return response;
    },
    (err: unknown) => {
      emitErroredLlmCall(options, PROVIDER, requestedModel, start, err);
      throw err;
    },
  );
}

function emitLlmCall(
  options: InstrumentOptions,
  start: number,
  requestedModel: string | undefined,
  response: CohereResponse,
  streamInfo?: { stream: true; finishReason?: string; toolCalls: number },
): void {
  const run = resolveRun(options);
  if (!run) return;
  try {
    const latencyMs = Date.now() - start;
    const billed = response.usage?.billedUnits ?? {};
    const data: Record<string, unknown> = {
      provider: PROVIDER,
      model: requestedModel ?? "unknown",
      input_tokens: billed.inputTokens ?? 0,
      output_tokens: billed.outputTokens ?? 0,
      latency_ms: latencyMs,
    };
    putIfString(data, "finish_reason", streamInfo?.finishReason ?? response.finishReason);
    const toolCalls = streamInfo ? streamInfo.toolCalls : response.message?.toolCalls?.length ?? 0;
    if (toolCalls > 0) data["tool_calls"] = toolCalls;
    if (streamInfo) data["stream"] = true;
    if (options.mode === "batch") data["mode"] = "batch";
    run.event("llm_call", data);
  } catch {
    // swallow
  }
}
