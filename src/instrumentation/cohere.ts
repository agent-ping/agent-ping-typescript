import { wrapAsyncIterable } from "./streaming.js";
import {
  FirstChunkClock,
  contentCaptureEnabled,
  emitErroredLlmCall,
  isPromiseLike,
  messageText,
  putContent,
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
  message?: { toolCalls?: unknown[]; content?: unknown };
}

/** The request fields capture reads. */
interface CohereRequest {
  model?: string;
  messages?: unknown;
}

/** Fields that ride along with the usage: request for capture, stream timing and text. */
interface CallExtras {
  request?: CohereRequest;
  ttftMs?: number;
  outputText?: string;
}

/** Stream events that carry model output rather than housekeeping. */
const CONTENT_EVENTS = new Set([
  "content-start",
  "content-delta",
  "tool-plan-delta",
  "tool-call-start",
  "tool-call-delta",
]);

interface InstrumentableCohereClient {
  chat: (...args: unknown[]) => unknown;
  chatStream?: (...args: unknown[]) => unknown;
  embed?: (...args: unknown[]) => unknown;
}

interface CohereStreamEvent {
  type?: string;
  delta?: { usage?: CohereUsage; finishReason?: string; message?: { content?: { text?: string } } };
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
    const firstArg = args[0] as CohereRequest | undefined;
    const requestedModel = firstArg?.model;

    let result: unknown;
    try {
      result = originalChat(...args);
    } catch (err) {
      emitErroredLlmCall(options, PROVIDER, requestedModel, start, err);
      throw err;
    }

    return wrapResponsePromise(result, options, start, firstArg);
  };

  const wrappedChatStream = originalChatStream
    ? (...args: unknown[]): unknown => {
        const start = Date.now();
        const firstArg = args[0] as CohereRequest | undefined;
        const requestedModel = firstArg?.model;
        try {
          const result = originalChatStream(...args);
          return wrapStreamPromise(result, options, start, firstArg);
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
  request: CohereRequest | undefined,
): unknown {
  if (isPromiseLike(result)) {
    return result.then(
      (stream: unknown) => wrapStreamResult(stream, options, start, request),
      (err: unknown) => {
        emitErroredLlmCall(options, PROVIDER, request?.model, start, err, { stream: true });
        throw err;
      },
    );
  }
  return wrapStreamResult(result, options, start, request);
}

function wrapStreamResult(
  stream: unknown,
  options: InstrumentOptions,
  start: number,
  request: CohereRequest | undefined,
): unknown {
  if (!stream || typeof stream !== "object") return stream;
  if (!(Symbol.asyncIterator in stream)) return stream;

  const requestedModel = request?.model;
  let billed: CohereBilledUnits | undefined;
  let finishReason: string | undefined;
  let toolCalls = 0;
  const clock = new FirstChunkClock();
  const text: string[] = [];
  const capture = contentCaptureEnabled();

  return wrapAsyncIterable(stream as AsyncIterable<CohereStreamEvent>, {
    onChunk(event: CohereStreamEvent): void {
      if (event?.type && CONTENT_EVENTS.has(event.type)) clock.mark();
      if (event?.type === "content-delta" && capture) {
        const piece = event.delta?.message?.content?.text;
        if (typeof piece === "string") text.push(piece);
      }
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
        { request, ttftMs: clock.ttftMs(start), outputText: text.join("") },
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
  request: CohereRequest | undefined,
): unknown {
  const emit = (response: CohereResponse): void =>
    emitLlmCall(options, start, request?.model, response, undefined, {
      request,
      outputText: messageText(response?.message?.content),
    });
  if (!isPromiseLike(result)) {
    emit(result as CohereResponse);
    return result;
  }
  return result.then(
    (response: unknown) => {
      emit(response as CohereResponse);
      return response;
    },
    (err: unknown) => {
      emitErroredLlmCall(options, PROVIDER, request?.model, start, err);
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
  extras: CallExtras = {},
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
    if (extras.ttftMs !== undefined) data["ttft_ms"] = extras.ttftMs;
    if (options.mode === "batch") data["mode"] = "batch";
    // Cohere carries the system prompt inside messages, so it stays in input.
    putContent(data, { input: extras.request?.messages, output: extras.outputText });
    run.event("llm_call", data);
  } catch {
    // swallow
  }
}
