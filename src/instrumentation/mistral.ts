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

interface MistralUsage {
  promptTokens?: number;
  completionTokens?: number;
  totalTokens?: number;
}

interface MistralToolCall {
  index?: number;
  id?: string;
}

interface MistralChoice {
  finishReason?: string | null;
  message?: { toolCalls?: MistralToolCall[] | null; content?: unknown };
  delta?: { toolCalls?: MistralToolCall[] | null; content?: unknown };
}

/** The request fields capture reads. */
interface MistralRequest {
  model?: string;
  messages?: unknown;
}

/** Fields that ride along with the usage: request for capture, stream timing and text. */
interface CallExtras {
  request?: MistralRequest;
  ttftMs?: number;
  outputText?: string;
}

interface MistralResponse {
  model?: string;
  usage?: MistralUsage;
  choices?: MistralChoice[];
}

interface MistralChatNamespace {
  complete: (...args: unknown[]) => unknown;
  stream?: (...args: unknown[]) => unknown;
}

interface MistralClient {
  chat: MistralChatNamespace;
}

export interface InstrumentOptions extends BaseInstrumentOptions {
  mode?: "standard" | "batch";
}

const PROVIDER = "mistral";

export function instrumentMistral<T extends MistralClient>(
  client: T,
  options: InstrumentOptions = {},
): T {
  if (!client || typeof client !== "object" || !client.chat) {
    return client;
  }

  const originalComplete = client.chat.complete.bind(client.chat);
  const originalStream = client.chat.stream?.bind(client.chat);

  const wrappedComplete = (...args: unknown[]): unknown => {
    const start = Date.now();
    const firstArg = args[0] as MistralRequest | undefined;
    const requestedModel = firstArg?.model;

    let result: unknown;
    try {
      result = originalComplete(...args);
    } catch (err) {
      emitErroredLlmCall(options, PROVIDER, requestedModel, start, err);
      throw err;
    }

    return wrapResponsePromise(result, options, start, firstArg);
  };

  const wrappedStream = originalStream
    ? (...args: unknown[]): unknown => {
        const start = Date.now();
        const firstArg = args[0] as MistralRequest | undefined;
        const requestedModel = firstArg?.model;
        try {
          const result = originalStream(...args);
          return wrapStreamPromise(result, options, start, firstArg);
        } catch (err) {
          emitErroredLlmCall(options, PROVIDER, requestedModel, start, err, { stream: true });
          throw err;
        }
      }
    : undefined;

  return new Proxy(client, {
    get(target, prop, receiver) {
      if (prop === "chat") {
        return new Proxy(target.chat, {
          get(t, p, r) {
            if (p === "complete") return wrappedComplete;
            if (p === "stream" && wrappedStream) return wrappedStream;
            return Reflect.get(t, p, r);
          },
        });
      }
      return Reflect.get(target, prop, receiver);
    },
  });
}

interface MistralStreamEvent {
  data?: MistralResponse;
}

function wrapStreamPromise(
  result: unknown,
  options: InstrumentOptions,
  start: number,
  request: MistralRequest | undefined,
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
  request: MistralRequest | undefined,
): unknown {
  if (!stream || typeof stream !== "object") return stream;
  if (!(Symbol.asyncIterator in stream)) return stream;

  let model: string | undefined = request?.model;
  let usage: MistralUsage | undefined;
  let finishReason: string | undefined;
  const toolCallIndexes = new Set<number>();
  const clock = new FirstChunkClock();
  const text: string[] = [];
  const capture = contentCaptureEnabled();

  return wrapAsyncIterable(stream as AsyncIterable<MistralStreamEvent>, {
    onChunk(event: MistralStreamEvent): void {
      const data = event?.data;
      if (data?.model) model = data.model;
      if (data?.usage) usage = data.usage;
      const choice = data?.choices?.[0];
      if (choice?.finishReason) finishReason = choice.finishReason;
      const toolCalls = choice?.delta?.toolCalls ?? [];
      for (const call of toolCalls) {
        toolCallIndexes.add(call.index ?? toolCallIndexes.size);
      }
      const content = choice?.delta?.content;
      const hasText = typeof content === "string" ? content.length > 0 : Array.isArray(content) && content.length > 0;
      if (hasText || toolCalls.length > 0) clock.mark();
      if (capture && hasText) text.push(messageText(content));
    },
    onDone(): void {
      emitLlmCall(
        options,
        start,
        model,
        { model, usage },
        { stream: true, finishReason, toolCalls: toolCallIndexes.size },
        { request, ttftMs: clock.ttftMs(start), outputText: text.join("") },
      );
    },
    onError(err: unknown): void {
      emitErroredLlmCall(options, PROVIDER, model, start, err, { stream: true });
    },
  });
}

function wrapResponsePromise(
  result: unknown,
  options: InstrumentOptions,
  start: number,
  request: MistralRequest | undefined,
): unknown {
  const emit = (response: MistralResponse): void =>
    emitLlmCall(options, start, request?.model, response, undefined, {
      request,
      outputText: messageText(response?.choices?.[0]?.message?.content),
    });
  if (!isPromiseLike(result)) {
    emit(result as MistralResponse);
    return result;
  }
  return result.then(
    (response: unknown) => {
      emit(response as MistralResponse);
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
  response: MistralResponse,
  streamInfo?: { stream: true; finishReason?: string; toolCalls: number },
  extras: CallExtras = {},
): void {
  const run = resolveRun(options);
  if (!run) return;
  try {
    const latencyMs = Date.now() - start;
    const usage = response.usage ?? {};
    const choice = response.choices?.[0];
    const data: Record<string, unknown> = {
      provider: PROVIDER,
      model: response.model ?? requestedModel ?? "unknown",
      input_tokens: usage.promptTokens ?? 0,
      output_tokens: usage.completionTokens ?? 0,
      latency_ms: latencyMs,
    };
    putIfString(data, "finish_reason", streamInfo?.finishReason ?? choice?.finishReason);
    const toolCalls = streamInfo ? streamInfo.toolCalls : choice?.message?.toolCalls?.length ?? 0;
    if (toolCalls > 0) data["tool_calls"] = toolCalls;
    if (streamInfo) data["stream"] = true;
    if (extras.ttftMs !== undefined) data["ttft_ms"] = extras.ttftMs;
    if (options.mode === "batch") data["mode"] = "batch";
    // Mistral carries the system prompt inside messages, so it stays in input.
    putContent(data, { input: extras.request?.messages, output: extras.outputText });
    run.event("llm_call", data);
  } catch {
    // swallow
  }
}
