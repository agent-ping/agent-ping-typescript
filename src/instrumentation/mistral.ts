import { wrapAsyncIterable } from "./streaming.js";
import {
  emitErroredLlmCall,
  isPromiseLike,
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
  message?: { toolCalls?: MistralToolCall[] | null };
  delta?: { toolCalls?: MistralToolCall[] | null };
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
    const firstArg = args[0] as { model?: string } | undefined;
    const requestedModel = firstArg?.model;

    let result: unknown;
    try {
      result = originalComplete(...args);
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
  let usage: MistralUsage | undefined;
  let finishReason: string | undefined;
  const toolCallIndexes = new Set<number>();

  return wrapAsyncIterable(stream as AsyncIterable<MistralStreamEvent>, {
    onChunk(event: MistralStreamEvent): void {
      const data = event?.data;
      if (data?.model) model = data.model;
      if (data?.usage) usage = data.usage;
      const choice = data?.choices?.[0];
      if (choice?.finishReason) finishReason = choice.finishReason;
      for (const call of choice?.delta?.toolCalls ?? []) {
        toolCallIndexes.add(call.index ?? toolCallIndexes.size);
      }
    },
    onDone(): void {
      emitLlmCall(options, start, model, { model, usage }, {
        stream: true,
        finishReason,
        toolCalls: toolCallIndexes.size,
      });
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
  requestedModel: string | undefined,
): unknown {
  if (!isPromiseLike(result)) {
    emitLlmCall(options, start, requestedModel, result as MistralResponse);
    return result;
  }
  return result.then(
    (response: unknown) => {
      emitLlmCall(options, start, requestedModel, response as MistralResponse);
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
  response: MistralResponse,
  streamInfo?: { stream: true; finishReason?: string; toolCalls: number },
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
    if (options.mode === "batch") data["mode"] = "batch";
    run.event("llm_call", data);
  } catch {
    // swallow
  }
}
