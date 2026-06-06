import { getActiveRun } from "../context.js";
import { wrapAsyncIterable } from "./streaming.js";

interface MistralUsage {
  promptTokens?: number;
  completionTokens?: number;
  totalTokens?: number;
}

interface MistralResponse {
  model?: string;
  usage?: MistralUsage;
}

interface MistralChatNamespace {
  complete: (...args: unknown[]) => unknown;
  stream?: (...args: unknown[]) => unknown;
}

interface MistralClient {
  chat: MistralChatNamespace;
}

interface RunLike {
  event: (type: string, data: Record<string, unknown>) => void;
}

interface InstrumentOptions {
  run?: RunLike;
  mode?: "standard" | "batch";
}

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
  data?: { model?: string; usage?: MistralUsage };
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
  let usage: MistralUsage | undefined;

  return wrapAsyncIterable(stream as AsyncIterable<MistralStreamEvent>, {
    onChunk(event: MistralStreamEvent): void {
      const data = event?.data;
      if (data?.model) model = data.model;
      if (data?.usage) usage = data.usage;
    },
    onDone(): void {
      emitLlmCall(options, start, model, { model, usage });
    },
    onError(err: unknown): void {
      emitFailure(options, start, model, err);
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
      emitFailure(options, start, requestedModel, err);
      throw err;
    },
  );
}

function emitLlmCall(
  options: InstrumentOptions,
  start: number,
  requestedModel: string | undefined,
  response: MistralResponse,
): void {
  const run = options.run ?? getActiveRun();
  if (!run) return;
  try {
    const latencyMs = Date.now() - start;
    const usage = response.usage ?? {};
    const data: Record<string, unknown> = {
      provider: "mistral",
      model: response.model ?? requestedModel ?? "unknown",
      input_tokens: usage.promptTokens ?? 0,
      output_tokens: usage.completionTokens ?? 0,
      latency_ms: latencyMs,
    };
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
      provider: "mistral",
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
