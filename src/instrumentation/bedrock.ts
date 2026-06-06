/**
 * AWS Bedrock instrumentation.
 *
 * Wraps `BedrockRuntimeClient.send` from `@aws-sdk/client-bedrock-runtime`.
 * Inspects the command type to identify Converse / ConverseStream /
 * InvokeModel / InvokeModelWithResponseStream calls and emits llm_call
 * events with provider="bedrock", the model id, token counts, and latency.
 *
 * The AWS SDK v3 uses a command pattern: client.send(new ConverseCommand({...})).
 * We wrap `send` and detect the command class via its constructor name —
 * which is stable across the AWS SDK and survives minification (the SDK
 * sets `Command.name` explicitly).
 */

import { getActiveRun } from "../context.js";

interface BedrockUsage {
  inputTokens?: number;
  outputTokens?: number;
  totalTokens?: number;
  cacheReadInputTokens?: number;
  cacheWriteInputTokens?: number;
}

interface BedrockConverseResponse {
  usage?: BedrockUsage;
  output?: unknown;
  stopReason?: string;
}

interface BedrockInvokeModelResponse {
  body?: unknown;
  $metadata?: { httpHeaders?: Record<string, string> };
}

interface BedrockConverseStreamResponse {
  stream?: AsyncIterable<{ metadata?: { usage?: BedrockUsage } }>;
}

interface CommandInput {
  modelId?: string;
}

interface AWSCommand {
  input?: CommandInput;
  constructor: { name?: string };
}

interface BedrockRuntimeClientLike {
  send: (command: AWSCommand, ...rest: unknown[]) => unknown;
}

interface RunLike {
  event: (type: string, data: Record<string, unknown>) => void;
}

interface InstrumentOptions {
  run?: RunLike;
}

const CHAT_COMMANDS = new Set(["ConverseCommand", "InvokeModelCommand"]);
const STREAM_COMMANDS = new Set([
  "ConverseStreamCommand",
  "InvokeModelWithResponseStreamCommand",
]);

export function instrumentBedrock<T extends BedrockRuntimeClientLike>(
  client: T,
  options: InstrumentOptions = {},
): T {
  if (!client || typeof client.send !== "function") {
    return client;
  }

  const originalSend = client.send.bind(client);

  const wrappedSend = (command: AWSCommand, ...rest: unknown[]): unknown => {
    const commandName = command?.constructor?.name ?? "";
    const isChat = CHAT_COMMANDS.has(commandName);
    const isStream = STREAM_COMMANDS.has(commandName);

    if (!isChat && !isStream) {
      // Not a cost-bearing operation; pass straight through.
      return originalSend(command, ...rest);
    }

    const start = Date.now();
    const model = command?.input?.modelId ?? "unknown";

    let result: unknown;
    try {
      result = originalSend(command, ...rest);
    } catch (err) {
      emitFailure(options, start, model, err);
      throw err;
    }

    if (isStream) {
      return wrapStreamPromise(result, options, start, model, commandName);
    }
    return wrapChatPromise(result, options, start, model, commandName);
  };

  return new Proxy(client, {
    get(target, prop, receiver) {
      if (prop === "send") return wrappedSend;
      return Reflect.get(target, prop, receiver);
    },
  });
}

function wrapChatPromise(
  result: unknown,
  options: InstrumentOptions,
  start: number,
  model: string,
  commandName: string,
): unknown {
  if (!isPromiseLike(result)) {
    emitChat(options, start, model, commandName, result as BedrockConverseResponse | BedrockInvokeModelResponse);
    return result;
  }
  return result.then(
    (response: unknown) => {
      emitChat(options, start, model, commandName, response as BedrockConverseResponse | BedrockInvokeModelResponse);
      return response;
    },
    (err: unknown) => {
      emitFailure(options, start, model, err);
      throw err;
    },
  );
}

function emitChat(
  options: InstrumentOptions,
  start: number,
  model: string,
  commandName: string,
  response: BedrockConverseResponse | BedrockInvokeModelResponse,
): void {
  const run = options.run ?? getActiveRun();
  if (!run) return;
  try {
    const latencyMs = Date.now() - start;
    const data: Record<string, unknown> = {
      provider: "bedrock",
      model,
      latency_ms: latencyMs,
    };

    if (commandName === "ConverseCommand") {
      const usage = (response as BedrockConverseResponse).usage ?? {};
      const input = usage.inputTokens ?? 0;
      const cached = usage.cacheReadInputTokens ?? 0;
      data["input_tokens"] = cached > 0 ? Math.max(0, input - cached) : input;
      data["output_tokens"] = usage.outputTokens ?? 0;
      if (cached > 0) data["cached_input_tokens"] = cached;
      if (usage.cacheWriteInputTokens) data["cache_creation_input_tokens"] = usage.cacheWriteInputTokens;
    } else {
      // InvokeModel: tokens come from $metadata.httpHeaders.
      const headers = (response as BedrockInvokeModelResponse).$metadata?.httpHeaders ?? {};
      const input = Number(headers["x-amzn-bedrock-input-token-count"] ?? 0);
      const output = Number(headers["x-amzn-bedrock-output-token-count"] ?? 0);
      if (!Number.isNaN(input)) data["input_tokens"] = input;
      if (!Number.isNaN(output)) data["output_tokens"] = output;
    }

    run.event("llm_call", data);
  } catch {
    // swallow
  }
}

function wrapStreamPromise(
  result: unknown,
  options: InstrumentOptions,
  start: number,
  model: string,
  commandName: string,
): unknown {
  if (isPromiseLike(result)) {
    return result.then(
      (response: unknown) => wrapStreamResult(response, options, start, model, commandName),
      (err: unknown) => {
        emitFailure(options, start, model, err);
        throw err;
      },
    );
  }
  return wrapStreamResult(result, options, start, model, commandName);
}

function wrapStreamResult(
  response: unknown,
  options: InstrumentOptions,
  start: number,
  model: string,
  commandName: string,
): unknown {
  if (!response || typeof response !== "object") return response;
  // ConverseStreamCommand returns { stream: AsyncIterable<...> }
  // InvokeModelWithResponseStreamCommand returns { body: AsyncIterable<...> }
  const streamKey = commandName === "ConverseStreamCommand" ? "stream" : "body";
  const stream = (response as Record<string, unknown>)[streamKey];
  if (!stream || typeof stream !== "object") return response;
  if (!(Symbol.asyncIterator in stream)) return response;

  let lastUsage: BedrockUsage | undefined;

  const wrapped: AsyncIterable<unknown> = {
    async *[Symbol.asyncIterator]() {
      try {
        for await (const event of stream as AsyncIterable<{ metadata?: { usage?: BedrockUsage } }>) {
          if (event && typeof event === "object" && "metadata" in event && event.metadata?.usage) {
            lastUsage = event.metadata.usage;
          }
          yield event;
        }
      } finally {
        emitStreamEnd(options, start, model, lastUsage);
      }
    },
  };

  return new Proxy(response as object, {
    get(target, prop, receiver) {
      if (prop === streamKey) return wrapped;
      return Reflect.get(target, prop, receiver);
    },
  });
}

function emitStreamEnd(
  options: InstrumentOptions,
  start: number,
  model: string,
  usage: BedrockUsage | undefined,
): void {
  const run = options.run ?? getActiveRun();
  if (!run) return;
  try {
    const latencyMs = Date.now() - start;
    const data: Record<string, unknown> = {
      provider: "bedrock",
      model,
      latency_ms: latencyMs,
    };
    if (usage) {
      const input = usage.inputTokens ?? 0;
      const cached = usage.cacheReadInputTokens ?? 0;
      data["input_tokens"] = cached > 0 ? Math.max(0, input - cached) : input;
      data["output_tokens"] = usage.outputTokens ?? 0;
      if (cached > 0) data["cached_input_tokens"] = cached;
    }
    run.event("llm_call", data);
  } catch {
    // swallow
  }
}

function emitFailure(
  options: InstrumentOptions,
  start: number,
  model: string,
  err: unknown,
): void {
  const run = options.run ?? getActiveRun();
  if (!run) return;
  try {
    run.event("llm_call_error", {
      provider: "bedrock",
      model,
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
