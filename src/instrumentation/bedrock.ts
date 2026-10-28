/**
 * AWS Bedrock instrumentation.
 *
 * Wraps `BedrockRuntimeClient.send` from `@aws-sdk/client-bedrock-runtime`.
 * Inspects the command type to identify Converse / ConverseStream /
 * InvokeModel / InvokeModelWithResponseStream calls and emits llm_call
 * events with provider="bedrock", the model id, token counts, and latency.
 *
 * The AWS SDK v3 uses a command pattern: client.send(new ConverseCommand({...})).
 * We wrap `send` and detect the command class via its constructor name,
 * which is stable across the AWS SDK and survives minification (the SDK
 * sets `Command.name` explicitly).
 *
 * ConverseStream reports `ttft_ms` from the first content block event. With
 * `init({ captureLlmContent: true })` Converse and ConverseStream also carry
 * `messages` as `input`, `system` as `system_instructions` and the reply's
 * text blocks as `output`. InvokeModel bodies are provider-specific bytes
 * and are not captured.
 */

import {
  FirstChunkClock,
  contentCaptureEnabled,
  emitErroredLlmCall,
  isPromiseLike,
  putContent,
  putIfPositive,
  putIfString,
  resolveRun,
  stringifyPayload,
  type InstrumentOptions,
  type WireMessage,
} from "./shared.js";

interface BedrockUsage {
  inputTokens?: number;
  outputTokens?: number;
  totalTokens?: number;
  cacheReadInputTokens?: number;
  cacheWriteInputTokens?: number;
}

interface BedrockContentBlock {
  toolUse?: unknown;
  text?: string;
}

interface BedrockConverseResponse {
  usage?: BedrockUsage;
  output?: { message?: { content?: BedrockContentBlock[] } };
  stopReason?: string;
}

interface BedrockInvokeModelResponse {
  body?: unknown;
  $metadata?: { httpHeaders?: Record<string, string> };
}

interface BedrockStreamEvent {
  metadata?: { usage?: BedrockUsage };
  messageStop?: { stopReason?: string };
  contentBlockStart?: { start?: { toolUse?: unknown } };
  contentBlockDelta?: { delta?: { text?: string; toolUse?: unknown } };
}

interface CommandInput {
  modelId?: string;
  messages?: unknown;
  system?: unknown;
}

interface AWSCommand {
  input?: CommandInput;
  constructor: { name?: string };
}

interface BedrockRuntimeClientLike {
  send: (command: AWSCommand, ...rest: unknown[]) => unknown;
}

const PROVIDER = "bedrock";

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
      emitErroredLlmCall(options, PROVIDER, model, start, err, isStream ? { stream: true } : {});
      throw err;
    }

    if (isStream) {
      return wrapStreamPromise(result, options, start, model, commandName, command?.input);
    }
    return wrapChatPromise(result, options, start, model, commandName, command?.input);
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
  input: CommandInput | undefined,
): unknown {
  if (!isPromiseLike(result)) {
    emitChat(options, start, model, commandName, result as BedrockConverseResponse | BedrockInvokeModelResponse, input);
    return result;
  }
  return result.then(
    (response: unknown) => {
      emitChat(options, start, model, commandName, response as BedrockConverseResponse | BedrockInvokeModelResponse, input);
      return response;
    },
    (err: unknown) => {
      emitErroredLlmCall(options, PROVIDER, model, start, err);
      throw err;
    },
  );
}

/**
 * Converse reports inputTokens net of cache reads and writes, as the
 * underlying Anthropic models do. Ingest prices from gross input, so add
 * them back and report the cached parts alongside.
 */
function putUsage(data: Record<string, unknown>, usage: BedrockUsage): void {
  const cached = usage.cacheReadInputTokens ?? 0;
  const cacheWrite = usage.cacheWriteInputTokens ?? 0;
  data["input_tokens"] = (usage.inputTokens ?? 0) + cached + cacheWrite;
  data["output_tokens"] = usage.outputTokens ?? 0;
  putIfPositive(data, "cached_input_tokens", cached);
  putIfPositive(data, "cache_creation_input_tokens", cacheWrite);
}

/**
 * Converse content blocks as text: `text` blocks verbatim, anything else
 * (toolUse, toolResult, image, document) as a bracketed marker.
 */
function blocksText(blocks: unknown): string {
  if (typeof blocks === "string") return blocks;
  if (!Array.isArray(blocks)) return "";
  return blocks
    .map((block) => {
      if (typeof block === "string") return block;
      if (typeof block !== "object" || block === null) return stringifyPayload(block, 200) ?? "";
      const record = block as Record<string, unknown>;
      if (typeof record["text"] === "string") return record["text"];
      const kind = Object.keys(record)[0];
      if (kind === undefined) return "";
      const inner = record[kind];
      const name = typeof inner === "object" && inner !== null ? (inner as { name?: unknown }).name : undefined;
      return `[${kind}${typeof name === "string" ? ` ${name}` : ""}]`;
    })
    .join("\n");
}

function renderMessages(messages: unknown): WireMessage[] | undefined {
  if (!Array.isArray(messages)) return undefined;
  return messages
    .filter((m) => typeof m === "object" && m !== null)
    .map((m) => {
      const record = m as { role?: unknown; content?: unknown };
      return {
        role: typeof record.role === "string" && record.role ? record.role : "user",
        content: blocksText(record.content),
      };
    });
}

function responseText(response: BedrockConverseResponse): string {
  return (response.output?.message?.content ?? [])
    .filter((block) => typeof block?.text === "string")
    .map((block) => block.text as string)
    .join("");
}

function putConverseContent(data: Record<string, unknown>, input: CommandInput | undefined, output: string): void {
  putContent(data, {
    input: renderMessages(input?.messages),
    system: input?.system ? blocksText(input.system) : undefined,
    output,
  });
}

function emitChat(
  options: InstrumentOptions,
  start: number,
  model: string,
  commandName: string,
  response: BedrockConverseResponse | BedrockInvokeModelResponse,
  input: CommandInput | undefined,
): void {
  const run = resolveRun(options);
  if (!run) return;
  try {
    const latencyMs = Date.now() - start;
    const data: Record<string, unknown> = {
      provider: PROVIDER,
      model,
      latency_ms: latencyMs,
    };

    if (commandName === "ConverseCommand") {
      const converse = response as BedrockConverseResponse;
      putUsage(data, converse.usage ?? {});
      putIfString(data, "finish_reason", converse.stopReason);
      const toolCalls = (converse.output?.message?.content ?? []).filter(
        (block) => block?.toolUse !== undefined,
      ).length;
      if (toolCalls > 0) data["tool_calls"] = toolCalls;
      putConverseContent(data, input, responseText(converse));
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
  input: CommandInput | undefined,
): unknown {
  if (isPromiseLike(result)) {
    return result.then(
      (response: unknown) => wrapStreamResult(response, options, start, model, commandName, input),
      (err: unknown) => {
        emitErroredLlmCall(options, PROVIDER, model, start, err, { stream: true });
        throw err;
      },
    );
  }
  return wrapStreamResult(result, options, start, model, commandName, input);
}

function wrapStreamResult(
  response: unknown,
  options: InstrumentOptions,
  start: number,
  model: string,
  commandName: string,
  input: CommandInput | undefined,
): unknown {
  if (!response || typeof response !== "object") return response;
  // ConverseStreamCommand returns { stream: AsyncIterable<...> }
  // InvokeModelWithResponseStreamCommand returns { body: AsyncIterable<...> }
  const streamKey = commandName === "ConverseStreamCommand" ? "stream" : "body";
  const stream = (response as Record<string, unknown>)[streamKey];
  if (!stream || typeof stream !== "object") return response;
  if (!(Symbol.asyncIterator in stream)) return response;

  let lastUsage: BedrockUsage | undefined;
  let stopReason: string | undefined;
  let toolCalls = 0;
  const clock = new FirstChunkClock();
  const text: string[] = [];
  const capture = contentCaptureEnabled();
  const converse = commandName === "ConverseStreamCommand";

  const wrapped: AsyncIterable<unknown> = {
    async *[Symbol.asyncIterator]() {
      let failed = false;
      try {
        for await (const event of stream as AsyncIterable<BedrockStreamEvent>) {
          if (event && typeof event === "object") {
            if (event.metadata?.usage) lastUsage = event.metadata.usage;
            if (event.messageStop?.stopReason) stopReason = event.messageStop.stopReason;
            if (event.contentBlockStart?.start?.toolUse !== undefined) toolCalls += 1;
            if (event.contentBlockStart !== undefined || event.contentBlockDelta !== undefined) clock.mark();
            const piece = event.contentBlockDelta?.delta?.text;
            if (capture && typeof piece === "string") text.push(piece);
          }
          yield event;
        }
      } catch (err) {
        failed = true;
        emitErroredLlmCall(options, PROVIDER, model, start, err, { stream: true });
        throw err;
      } finally {
        if (!failed) {
          emitStreamEnd(options, start, model, lastUsage, stopReason, toolCalls, {
            ttftMs: converse ? clock.ttftMs(start) : undefined,
            input: converse ? input : undefined,
            outputText: converse ? text.join("") : undefined,
          });
        }
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
  stopReason: string | undefined,
  toolCalls: number,
  extras: { ttftMs?: number; input?: CommandInput; outputText?: string } = {},
): void {
  const run = resolveRun(options);
  if (!run) return;
  try {
    const latencyMs = Date.now() - start;
    const data: Record<string, unknown> = {
      provider: PROVIDER,
      model,
      latency_ms: latencyMs,
      stream: true,
    };
    if (usage) putUsage(data, usage);
    putIfString(data, "finish_reason", stopReason);
    if (toolCalls > 0) data["tool_calls"] = toolCalls;
    if (extras.ttftMs !== undefined) data["ttft_ms"] = extras.ttftMs;
    if (extras.input !== undefined || extras.outputText !== undefined) {
      putConverseContent(data, extras.input, extras.outputText ?? "");
    }
    run.event("llm_call", data);
  } catch {
    // swallow
  }
}
