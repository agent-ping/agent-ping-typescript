import { wrapAsyncIterable } from "./streaming.js";
import {
  emitErroredLlmCall,
  isPromiseLike,
  putIfPositive,
  putIfString,
  resolveRun,
  type InstrumentOptions as BaseInstrumentOptions,
} from "./shared.js";

/* Chat Completions API shapes. */

interface OpenAIUsage {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
  prompt_tokens_details?: { cached_tokens?: number };
  completion_tokens_details?: { reasoning_tokens?: number };
}

interface OpenAIToolCall {
  index?: number;
  id?: string;
}

interface OpenAIChoice {
  finish_reason?: string | null;
  message?: { tool_calls?: OpenAIToolCall[] };
  delta?: { tool_calls?: OpenAIToolCall[] };
}

interface OpenAIResponse {
  model?: string;
  usage?: OpenAIUsage;
  choices?: OpenAIChoice[];
}

interface OpenAIStreamChunk {
  model?: string;
  usage?: OpenAIUsage;
  choices?: OpenAIChoice[];
}

/* Responses API shapes. */

interface ResponsesUsage {
  input_tokens?: number;
  output_tokens?: number;
  total_tokens?: number;
  input_tokens_details?: { cached_tokens?: number };
  output_tokens_details?: { reasoning_tokens?: number };
}

interface ResponsesOutputItem {
  type?: string;
}

interface ResponsesResponse {
  model?: string;
  usage?: ResponsesUsage;
  output?: ResponsesOutputItem[];
  status?: string;
  incomplete_details?: { reason?: string } | null;
}

interface ResponsesStreamEvent {
  type?: string;
  response?: ResponsesResponse;
}

interface CreateNamespace {
  create: (...args: unknown[]) => unknown;
}

interface OpenAIClient {
  chat?: { completions?: CreateNamespace };
  responses?: CreateNamespace;
}

export interface InstrumentOptions extends BaseInstrumentOptions {
  /** Report calls as batch-priced. */
  mode?: "standard" | "batch";
}

/** Common fields extracted from either API, ready to become an `llm_call`. */
interface CallSummary {
  model?: string;
  inputTokens: number;
  outputTokens: number;
  cachedTokens: number;
  reasoningTokens: number;
  finishReason?: string;
  toolCalls: number;
}

/**
 * Instrument an OpenAI client. Wraps `chat.completions.create` and
 * `responses.create`, streamed or not, and emits one `llm_call` per
 * request. The instrumented client is what the OpenAI Agents SDK should
 * be given through `setDefaultOpenAIClient`, so agent model calls are
 * priced from the same place.
 */
export function instrumentOpenAI<T extends OpenAIClient>(
  client: T,
  options: InstrumentOptions = {},
): T {
  if (!client || typeof client !== "object") return client;

  const chatCompletions = client.chat?.completions;
  const responses = client.responses;
  if (!chatCompletions && !responses) return client;

  const wrappedChatCreate = chatCompletions
    ? wrapCreate(chatCompletions, options, chatSummary, chatStreamSummary)
    : undefined;
  const wrappedResponsesCreate = responses
    ? wrapCreate(responses, options, responsesSummary, responsesStreamSummary)
    : undefined;

  const proxy = new Proxy(client, {
    get(target, prop, receiver) {
      if (prop === "chat" && wrappedChatCreate && target.chat) {
        return new Proxy(target.chat, {
          get(t1, p1, r1) {
            if (p1 === "completions" && t1.completions) {
              return new Proxy(t1.completions, {
                get(t2, p2, r2) {
                  if (p2 === "create") return wrappedChatCreate;
                  return Reflect.get(t2, p2, r2);
                },
              });
            }
            return Reflect.get(t1, p1, r1);
          },
        });
      }
      if (prop === "responses" && wrappedResponsesCreate && target.responses) {
        return new Proxy(target.responses, {
          get(t1, p1, r1) {
            if (p1 === "create") return wrappedResponsesCreate;
            return Reflect.get(t1, p1, r1);
          },
        });
      }
      return Reflect.get(target, prop, receiver);
    },
  });

  return proxy;
}

type Summarise<R> = (response: R, requestedModel: string | undefined) => CallSummary;
type StreamSummariser<C> = () => {
  onChunk: (chunk: C) => void;
  summary: (requestedModel: string | undefined) => CallSummary;
};

function wrapCreate<R, C>(
  namespace: CreateNamespace,
  options: InstrumentOptions,
  summarise: Summarise<R>,
  streamSummariser: StreamSummariser<C>,
): (...args: unknown[]) => unknown {
  const originalCreate = namespace.create.bind(namespace);

  return (...args: unknown[]): unknown => {
    const start = Date.now();
    const firstArg = args[0] as { stream?: boolean; model?: string } | undefined;
    const isStream = Boolean(firstArg?.stream);
    const result = originalCreate(...args);
    if (isStream) {
      return wrapStreamPromise(result, options, start, firstArg?.model, streamSummariser);
    }
    return wrapResponsePromise(result, options, start, firstArg?.model, summarise);
  };
}

function wrapResponsePromise<R>(
  result: unknown,
  options: InstrumentOptions,
  start: number,
  requestedModel: string | undefined,
  summarise: Summarise<R>,
): unknown {
  if (!isPromiseLike(result)) return result;
  return result.then(
    (response: unknown) => {
      emitLlmCall(options, start, summarise(response as R, requestedModel), false);
      return response;
    },
    (err: unknown) => {
      emitErroredLlmCall(options, "openai", requestedModel, start, err);
      throw err;
    },
  );
}

function wrapStreamPromise<C>(
  result: unknown,
  options: InstrumentOptions,
  start: number,
  requestedModel: string | undefined,
  streamSummariser: StreamSummariser<C>,
): unknown {
  if (!isPromiseLike(result)) {
    return wrapStreamResult(result, options, start, requestedModel, streamSummariser);
  }
  return result.then(
    (stream: unknown) =>
      wrapStreamResult(stream, options, start, requestedModel, streamSummariser),
    (err: unknown) => {
      emitErroredLlmCall(options, "openai", requestedModel, start, err, { stream: true });
      throw err;
    },
  );
}

function wrapStreamResult<C>(
  stream: unknown,
  options: InstrumentOptions,
  start: number,
  requestedModel: string | undefined,
  streamSummariser: StreamSummariser<C>,
): unknown {
  if (!stream || typeof stream !== "object") return stream;
  if (!(Symbol.asyncIterator in stream)) return stream;

  const collector = streamSummariser();

  const observer = {
    onChunk(chunk: C): void {
      try {
        collector.onChunk(chunk);
      } catch {
        // swallow
      }
    },
    onDone(): void {
      emitLlmCall(options, start, collector.summary(requestedModel), true);
    },
    onError(err: unknown): void {
      const model = collector.summary(requestedModel).model;
      emitErroredLlmCall(options, "openai", model, start, err, { stream: true });
    },
  };

  const wrappedIterable = wrapAsyncIterable(stream as AsyncIterable<C>, observer);

  return new Proxy(stream as object, {
    get(target, prop, receiver) {
      if (prop === Symbol.asyncIterator) {
        return wrappedIterable[Symbol.asyncIterator].bind(wrappedIterable);
      }
      return Reflect.get(target, prop, receiver);
    },
  });
}

/* Chat Completions extraction. */

function chatSummary(response: OpenAIResponse, requestedModel: string | undefined): CallSummary {
  const usage = response.usage ?? {};
  const choice = response.choices?.[0];
  return {
    model: response.model ?? requestedModel,
    // prompt_tokens is gross: it already includes the cached prefix.
    inputTokens: usage.prompt_tokens ?? 0,
    outputTokens: usage.completion_tokens ?? 0,
    cachedTokens: usage.prompt_tokens_details?.cached_tokens ?? 0,
    reasoningTokens: usage.completion_tokens_details?.reasoning_tokens ?? 0,
    finishReason: choice?.finish_reason ?? undefined,
    toolCalls: choice?.message?.tool_calls?.length ?? 0,
  };
}

function chatStreamSummary(): ReturnType<StreamSummariser<OpenAIStreamChunk>> {
  let model: string | undefined;
  let usage: OpenAIUsage | undefined;
  let finishReason: string | undefined;
  // Tool calls stream as deltas keyed by index; count distinct indexes.
  const toolCallIndexes = new Set<number>();
  return {
    onChunk(chunk) {
      if (chunk?.model) model = chunk.model;
      // Usage arrives on the final chunk when stream_options.include_usage is set.
      if (chunk?.usage) usage = chunk.usage;
      const choice = chunk?.choices?.[0];
      if (choice?.finish_reason) finishReason = choice.finish_reason;
      for (const call of choice?.delta?.tool_calls ?? []) {
        toolCallIndexes.add(call.index ?? toolCallIndexes.size);
      }
    },
    summary(requestedModel) {
      return {
        model: model ?? requestedModel,
        inputTokens: usage?.prompt_tokens ?? 0,
        outputTokens: usage?.completion_tokens ?? 0,
        cachedTokens: usage?.prompt_tokens_details?.cached_tokens ?? 0,
        reasoningTokens: usage?.completion_tokens_details?.reasoning_tokens ?? 0,
        finishReason,
        toolCalls: toolCallIndexes.size,
      };
    },
  };
}

/* Responses API extraction. */

function responsesSummary(
  response: ResponsesResponse,
  requestedModel: string | undefined,
): CallSummary {
  const usage = response.usage ?? {};
  const toolCalls = (response.output ?? []).filter((item) => isToolCallItem(item)).length;
  return {
    model: response.model ?? requestedModel,
    // input_tokens is gross: it already includes the cached prefix.
    inputTokens: usage.input_tokens ?? 0,
    outputTokens: usage.output_tokens ?? 0,
    cachedTokens: usage.input_tokens_details?.cached_tokens ?? 0,
    reasoningTokens: usage.output_tokens_details?.reasoning_tokens ?? 0,
    finishReason: responsesFinishReason(response, toolCalls),
    toolCalls,
  };
}

function responsesStreamSummary(): ReturnType<StreamSummariser<ResponsesStreamEvent>> {
  let final: ResponsesResponse | undefined;
  return {
    onChunk(event) {
      // The terminal event carries the whole response, usage included.
      if (
        event?.response &&
        (event.type === "response.completed" ||
          event.type === "response.incomplete" ||
          event.type === "response.failed")
      ) {
        final = event.response;
      } else if (event?.type === "response.created" && event.response?.model) {
        final = final ?? { model: event.response.model };
      }
    },
    summary(requestedModel) {
      return responsesSummary(final ?? {}, requestedModel);
    },
  };
}

/**
 * The Responses API reports a status instead of a finish reason. Map it
 * onto the Chat Completions vocabulary so both APIs read the same on the
 * timeline: an incomplete response carries its reason (for example
 * `max_output_tokens`), a complete one is `tool_calls` when the model
 * asked for tools and `stop` otherwise.
 */
function responsesFinishReason(
  response: ResponsesResponse,
  toolCalls: number,
): string | undefined {
  if (response.status === "incomplete") {
    return response.incomplete_details?.reason ?? "incomplete";
  }
  if (response.status === "completed") {
    return toolCalls > 0 ? "tool_calls" : "stop";
  }
  return response.status ?? undefined;
}

function isToolCallItem(item: ResponsesOutputItem): boolean {
  const type = item?.type ?? "";
  return (
    type === "function_call" ||
    type === "custom_tool_call" ||
    type === "computer_call" ||
    type === "mcp_call" ||
    type === "code_interpreter_call" ||
    type === "file_search_call" ||
    type === "web_search_call" ||
    type === "image_generation_call" ||
    type === "local_shell_call"
  );
}

/* Emission. */

function emitLlmCall(
  options: InstrumentOptions,
  start: number,
  summary: CallSummary,
  stream: boolean,
): void {
  const run = resolveRun(options);
  if (!run) return;
  try {
    const data: Record<string, unknown> = {
      provider: "openai",
      model: summary.model ?? "unknown",
      input_tokens: summary.inputTokens,
      output_tokens: summary.outputTokens,
      latency_ms: Date.now() - start,
    };
    putIfPositive(data, "cached_input_tokens", summary.cachedTokens);
    putIfPositive(data, "reasoning_tokens", summary.reasoningTokens);
    putIfString(data, "finish_reason", summary.finishReason);
    if (summary.toolCalls > 0) data["tool_calls"] = summary.toolCalls;
    if (stream) data["stream"] = true;
    if (options.mode === "batch") data["mode"] = "batch";
    run.event("llm_call", data);
  } catch {
    // swallow
  }
}
