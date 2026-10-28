import { wrapAsyncIterable } from "./streaming.js";
import {
  FirstChunkClock,
  contentCaptureEnabled,
  emitErroredLlmCall,
  isPromiseLike,
  messageText,
  putContent,
  putIfPositive,
  putIfString,
  resolveRun,
  type InstrumentOptions as BaseInstrumentOptions,
  type WireMessage,
} from "./shared.js";

interface GeminiUsage {
  promptTokenCount?: number;
  candidatesTokenCount?: number;
  cachedContentTokenCount?: number;
  thoughtsTokenCount?: number;
  totalTokenCount?: number;
}

interface GeminiPart {
  text?: string;
  thought?: boolean;
  functionCall?: { name?: string } | unknown;
  functionResponse?: { name?: string } | unknown;
  inlineData?: unknown;
  fileData?: unknown;
  executableCode?: unknown;
  codeExecutionResult?: unknown;
}

/** The request fields capture reads. */
interface GeminiRequest {
  model?: string;
  contents?: unknown;
  config?: { systemInstruction?: unknown } | null;
}

/** Fields that ride along with the usage: request for capture, stream timing and text. */
interface CallExtras {
  request?: GeminiRequest;
  ttftMs?: number;
  outputText?: string;
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
    const firstArg = args[0] as GeminiRequest | undefined;
    const requestedModel = firstArg?.model;

    let result: unknown;
    try {
      result = originalGenerate(...args);
    } catch (err) {
      emitErroredLlmCall(options, PROVIDER, requestedModel, start, err);
      throw err;
    }

    return wrapResponsePromise(result, options, start, firstArg);
  };

  const wrappedStream = originalStream
    ? (...args: unknown[]): unknown => {
        const start = Date.now();
        const firstArg = args[0] as GeminiRequest | undefined;
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
  request: GeminiRequest | undefined,
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
  request: GeminiRequest | undefined,
): unknown {
  if (!stream || typeof stream !== "object") return stream;
  if (!(Symbol.asyncIterator in stream)) return stream;

  let model: string | undefined = request?.model;
  let usage: GeminiUsage | undefined;
  let finishReason: string | undefined;
  let toolCalls = 0;
  const clock = new FirstChunkClock();
  const text: string[] = [];
  const capture = contentCaptureEnabled();

  return wrapAsyncIterable(stream as AsyncIterable<GeminiStreamChunk>, {
    onChunk(chunk: GeminiStreamChunk): void {
      if (hasContent(chunk)) clock.mark();
      if (capture) text.push(textOf(chunk));
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
        { request, ttftMs: clock.ttftMs(start), outputText: text.join("") },
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
  request: GeminiRequest | undefined,
): unknown {
  const emit = (response: GeminiResponse): void =>
    emitLlmCall(options, start, request?.model, response, undefined, {
      request,
      outputText: textOf(response),
    });
  if (!isPromiseLike(result)) {
    emit(result as GeminiResponse);
    return result;
  }
  return result.then(
    (response: unknown) => {
      emit(response as GeminiResponse);
      return response;
    },
    (err: unknown) => {
      emitErroredLlmCall(options, PROVIDER, request?.model, start, err);
      throw err;
    },
  );
}

function countFunctionCalls(candidate: GeminiCandidate | undefined): number {
  return (candidate?.content?.parts ?? []).filter((p) => p?.functionCall !== undefined).length;
}

const ROLE_MAP: Record<string, string> = { model: "assistant" };

function asRecord(value: unknown): Record<string, unknown> | undefined {
  return typeof value === "object" && value !== null ? (value as Record<string, unknown>) : undefined;
}

/** One Part as text: its text, or a bracketed marker for anything else. */
function partText(part: unknown): string {
  if (typeof part === "string") return part;
  const record = asRecord(part);
  if (!record) return messageText(part);
  if (typeof record["text"] === "string") return record["text"];
  for (const kind of ["functionCall", "functionResponse"]) {
    if (record[kind] !== undefined && record[kind] !== null) {
      const name = asRecord(record[kind])?.["name"];
      return `[${kind}${typeof name === "string" ? ` ${name}` : ""}]`;
    }
  }
  for (const kind of ["inlineData", "fileData", "executableCode", "codeExecutionResult"]) {
    if (record[kind] !== undefined && record[kind] !== null) return `[${kind}]`;
  }
  return messageText(part);
}

/**
 * `contents` as the wire's [{role, content}]. A string, or a list of strings
 * and parts, is one user message; a list of Content objects is a
 * conversation, with Gemini's `model` role reported as `assistant`.
 */
function renderContents(contents: unknown): WireMessage[] | undefined {
  if (contents === undefined || contents === null) return undefined;
  if (typeof contents === "string") return [{ role: "user", content: contents }];
  const items = Array.isArray(contents) ? contents : [contents];
  const out: WireMessage[] = [];
  let loose: string[] = [];
  for (const item of items) {
    const record = typeof item === "string" ? undefined : asRecord(item);
    const parts = record?.["parts"];
    if (parts === undefined && (record === undefined || record["role"] === undefined)) {
      loose.push(partText(item));
      continue;
    }
    if (loose.length > 0) {
      out.push({ role: "user", content: loose.join("\n") });
      loose = [];
    }
    const rawRole = record?.["role"];
    const role = typeof rawRole === "string" && rawRole ? (ROLE_MAP[rawRole] ?? rawRole) : "user";
    const list = Array.isArray(parts) ? parts : [];
    out.push({ role, content: list.map(partText).join("\n") });
  }
  if (loose.length > 0) out.push({ role: "user", content: loose.join("\n") });
  return out;
}

function systemOf(request: GeminiRequest | undefined): string | undefined {
  const instruction = request?.config?.systemInstruction;
  if (instruction === undefined || instruction === null) return undefined;
  if (typeof instruction === "string") return instruction;
  const parts = asRecord(instruction)?.["parts"];
  if (Array.isArray(parts)) return parts.map(partText).join("\n");
  if (Array.isArray(instruction)) return instruction.map(partText).join("\n");
  return partText(instruction);
}

/** The first candidate's text parts, joined; thought summaries excluded. */
function textOf(response: GeminiResponse | undefined): string {
  const parts = response?.candidates?.[0]?.content?.parts ?? [];
  return parts
    .filter((p) => typeof p?.text === "string" && !p.thought)
    .map((p) => p.text as string)
    .join("");
}

function hasContent(chunk: GeminiStreamChunk | undefined): boolean {
  const parts = chunk?.candidates?.[0]?.content?.parts ?? [];
  return parts.some((p) => Boolean(p?.text) || (p?.functionCall !== undefined && p?.functionCall !== null));
}

function emitLlmCall(
  options: InstrumentOptions,
  start: number,
  requestedModel: string | undefined,
  response: GeminiResponse,
  streamInfo?: { stream: true; finishReason?: string; toolCalls: number },
  extras: CallExtras = {},
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
    if (extras.ttftMs !== undefined) data["ttft_ms"] = extras.ttftMs;
    if (options.mode === "batch") data["mode"] = "batch";
    putContent(data, {
      input: renderContents(extras.request?.contents),
      system: systemOf(extras.request),
      output: extras.outputText,
    });
    run.event("llm_call", data);
  } catch {
    // swallow
  }
}
