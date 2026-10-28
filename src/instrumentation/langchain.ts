/**
 * LangChain.js integration.
 *
 * Provides `AgentPingLangChainCallbackHandler`, an object that conforms to
 * LangChain.js's `BaseCallbackHandler` shape. Attach via:
 *
 *     await chain.invoke(input, {
 *       callbacks: [new agentping.AgentPingLangChainCallbackHandler()],
 *     });
 *
 * Constructor argument is optional. When omitted the handler resolves
 * the active run via AsyncLocalStorage, matching the Python `agentping.run`
 * context manager DX.
 *
 * Emits:
 *  - `llm_call` on every model completion, with provider, model, gross
 *    input tokens, output tokens, cached and reasoning tokens where the
 *    model reports them, `finish_reason` and the number of tool calls;
 *    streamed calls (those that fired `handleLLMNewToken`) add
 *    `stream: true` and `ttft_ms`, the time to the first content token;
 *    with `init({ captureLlmContent: true })` the prompt messages go out
 *    as `input` (`human` -> `user`, `ai` -> `assistant`, `tool` -> `tool`),
 *    system messages as `system_instructions` and the completion as
 *    `output`
 *  - `llm_call` with `status: "error"` when a model call throws
 *  - `tool_call` when a tool finishes or throws, with its input, output,
 *    latency and `tool_invocation_id` (the LangChain run id)
 *  - `error` when a chain throws
 *
 * LangChain.js doesn't ship type definitions for handlers as a stable
 * interface, so we duck-type: implement the method names LangChain calls
 * (`handleLLMEnd`, `handleToolStart`, etc.) and let it find them.
 */

import { getActiveRun } from "../context.js";
import {
  DEFAULT_TOOL_PAYLOAD_MAX_CHARS,
  FirstChunkClock,
  contentCaptureEnabled,
  errorFields,
  messageText,
  putContent,
  putIfPositive,
  putIfString,
  stringifyPayload,
  type RunLike,
  type ToolPayloadOptions,
  type WireMessage,
} from "./shared.js";

interface LangChainTokenUsage {
  promptTokens?: number;
  completionTokens?: number;
  inputTokens?: number;
  outputTokens?: number;
  totalTokens?: number;
}

/** `AIMessage.usage_metadata`; LangChain normalises every provider to this. */
interface LangChainUsageMetadata {
  input_tokens?: number;
  output_tokens?: number;
  total_tokens?: number;
  input_token_details?: { cache_read?: number; cache_creation?: number };
  output_token_details?: { reasoning?: number };
}

interface LangChainGeneration {
  /** Completion text of a plain LLM generation. */
  text?: string;
  generationInfo?: { finish_reason?: string; finishReason?: string };
  message?: {
    /** The AIMessage content: a string or a list of content parts. */
    content?: unknown;
    usage_metadata?: LangChainUsageMetadata;
    tool_calls?: unknown[];
    response_metadata?: {
      finish_reason?: string;
      stop_reason?: string;
      model_name?: string;
      model?: string;
    };
  };
}

interface LLMResult {
  llmOutput?: {
    tokenUsage?: LangChainTokenUsage;
    modelName?: string;
    model?: string;
    provider?: string;
  };
  generations?: LangChainGeneration[][];
}

interface Serialized {
  id?: string[];
  name?: string;
  kwargs?: { modelName?: string; model?: string };
}

/** `fields.chunk` of `handleLLMNewToken`: a GenerationChunk or ChatGenerationChunk. */
interface LangChainChunk {
  text?: string;
  message?: { tool_call_chunks?: unknown[] };
}

interface LlmStart {
  start: number;
  model?: string;
  className?: string;
  clock: FirstChunkClock;
  /** Set once `handleLLMNewToken` fires, so the call was streamed. */
  streamed: boolean;
  /** Prompt strings (`handleLLMStart`), kept only when capture is on. */
  prompts?: unknown;
  /** Chat messages (`handleChatModelStart`), kept only when capture is on. */
  messages?: unknown;
}

interface ToolStart {
  start: number;
  tool: string;
  input?: string;
}

export class AgentPingLangChainCallbackHandler {
  /** Required by LangChain.js to identify the handler. */
  name = "AgentPingLangChainCallbackHandler";

  private explicitRun?: RunLike;
  private options: ToolPayloadOptions;
  private llmStarts = new Map<string, LlmStart>();
  private toolStarts = new Map<string, ToolStart>();

  constructor(run?: RunLike, options: ToolPayloadOptions = {}) {
    this.explicitRun = run;
    this.options = options;
  }

  private get run(): RunLike | undefined {
    return this.explicitRun ?? getActiveRun();
  }

  async handleLLMStart(
    llm: Serialized | undefined,
    prompts: string[] | undefined,
    runId: string,
  ): Promise<void> {
    this.rememberLlmStart(llm, runId, { prompts });
  }

  async handleChatModelStart(
    llm: Serialized | undefined,
    messages: unknown,
    runId: string,
  ): Promise<void> {
    this.rememberLlmStart(llm, runId, { messages });
  }

  private rememberLlmStart(
    llm: Serialized | undefined,
    runId: string,
    content: { prompts?: unknown; messages?: unknown } = {},
  ): void {
    const info: LlmStart = {
      start: Date.now(),
      model: llm?.kwargs?.modelName ?? llm?.kwargs?.model,
      className: llm?.id?.[llm.id.length - 1],
      clock: new FirstChunkClock(),
      streamed: false,
    };
    // Only hold on to the prompt when it will be sent.
    if (contentCaptureEnabled()) {
      info.prompts = content.prompts;
      info.messages = content.messages;
    }
    this.llmStarts.set(runId, info);
  }

  /**
   * Called once per streamed chunk. Marks the time to first token when the
   * chunk carries content (text or a tool-call delta); OpenAI-style
   * role-only first chunks arrive as an empty token and don't count.
   */
  async handleLLMNewToken(
    token: string,
    _idx?: unknown,
    runId?: string,
    _parentRunId?: string,
    _tags?: string[],
    fields?: { chunk?: LangChainChunk },
  ): Promise<void> {
    if (!runId) return;
    const startInfo = this.llmStarts.get(runId);
    if (!startInfo) return;
    try {
      startInfo.streamed = true;
      const hasText = typeof token === "string" && token.length > 0;
      const hasToolCall = (fields?.chunk?.message?.tool_call_chunks?.length ?? 0) > 0;
      if (hasText || hasToolCall) startInfo.clock.mark();
    } catch {
      // swallow
    }
  }

  async handleLLMEnd(output: LLMResult, runId: string): Promise<void> {
    const startInfo = this.llmStarts.get(runId);
    this.llmStarts.delete(runId);
    const run = this.run;
    if (!run) return;
    const latencyMs = startInfo ? Date.now() - startInfo.start : 0;

    try {
      const llmOutput = output.llmOutput ?? {};
      const generation = output.generations?.[0]?.[0];
      const message = generation?.message;
      const meta = message?.response_metadata;
      const model =
        llmOutput.modelName ??
        llmOutput.model ??
        meta?.model_name ??
        meta?.model ??
        startInfo?.model ??
        "unknown";
      const provider = inferProvider(model, llmOutput.provider, startInfo?.className);

      const data: Record<string, unknown> = {
        provider,
        model,
        latency_ms: latencyMs,
      };

      // usage_metadata is LangChain's normalised, gross count; tokenUsage
      // is the older per-provider summary and only carries the totals.
      const usage = message?.usage_metadata;
      const legacy = llmOutput.tokenUsage ?? {};
      const inputTokens = usage?.input_tokens ?? legacy.inputTokens ?? legacy.promptTokens;
      const outputTokens = usage?.output_tokens ?? legacy.outputTokens ?? legacy.completionTokens;
      if (inputTokens !== undefined) data["input_tokens"] = inputTokens;
      if (outputTokens !== undefined) data["output_tokens"] = outputTokens;
      putIfPositive(data, "cached_input_tokens", usage?.input_token_details?.cache_read);
      putIfPositive(data, "cache_creation_input_tokens", usage?.input_token_details?.cache_creation);
      putIfPositive(data, "reasoning_tokens", usage?.output_token_details?.reasoning);

      putIfString(
        data,
        "finish_reason",
        generation?.generationInfo?.finish_reason ??
          generation?.generationInfo?.finishReason ??
          meta?.finish_reason ??
          meta?.stop_reason,
      );
      const toolCalls = message?.tool_calls?.length ?? 0;
      if (toolCalls > 0) data["tool_calls"] = toolCalls;
      if (startInfo?.streamed) data["stream"] = true;
      if (startInfo) startInfo.clock.put(data, startInfo.start);

      if (contentCaptureEnabled()) {
        const prompt = startInfo ? promptOf(startInfo) : {};
        putContent(data, {
          input: prompt.input,
          system: prompt.system,
          output: message !== undefined ? message.content : generation?.text,
        });
      }

      run.event("llm_call", data);
    } catch {
      // swallow
    }
  }

  async handleLLMError(err: unknown, runId?: string): Promise<void> {
    const startInfo = runId ? this.llmStarts.get(runId) : undefined;
    if (runId) this.llmStarts.delete(runId);
    const run = this.run;
    if (!run) return;
    try {
      const model = startInfo?.model ?? "unknown";
      run.event("llm_call", {
        provider: inferProvider(model, undefined, startInfo?.className),
        model,
        latency_ms: startInfo ? Date.now() - startInfo.start : 0,
        status: "error",
        ...errorFields(err),
      });
    } catch {
      // swallow
    }
  }

  async handleToolStart(
    tool: Serialized | undefined,
    inputStr: string,
    runId: string,
  ): Promise<void> {
    const name = tool?.name ?? tool?.id?.[tool.id.length - 1] ?? "unknown";
    this.toolStarts.set(runId, {
      start: Date.now(),
      tool: name,
      input: inputStr === undefined ? undefined : String(inputStr),
    });
  }

  async handleToolEnd(output: unknown, runId: string): Promise<void> {
    const startInfo = this.toolStarts.get(runId);
    this.toolStarts.delete(runId);
    const run = this.run;
    if (!run) return;
    try {
      const data = this.toolCallData(startInfo, runId, "success");
      if (this.capturePayloads()) {
        const rendered = stringifyPayload(toolOutput(output), this.maxChars());
        if (rendered !== undefined) data["output"] = rendered;
      }
      run.event("tool_call", data);
    } catch {
      // swallow
    }
  }

  async handleToolError(err: unknown, runId: string): Promise<void> {
    const startInfo = this.toolStarts.get(runId);
    this.toolStarts.delete(runId);
    const run = this.run;
    if (!run) return;
    try {
      const data = this.toolCallData(startInfo, runId, "error");
      Object.assign(data, errorFields(err));
      run.event("tool_call", data);
    } catch {
      // swallow
    }
  }

  async handleChainError(err: unknown): Promise<void> {
    const run = this.run;
    if (!run) return;
    try {
      const fields = errorFields(err);
      run.event("error", {
        message: fields.error,
        exception: fields.exception,
      });
    } catch {
      // swallow
    }
  }

  private toolCallData(
    startInfo: ToolStart | undefined,
    runId: string,
    status: "success" | "error",
  ): Record<string, unknown> {
    const data: Record<string, unknown> = {
      tool: startInfo?.tool ?? "unknown",
      status,
      tool_invocation_id: runId,
      latency_ms: startInfo ? Date.now() - startInfo.start : 0,
    };
    if (this.capturePayloads() && startInfo?.input !== undefined) {
      data["input"] = stringifyPayload(startInfo.input, this.maxChars());
    }
    return data;
  }

  private capturePayloads(): boolean {
    return this.options.captureToolPayloads ?? true;
  }

  private maxChars(): number {
    return this.options.toolPayloadMaxChars ?? DEFAULT_TOOL_PAYLOAD_MAX_CHARS;
  }
}

/**
 * The prompt a call started with, as wire messages. Chat messages come as
 * `BaseMessage[][]` (one list per prompt); system messages are split off
 * into `system`. Plain prompts are strings and become user messages.
 */
function promptOf(startInfo: LlmStart): { input?: unknown; system?: string } {
  if (startInfo.messages !== undefined) return chatMessages(startInfo.messages);
  if (startInfo.prompts !== undefined) return { input: startInfo.prompts };
  return {};
}

function chatMessages(messages: unknown): { input?: WireMessage[]; system?: string } {
  if (!Array.isArray(messages)) return {};
  const flat: unknown[] = [];
  for (const entry of messages) {
    if (Array.isArray(entry)) flat.push(...entry);
    else flat.push(entry);
  }
  const input: WireMessage[] = [];
  const system: string[] = [];
  for (const message of flat) {
    if (typeof message === "string") {
      input.push({ role: "user", content: message });
      continue;
    }
    if (!message || typeof message !== "object") continue;
    const record = message as Record<string, unknown>;
    const role = langchainRole(record);
    const content = messageText(record["content"]);
    if (role === "system") {
      if (content) system.push(content);
    } else {
      input.push({ role, content });
    }
  }
  return {
    input: input.length > 0 ? input : undefined,
    system: system.length > 0 ? system.join("\n\n") : undefined,
  };
}

const ROLE_BY_LANGCHAIN_TYPE: Record<string, string> = {
  human: "user",
  ai: "assistant",
  system: "system",
  tool: "tool",
  function: "tool",
};

/**
 * A BaseMessage's role on the wire. LangChain names it `_getType()` (or
 * `type`): `human`, `ai`, `system`, `tool`, `function` or `generic` (a
 * ChatMessage, which carries its own `role`). Serialized messages only
 * have the class name, e.g. `HumanMessage`, which maps the same way.
 */
function langchainRole(message: Record<string, unknown>): string {
  let type: unknown;
  const getType = message["_getType"];
  if (typeof getType === "function") {
    try {
      type = (getType as () => unknown).call(message);
    } catch {
      type = undefined;
    }
  }
  if (typeof type !== "string" || !type) type = message["type"];
  if (typeof type !== "string" || !type) {
    const id = message["lc_id"];
    const name = Array.isArray(id) ? id[id.length - 1] : message.constructor?.name;
    if (typeof name === "string" && name !== "Object") {
      type = name.replace(/Message(Chunk)?$/, "").toLowerCase();
    }
  }
  const role = message["role"];
  if ((typeof type !== "string" || !type || type === "generic") && typeof role === "string" && role) {
    return ROLE_BY_LANGCHAIN_TYPE[role] ?? role;
  }
  if (typeof type !== "string" || !type) return "user";
  return ROLE_BY_LANGCHAIN_TYPE[type] ?? type;
}

/** Tools may return a ToolMessage; its `content` is what the model saw. */
function toolOutput(output: unknown): unknown {
  if (output && typeof output === "object" && "content" in output) {
    return (output as { content: unknown }).content;
  }
  return output;
}

function inferProvider(model: string, explicit?: string, className?: string): string {
  if (explicit) return explicit;
  const c = (className ?? "").toLowerCase();
  if (c.includes("anthropic")) return "anthropic";
  if (c.includes("openai")) return "openai";
  if (c.includes("google") || c.includes("gemini") || c.includes("vertex")) return "gemini";
  if (c.includes("mistral")) return "mistral";
  if (c.includes("cohere")) return "cohere";
  if (c.includes("bedrock")) return "bedrock";
  const m = model.toLowerCase();
  if (m.startsWith("claude") || m.includes("anthropic")) return "anthropic";
  if (m.startsWith("gpt") || m.startsWith("o1") || m.startsWith("o3") || m.startsWith("o4") || m.includes("openai")) return "openai";
  if (m.startsWith("gemini") || m.includes("google")) return "gemini";
  if (m.startsWith("mistral") || m.includes("mixtral")) return "mistral";
  if (m.startsWith("command") || m.includes("cohere")) return "cohere";
  return "langchain";
}
