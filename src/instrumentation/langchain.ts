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
 *    model reports them, `finish_reason` and the number of tool calls
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
  errorFields,
  putIfPositive,
  putIfString,
  stringifyPayload,
  type RunLike,
  type ToolPayloadOptions,
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
  generationInfo?: { finish_reason?: string; finishReason?: string };
  message?: {
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

interface LlmStart {
  start: number;
  model?: string;
  className?: string;
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
    _prompts: string[] | undefined,
    runId: string,
  ): Promise<void> {
    this.rememberLlmStart(llm, runId);
  }

  async handleChatModelStart(
    llm: Serialized | undefined,
    _messages: unknown,
    runId: string,
  ): Promise<void> {
    this.rememberLlmStart(llm, runId);
  }

  private rememberLlmStart(llm: Serialized | undefined, runId: string): void {
    this.llmStarts.set(runId, {
      start: Date.now(),
      model: llm?.kwargs?.modelName ?? llm?.kwargs?.model,
      className: llm?.id?.[llm.id.length - 1],
    });
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
