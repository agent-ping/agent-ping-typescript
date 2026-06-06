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
 * LangChain.js doesn't ship type definitions for handlers as a stable
 * interface, so we duck-type: implement the method names LangChain calls
 * (`handleLLMEnd`, `handleToolStart`, etc.) and let it find them.
 */

import { getActiveRun } from "../context.js";

interface RunLike {
  event: (type: string, data: Record<string, unknown>) => void;
}

interface LangChainTokenUsage {
  promptTokens?: number;
  completionTokens?: number;
  inputTokens?: number;
  outputTokens?: number;
  totalTokens?: number;
}

interface LLMResult {
  llmOutput?: {
    tokenUsage?: LangChainTokenUsage;
    modelName?: string;
    model?: string;
    provider?: string;
  };
  generations?: unknown[];
}

export class AgentPingLangChainCallbackHandler {
  /** Required by LangChain.js to identify the handler. */
  name = "AgentPingLangChainCallbackHandler";

  private explicitRun?: RunLike;
  private llmStarts = new Map<string, { start: number; model?: string }>();

  constructor(run?: RunLike) {
    this.explicitRun = run;
  }

  private get run(): RunLike | undefined {
    return this.explicitRun ?? getActiveRun();
  }

  async handleLLMStart(
    llm: { id?: string[]; kwargs?: { modelName?: string; model?: string } } | undefined,
    _prompts: string[] | undefined,
    runId: string,
  ): Promise<void> {
    const model = llm?.kwargs?.modelName ?? llm?.kwargs?.model;
    this.llmStarts.set(runId, { start: Date.now(), model });
  }

  async handleChatModelStart(
    llm: { id?: string[]; kwargs?: { modelName?: string; model?: string } } | undefined,
    _messages: unknown,
    runId: string,
  ): Promise<void> {
    const model = llm?.kwargs?.modelName ?? llm?.kwargs?.model;
    this.llmStarts.set(runId, { start: Date.now(), model });
  }

  async handleLLMEnd(output: LLMResult, runId: string): Promise<void> {
    const run = this.run;
    if (!run) return;
    const startInfo = this.llmStarts.get(runId);
    this.llmStarts.delete(runId);
    const latencyMs = startInfo ? Date.now() - startInfo.start : 0;
    const requestedModel = startInfo?.model;

    try {
      const llmOutput = output.llmOutput ?? {};
      const usage = llmOutput.tokenUsage ?? {};
      const model = llmOutput.modelName ?? llmOutput.model ?? requestedModel ?? "unknown";
      const provider = inferProvider(model, llmOutput);

      const data: Record<string, unknown> = {
        provider,
        model,
        latency_ms: latencyMs,
      };
      const inputTokens = usage.inputTokens ?? usage.promptTokens;
      const outputTokens = usage.outputTokens ?? usage.completionTokens;
      if (inputTokens !== undefined) data["input_tokens"] = inputTokens;
      if (outputTokens !== undefined) data["output_tokens"] = outputTokens;

      run.event("llm_call", data);
    } catch {
      // swallow
    }
  }

  async handleToolStart(
    tool: { id?: string[]; name?: string } | undefined,
    inputStr: string,
    _runId: string,
  ): Promise<void> {
    const run = this.run;
    if (!run) return;
    try {
      run.event("tool_call", {
        tool: tool?.name ?? tool?.id?.[tool.id.length - 1] ?? "unknown",
        args_preview: String(inputStr).slice(0, 200),
      });
    } catch {
      // swallow
    }
  }

  async handleChainError(err: Error): Promise<void> {
    const run = this.run;
    if (!run) return;
    try {
      run.event("error", { message: String(err.message).slice(0, 500) });
    } catch {
      // swallow
    }
  }

  async handleLLMError(err: Error): Promise<void> {
    const run = this.run;
    if (!run) return;
    try {
      run.event("llm_call_error", { message: String(err.message).slice(0, 500) });
    } catch {
      // swallow
    }
  }
}

function inferProvider(model: string, llmOutput: LLMResult["llmOutput"]): string {
  if (llmOutput?.provider) return llmOutput.provider;
  const m = model.toLowerCase();
  if (m.startsWith("claude") || m.includes("anthropic")) return "anthropic";
  if (m.startsWith("gpt") || m.startsWith("o1") || m.startsWith("o3") || m.includes("openai")) return "openai";
  if (m.startsWith("gemini") || m.includes("google")) return "gemini";
  if (m.startsWith("mistral") || m.includes("mixtral")) return "mistral";
  if (m.startsWith("command") || m.includes("cohere")) return "cohere";
  return "langchain";
}
