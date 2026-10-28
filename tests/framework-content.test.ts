import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import * as agentping from "../src/index.js";
import { AgentPingLangChainCallbackHandler } from "../src/instrumentation/langchain.js";
import { AgentPingHooks } from "../src/instrumentation/openai-agents.js";
import { withAgentPing } from "../src/instrumentation/vercel-ai.js";
import { _resetWarningsForTests } from "../src/warnings.js";

const VALID_KEY = `apk_eu_${"a".repeat(32)}`;

type Call = { url: string; body: Record<string, unknown> };
type LlmCall = Record<string, unknown>;
type WireMessage = { role: string; content: string };

function capture(): { calls: Call[]; fetchImpl: typeof fetch } {
  const calls: Call[] = [];
  const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
    calls.push({ url, body: JSON.parse(init!.body as string) as Record<string, unknown> });
    return new Response("{}", { status: 202 });
  });
  return { calls, fetchImpl: fetchMock as unknown as typeof fetch };
}

function init(fetchImpl: typeof fetch, extra: Record<string, unknown> = {}): void {
  agentping.init({
    apiKey: VALID_KEY,
    baseUrl: "https://api.example.com",
    flushIntervalMs: 5,
    batchSize: 10,
    fetchImpl,
    ...extra,
  });
}

function llmCalls(calls: Call[]): LlmCall[] {
  return calls
    .filter((c) => c.url.includes("/events"))
    .flatMap((c) => c.body["events"] as Array<{ type: string; data: LlmCall }>)
    .filter((e) => e.type === "llm_call")
    .map((e) => e.data);
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** A LangChain.js BaseMessage look-alike: role via `_getType()`. */
function lcMessage(type: string, content: unknown): { _getType: () => string; content: unknown } {
  return { _getType: () => type, content };
}

describe("framework integrations: prompt capture and time to first token", () => {
  beforeEach(() => {
    _resetWarningsForTests();
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
  });
  afterEach(() => {
    agentping.shutdown();
    vi.restoreAllMocks();
  });

  describe("LangChain.js", () => {
    const chatEnd = {
      llmOutput: { modelName: "claude-haiku-5-5" },
      generations: [
        [
          {
            text: "Route to billing.",
            generationInfo: { finish_reason: "stop" },
            message: {
              content: "Route to billing.",
              usage_metadata: { input_tokens: 10, output_tokens: 5 },
            },
          },
        ],
      ],
    };

    it("sends nothing about the prompt by default, and no stream fields off-stream", async () => {
      const { calls, fetchImpl } = capture();
      init(fetchImpl);
      const handler = new AgentPingLangChainCallbackHandler(agentping.run("triage"));

      await handler.handleChatModelStart(
        { kwargs: { model: "claude-haiku-5-5" } },
        [[lcMessage("system", "Be terse."), lcMessage("human", "hi")]],
        "rid-1",
      );
      await handler.handleLLMEnd(chatEnd, "rid-1");
      await agentping.flush({ timeoutMs: 1_000 });

      const [data] = llmCalls(calls);
      expect(data).toBeTruthy();
      expect(data!["input_tokens"]).toBe(10);
      expect("input" in data!).toBe(false);
      expect("output" in data!).toBe(false);
      expect("system_instructions" in data!).toBe(false);
      expect("stream" in data!).toBe(false);
      expect("ttft_ms" in data!).toBe(false);
    });

    it("maps chat message types to wire roles, splits system off, and reports ttft on streams", async () => {
      const { calls, fetchImpl } = capture();
      init(fetchImpl, { captureLlmContent: true, llmContentMaxChars: 40 });
      const handler = new AgentPingLangChainCallbackHandler(agentping.run("triage"));

      await handler.handleChatModelStart(
        { kwargs: { model: "claude-haiku-5-5" } },
        [
          [
            lcMessage("system", "Be terse."),
            lcMessage("human", [
              { type: "text", text: "Refund request from Acme ".repeat(10) },
              { type: "image_url", image_url: { url: "data:..." } },
            ]),
            lcMessage("ai", "Checking."),
            lcMessage("tool", "balance: 12"),
          ],
        ],
        "rid-2",
      );
      // A role-only first chunk arrives as an empty token: not content.
      await handler.handleLLMNewToken("", { prompt: 0, completion: 0 }, "rid-2");
      await sleep(15);
      await handler.handleLLMNewToken("Route ", { prompt: 0, completion: 0 }, "rid-2");
      await handler.handleLLMNewToken("to billing.", { prompt: 0, completion: 0 }, "rid-2");
      await handler.handleLLMEnd(chatEnd, "rid-2");
      await agentping.flush({ timeoutMs: 1_000 });

      const [data] = llmCalls(calls);
      const input = data!["input"] as WireMessage[];
      expect(input.map((m) => m.role)).toEqual(["user", "assistant", "tool"]);
      expect(input[0]!.content.endsWith("...")).toBe(true);
      expect(input[0]!.content.length).toBe(43);
      expect(input[1]!.content).toBe("Checking.");
      expect(input[2]!.content).toBe("balance: 12");
      expect(data!["system_instructions"]).toBe("Be terse.");
      expect(data!["output"]).toBe("Route to billing.");
      expect(data!["stream"]).toBe(true);
      expect(typeof data!["ttft_ms"]).toBe("number");
      expect(data!["ttft_ms"] as number).toBeGreaterThanOrEqual(10);
      expect(data!["ttft_ms"] as number).toBeLessThanOrEqual(data!["latency_ms"] as number);
    });

    it("recognises `type`, class names and ChatMessage roles, and tool-call chunks as content", async () => {
      const { calls, fetchImpl } = capture();
      init(fetchImpl, { captureLlmContent: true });
      const handler = new AgentPingLangChainCallbackHandler(agentping.run("triage"));

      class HumanMessage {
        constructor(public content: string) {}
      }
      await handler.handleChatModelStart(
        { kwargs: { model: "gpt-5" } },
        [
          [
            { type: "system", content: "Be terse." },
            new HumanMessage("hi"),
            { lc_id: ["langchain_core", "messages", "AIMessageChunk"], content: "ok" },
            { _getType: () => "generic", role: "reviewer", content: "approved" },
          ],
        ],
        "rid-3",
      );
      await handler.handleLLMNewToken("", { prompt: 0, completion: 0 }, "rid-3", undefined, [], {
        chunk: { text: "", message: { tool_call_chunks: [{ name: "lookup", args: "{" }] } },
      });
      await handler.handleLLMEnd(
        {
          generations: [[{ text: "", message: { content: "", tool_calls: [{ name: "lookup" }] } }]],
        },
        "rid-3",
      );
      await agentping.flush({ timeoutMs: 1_000 });

      const [data] = llmCalls(calls);
      expect(data!["system_instructions"]).toBe("Be terse.");
      expect(data!["input"]).toEqual([
        { role: "user", content: "hi" },
        { role: "assistant", content: "ok" },
        { role: "reviewer", content: "approved" },
      ]);
      expect("output" in data!).toBe(false);
      expect(data!["tool_calls"]).toBe(1);
      expect(data!["stream"]).toBe(true);
      expect(typeof data!["ttft_ms"]).toBe("number");
    });

    it("plain LLM prompts become user messages and the generation text is the output", async () => {
      const { calls, fetchImpl } = capture();
      init(fetchImpl, { captureLlmContent: true });
      const handler = new AgentPingLangChainCallbackHandler(agentping.run("triage"));

      await handler.handleLLMStart({ kwargs: { model: "gpt-5" } }, ["Summarise: hello"], "rid-4");
      await handler.handleLLMEnd(
        { llmOutput: { tokenUsage: { promptTokens: 4, completionTokens: 2 } }, generations: [[{ text: "Hello." }]] },
        "rid-4",
      );
      await agentping.flush({ timeoutMs: 1_000 });

      const [data] = llmCalls(calls);
      expect(data!["input"]).toEqual([{ role: "user", content: "Summarise: hello" }]);
      expect(data!["output"]).toBe("Hello.");
      expect("system_instructions" in data!).toBe(false);
      expect("stream" in data!).toBe(false);
    });
  });

  describe("Vercel AI SDK", () => {
    it("sends nothing about the completion by default and no stream fields without onChunk", async () => {
      const { calls, fetchImpl } = capture();
      init(fetchImpl);
      const helper = withAgentPing(agentping.run("agent"), { provider: "openai", model: "gpt-5" });
      expect(helper.onChunk).toBeTypeOf("function");

      helper.onFinish({ text: "Route to billing.", usage: { inputTokens: 10, outputTokens: 4 }, finishReason: "stop" });
      await agentping.flush({ timeoutMs: 1_000 });

      const [data] = llmCalls(calls);
      expect(data!["input_tokens"]).toBe(10);
      expect("output" in data!).toBe(false);
      expect("input" in data!).toBe(false);
      expect("stream" in data!).toBe(false);
      expect("ttft_ms" in data!).toBe(false);
    });

    it("captures text as output and reports ttft_ms from the first content chunk", async () => {
      const { calls, fetchImpl } = capture();
      init(fetchImpl, { captureLlmContent: true, llmContentMaxChars: 10 });
      const helper = withAgentPing(agentping.run("agent"), { provider: "openai", model: "gpt-5" });

      helper.onChunk!({ chunk: { type: "raw" } });
      helper.onChunk!({ chunk: { type: "source" } });
      await sleep(15);
      helper.onChunk!({ chunk: { type: "text-delta", text: "Route " } as { type: string } });
      helper.onChunk!({ chunk: { type: "text-delta", text: "to billing." } as { type: string } });
      helper.onFinish({ text: "Route to billing.", usage: { inputTokens: 10, outputTokens: 4 }, finishReason: "stop" });
      await agentping.flush({ timeoutMs: 1_000 });

      const [data] = llmCalls(calls);
      expect(data!["output"]).toBe("Route to b...");
      expect(data!["stream"]).toBe(true);
      expect(typeof data!["ttft_ms"]).toBe("number");
      expect(data!["ttft_ms"] as number).toBeGreaterThanOrEqual(10);
      expect(data!["ttft_ms"] as number).toBeLessThanOrEqual(data!["latency_ms"] as number);
      expect("input" in data!).toBe(false);
    });

    it("perStep: each step gets its own ttft_ms and output, and onFinish stays silent", async () => {
      const { calls, fetchImpl } = capture();
      init(fetchImpl, { captureLlmContent: true });
      const helper = withAgentPing(agentping.run("agent"), { provider: "openai", model: "gpt-5", perStep: true });

      helper.onChunk!({ chunk: { type: "tool-input-start" } });
      helper.onStepFinish!({
        text: "",
        usage: { inputTokens: 10, outputTokens: 4 },
        finishReason: "tool-calls",
        toolCalls: [{ toolName: "lookup", toolCallId: "c1" }],
      });
      await sleep(15);
      helper.onChunk!({ chunk: { type: "text-delta" } });
      helper.onStepFinish!({ text: "Route to billing.", usage: { inputTokens: 20, outputTokens: 9 }, finishReason: "stop" });
      helper.onFinish({ text: "Route to billing.", usage: { inputTokens: 30, outputTokens: 13 } });
      await agentping.flush({ timeoutMs: 1_000 });

      const llms = llmCalls(calls);
      expect(llms).toHaveLength(2);
      expect(llms[0]!["stream"]).toBe(true);
      expect(typeof llms[0]!["ttft_ms"]).toBe("number");
      expect(llms[0]!["ttft_ms"] as number).toBeLessThan(10);
      expect("output" in llms[0]!).toBe(false);
      expect(llms[1]!["stream"]).toBe(true);
      expect(llms[1]!["ttft_ms"] as number).toBeGreaterThanOrEqual(10);
      expect(llms[1]!["output"]).toBe("Route to billing.");
    });
  });

  describe("OpenAI Agents SDK", () => {
    const response = {
      usage: { inputTokens: 12, outputTokens: 6 },
      output: [
        { type: "reasoning", content: [] },
        { type: "message", role: "assistant", content: [{ type: "output_text", text: "Route to billing." }] },
      ],
    };

    it("sends nothing about the prompt by default", async () => {
      const { calls, fetchImpl } = capture();
      init(fetchImpl);
      const hooks = new AgentPingHooks(agentping.run("triage"));
      const ctx = {};
      const agent = { name: "Triage", model: "gpt-5", instructions: "Be terse." };

      await hooks.onLLMStart(ctx, agent, "Be terse.", "hi");
      await hooks.onLLMEnd(ctx, agent, response);
      await agentping.flush({ timeoutMs: 1_000 });

      const [data] = llmCalls(calls);
      expect(data!["input_tokens"]).toBe(12);
      expect("input" in data!).toBe(false);
      expect("output" in data!).toBe(false);
      expect("system_instructions" in data!).toBe(false);
      expect("ttft_ms" in data!).toBe(false);
    });

    it("captures the system prompt and input items from onLLMStart and output_text from the response", async () => {
      const { calls, fetchImpl } = capture();
      init(fetchImpl, { captureLlmContent: true });
      const hooks = new AgentPingHooks(agentping.run("triage"));
      const ctx = {};
      const agent = { name: "Triage", model: "gpt-5", instructions: () => "ignored" };

      await hooks.onLLMStart(ctx, agent, "Be terse.", [
        { type: "message", role: "user", content: [{ type: "input_text", text: "hi" }] },
        { type: "function_call", name: "lookup", callId: "c1", arguments: "{}" },
        { type: "function_call_result", callId: "c1", output: { type: "text", text: "12" } },
        { type: "message", role: "assistant", content: [{ type: "output_text", text: "Checking." }] },
      ]);
      await hooks.onLLMEnd(ctx, agent, response);
      await agentping.flush({ timeoutMs: 1_000 });

      const [data] = llmCalls(calls);
      expect(data!["system_instructions"]).toBe("Be terse.");
      expect(data!["input"]).toEqual([
        { role: "user", content: "hi" },
        { role: "assistant", content: "Checking." },
      ]);
      expect(data!["output"]).toBe("Route to billing.");
      expect("ttft_ms" in data!).toBe(false);
    });

    it("falls back to the agent's string instructions and a plain-string input", async () => {
      const { calls, fetchImpl } = capture();
      init(fetchImpl, { captureLlmContent: true });
      const hooks = new AgentPingHooks(agentping.run("triage"));
      const ctx = {};
      const agent = { name: "Triage", model: "gpt-5", instructions: "Be terse." };

      await hooks.onLLMStart(ctx, agent, undefined, "hi");
      await hooks.onLLMEnd(ctx, agent, { usage: { inputTokens: 1, outputTokens: 1 } });
      await agentping.flush({ timeoutMs: 1_000 });

      const [data] = llmCalls(calls);
      expect(data!["system_instructions"]).toBe("Be terse.");
      expect(data!["input"]).toEqual([{ role: "user", content: "hi" }]);
      expect("output" in data!).toBe(false);
    });
  });
});
