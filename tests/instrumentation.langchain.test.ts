import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import * as agentping from "../src/index.js";
import { AgentPingLangChainCallbackHandler } from "../src/instrumentation/langchain.js";
import { _resetWarningsForTests } from "../src/warnings.js";

const VALID_KEY = `apk_eu_${"a".repeat(32)}`;

describe("AgentPingLangChainCallbackHandler (TS)", () => {
  beforeEach(() => {
    _resetWarningsForTests();
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
  });

  afterEach(() => {
    agentping.shutdown();
    vi.restoreAllMocks();
  });

  it("handleLLMEnd emits llm_call with inferred provider from model name", async () => {
    const calls: Array<{ url: string; body: unknown }> = [];
    const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
      calls.push({ url, body: JSON.parse(init!.body as string) });
      return new Response("{}", { status: 202 });
    });
    agentping.init({
      apiKey: VALID_KEY,
      baseUrl: "https://api.example.com",
      flushIntervalMs: 5,
      batchSize: 10,
      fetchImpl: fetchMock as unknown as typeof fetch,
    });

    const run = agentping.run("rag-pipeline");
    const handler = new AgentPingLangChainCallbackHandler(run);

    await handler.handleChatModelStart({ kwargs: { modelName: "claude-sonnet-4-5" } }, [], "rid-1");
    await handler.handleLLMEnd(
      {
        llmOutput: {
          modelName: "claude-sonnet-4-5",
          tokenUsage: { inputTokens: 312, outputTokens: 88 },
        },
      },
      "rid-1",
    );

    await agentping.flush({ timeoutMs: 1_000 });
    const eventCall = calls.find((c) => c.url.includes("/events"));
    const body = eventCall!.body as { events: Array<{ type: string; data: Record<string, unknown> }> };
    const llm = body.events.find((e) => e.type === "llm_call")!;
    expect(llm.data["provider"]).toBe("anthropic");
    expect(llm.data["model"]).toBe("claude-sonnet-4-5");
    expect(llm.data["input_tokens"]).toBe(312);
    expect(llm.data["output_tokens"]).toBe(88);
  });

  it("handleLLMEnd reads usage_metadata, finish_reason and tool calls from the generation", async () => {
    const calls: Array<{ url: string; body: unknown }> = [];
    const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
      calls.push({ url, body: JSON.parse(init!.body as string) });
      return new Response("{}", { status: 202 });
    });
    agentping.init({
      apiKey: VALID_KEY,
      baseUrl: "https://api.example.com",
      flushIntervalMs: 5,
      batchSize: 10,
      fetchImpl: fetchMock as unknown as typeof fetch,
    });

    const run = agentping.run("rag-pipeline");
    const handler = new AgentPingLangChainCallbackHandler(run);

    await handler.handleChatModelStart(
      { id: ["langchain", "chat_models", "openai", "ChatOpenAI"], kwargs: { model: "gpt-5-mini" } },
      [],
      "rid-2",
    );
    await handler.handleLLMEnd(
      {
        generations: [
          [
            {
              generationInfo: { finish_reason: "tool_calls" },
              message: {
                tool_calls: [{ id: "call_1", name: "search", args: {} }],
                response_metadata: { model_name: "gpt-5-mini-2025-08-07", finish_reason: "tool_calls" },
                usage_metadata: {
                  input_tokens: 500,
                  output_tokens: 120,
                  total_tokens: 620,
                  input_token_details: { cache_read: 200 },
                  output_token_details: { reasoning: 40 },
                },
              },
            },
          ],
        ],
      },
      "rid-2",
    );

    await agentping.flush({ timeoutMs: 1_000 });
    const eventCall = calls.find((c) => c.url.includes("/events"));
    const body = eventCall!.body as { events: Array<{ type: string; data: Record<string, unknown> }> };
    const llm = body.events.find((e) => e.type === "llm_call")!;
    expect(llm.data["provider"]).toBe("openai");
    expect(llm.data["model"]).toBe("gpt-5-mini-2025-08-07");
    expect(llm.data["input_tokens"]).toBe(500);
    expect(llm.data["cached_input_tokens"]).toBe(200);
    expect(llm.data["output_tokens"]).toBe(120);
    expect(llm.data["reasoning_tokens"]).toBe(40);
    expect(llm.data["finish_reason"]).toBe("tool_calls");
    expect(llm.data["tool_calls"]).toBe(1);
  });

  it("handleLLMError emits an errored llm_call for the started model", async () => {
    const calls: Array<{ url: string; body: unknown }> = [];
    const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
      calls.push({ url, body: JSON.parse(init!.body as string) });
      return new Response("{}", { status: 202 });
    });
    agentping.init({
      apiKey: VALID_KEY,
      baseUrl: "https://api.example.com",
      flushIntervalMs: 5,
      batchSize: 10,
      fetchImpl: fetchMock as unknown as typeof fetch,
    });

    const run = agentping.run("rag-pipeline");
    const handler = new AgentPingLangChainCallbackHandler(run);
    await handler.handleChatModelStart({ kwargs: { model: "claude-sonnet-4-5" } }, [], "rid-3");
    await handler.handleLLMError(new Error("overloaded"), "rid-3");

    await agentping.flush({ timeoutMs: 1_000 });
    const eventCall = calls.find((c) => c.url.includes("/events"));
    const body = eventCall!.body as { events: Array<{ type: string; data: Record<string, unknown> }> };
    expect(body.events.find((e) => e.type === "llm_call_error")).toBeUndefined();
    const llm = body.events.find((e) => e.type === "llm_call")!;
    expect(llm.data["provider"]).toBe("anthropic");
    expect(llm.data["model"]).toBe("claude-sonnet-4-5");
    expect(llm.data["status"]).toBe("error");
    expect(llm.data["error"]).toBe("overloaded");
    expect(llm.data["exception"]).toBe("Error");
    expect(llm.data["latency_ms"]).toBeTypeOf("number");
  });

  it("emits tool_call on handleToolEnd with input, output and latency", async () => {
    const calls: Array<{ url: string; body: unknown }> = [];
    const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
      calls.push({ url, body: JSON.parse(init!.body as string) });
      return new Response("{}", { status: 202 });
    });
    agentping.init({
      apiKey: VALID_KEY,
      baseUrl: "https://api.example.com",
      flushIntervalMs: 5,
      batchSize: 10,
      fetchImpl: fetchMock as unknown as typeof fetch,
    });

    const run = agentping.run("tool-using");
    const handler = new AgentPingLangChainCallbackHandler(run);
    await handler.handleToolStart({ name: "search_db" }, '{"q":"pricing"}', "rid-1");
    await handler.handleToolEnd({ content: "3 rows" }, "rid-1");

    await agentping.flush({ timeoutMs: 1_000 });
    const eventCall = calls.find((c) => c.url.includes("/events"));
    const body = eventCall!.body as { events: Array<{ type: string; data: Record<string, unknown> }> };
    const toolCalls = body.events.filter((e) => e.type === "tool_call");
    expect(toolCalls).toHaveLength(1);
    const tool = toolCalls[0]!;
    expect(tool.data["tool"]).toBe("search_db");
    expect(tool.data["status"]).toBe("success");
    expect(tool.data["tool_invocation_id"]).toBe("rid-1");
    expect(tool.data["input"]).toBe('{"q":"pricing"}');
    expect(tool.data["output"]).toBe("3 rows");
    expect(tool.data["latency_ms"]).toBeTypeOf("number");
    expect("args_preview" in tool.data).toBe(false);
  });

  it("emits an errored tool_call on handleToolError", async () => {
    const calls: Array<{ url: string; body: unknown }> = [];
    const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
      calls.push({ url, body: JSON.parse(init!.body as string) });
      return new Response("{}", { status: 202 });
    });
    agentping.init({
      apiKey: VALID_KEY,
      baseUrl: "https://api.example.com",
      flushIntervalMs: 5,
      batchSize: 10,
      fetchImpl: fetchMock as unknown as typeof fetch,
    });

    class ToolTimeout extends Error {}
    const run = agentping.run("tool-using");
    const handler = new AgentPingLangChainCallbackHandler(run, { toolPayloadMaxChars: 8 });
    await handler.handleToolStart({ id: ["langchain", "tools", "SearchTool"] }, '{"q":"pricing"}', "rid-9");
    await handler.handleToolError(new ToolTimeout("timed out after 5s"), "rid-9");

    await agentping.flush({ timeoutMs: 1_000 });
    const eventCall = calls.find((c) => c.url.includes("/events"));
    const body = eventCall!.body as { events: Array<{ type: string; data: Record<string, unknown> }> };
    const tool = body.events.find((e) => e.type === "tool_call")!;
    expect(tool.data["tool"]).toBe("SearchTool");
    expect(tool.data["status"]).toBe("error");
    expect(tool.data["tool_invocation_id"]).toBe("rid-9");
    expect(tool.data["input"]).toBe('{"q":"pr...');
    expect(tool.data["error"]).toBe("timed out after 5s");
    expect(tool.data["exception"]).toBe("ToolTimeout");
    expect("output" in tool.data).toBe(false);
  });

  it("handleChainError emits error event", async () => {
    const calls: Array<{ url: string; body: unknown }> = [];
    const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
      calls.push({ url, body: JSON.parse(init!.body as string) });
      return new Response("{}", { status: 202 });
    });
    agentping.init({
      apiKey: VALID_KEY,
      baseUrl: "https://api.example.com",
      flushIntervalMs: 5,
      batchSize: 10,
      fetchImpl: fetchMock as unknown as typeof fetch,
    });

    const run = agentping.run("failing-chain");
    const handler = new AgentPingLangChainCallbackHandler(run);
    await handler.handleChainError(new Error("upstream rate limit hit"));

    await agentping.flush({ timeoutMs: 1_000 });
    const eventCall = calls.find((c) => c.url.includes("/events"));
    const body = eventCall!.body as { events: Array<{ type: string; data: Record<string, unknown> }> };
    const err = body.events.find((e) => e.type === "error")!;
    expect(String(err.data["message"])).toContain("rate limit");
    expect(err.data["exception"]).toBe("Error");
  });

  it("resolves active run from runScopeAsync when constructed with no arg", async () => {
    const calls: Array<{ url: string; body: unknown }> = [];
    const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
      calls.push({ url, body: JSON.parse(init!.body as string) });
      return new Response("{}", { status: 202 });
    });
    agentping.init({
      apiKey: VALID_KEY,
      baseUrl: "https://api.example.com",
      flushIntervalMs: 5,
      batchSize: 10,
      fetchImpl: fetchMock as unknown as typeof fetch,
    });

    const handler = new AgentPingLangChainCallbackHandler(); // no run arg!
    const run = agentping.run("scoped");
    await agentping.runScopeAsync(run, async () => {
      await handler.handleLLMEnd(
        {
          llmOutput: { modelName: "gpt-4o", tokenUsage: { inputTokens: 12, outputTokens: 3 } },
        },
        "rid-1",
      );
    });

    await agentping.flush({ timeoutMs: 1_000 });
    const eventCall = calls.find((c) => c.url.includes("/events"));
    const body = eventCall!.body as { events: Array<{ type: string; data: Record<string, unknown> }> };
    const llm = body.events.find((e) => e.type === "llm_call")!;
    expect(llm.data["provider"]).toBe("openai");
    expect(llm.data["model"]).toBe("gpt-4o");
  });
});
