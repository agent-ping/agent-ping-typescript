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

  it("handleToolStart emits tool_call event with name + args preview", async () => {
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

    await agentping.flush({ timeoutMs: 1_000 });
    const eventCall = calls.find((c) => c.url.includes("/events"));
    const body = eventCall!.body as { events: Array<{ type: string; data: Record<string, unknown> }> };
    const tool = body.events.find((e) => e.type === "tool_call")!;
    expect(tool.data["tool"]).toBe("search_db");
    expect(tool.data["args_preview"]).toContain("pricing");
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
