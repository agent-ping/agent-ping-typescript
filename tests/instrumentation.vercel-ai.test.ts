import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import * as agentping from "../src/index.js";
import { withAgentPing, agentPingOnFinish } from "../src/instrumentation/vercel-ai.js";
import { _resetWarningsForTests } from "../src/warnings.js";

const VALID_KEY = `apk_eu_${"a".repeat(32)}`;

describe("Vercel AI SDK helper", () => {
  beforeEach(() => {
    _resetWarningsForTests();
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
  });

  afterEach(() => {
    agentping.shutdown();
    vi.restoreAllMocks();
  });

  it("withAgentPing.onFinish emits llm_call from inputTokens/outputTokens (AI SDK v5+)", async () => {
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

    const run = agentping.run("chat");
    const helper = withAgentPing(run, { provider: "openai", model: "gpt-4o-mini" });

    helper.onFinish({
      usage: { inputTokens: 200, outputTokens: 80 },
      finishReason: "stop",
    });

    await agentping.flush({ timeoutMs: 1_000 });

    const eventCall = calls.find((c) => c.url.includes("/events"));
    const body = eventCall!.body as { events: Array<{ type: string; data: Record<string, unknown> }> };
    const llm = body.events.find((e) => e.type === "llm_call")!;
    expect(llm.data["provider"]).toBe("openai");
    expect(llm.data["model"]).toBe("gpt-4o-mini");
    expect(llm.data["input_tokens"]).toBe(200);
    expect(llm.data["output_tokens"]).toBe(80);
  });

  it("withAgentPing.onFinish reads promptTokens/completionTokens (AI SDK v4)", async () => {
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

    const run = agentping.run("chat");
    const helper = withAgentPing(run, { provider: "anthropic", model: "claude-sonnet-4-5" });
    helper.onFinish({ usage: { promptTokens: 410, completionTokens: 192 } });

    await agentping.flush({ timeoutMs: 1_000 });

    const eventCall = calls.find((c) => c.url.includes("/events"));
    const body = eventCall!.body as { events: Array<{ type: string; data: Record<string, unknown> }> };
    const llm = body.events.find((e) => e.type === "llm_call")!;
    expect(llm.data["provider"]).toBe("anthropic");
    expect(llm.data["input_tokens"]).toBe(410);
    expect(llm.data["output_tokens"]).toBe(192);
  });

  it("emits a tool_call event for each toolCall in the payload", async () => {
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

    const run = agentping.run("agent");
    const helper = withAgentPing(run, { provider: "openai", model: "gpt-4o" });
    helper.onFinish({
      usage: { inputTokens: 100, outputTokens: 30 },
      toolCalls: [
        { toolName: "search", toolCallId: "c1" },
        { toolName: "fetch", toolCallId: "c2" },
      ],
    });

    await agentping.flush({ timeoutMs: 1_000 });

    const eventCall = calls.find((c) => c.url.includes("/events"));
    const body = eventCall!.body as { events: Array<{ type: string; data: Record<string, unknown> }> };
    const toolCalls = body.events.filter((e) => e.type === "tool_call");
    expect(toolCalls).toHaveLength(2);
    expect(toolCalls.map((c) => c.data["tool"])).toEqual(["search", "fetch"]);
  });

  it("emits finish_reason event for non-stop reasons", async () => {
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

    const run = agentping.run("agent");
    const helper = withAgentPing(run, { provider: "openai", model: "gpt-4o" });
    helper.onFinish({
      usage: { inputTokens: 50, outputTokens: 0 },
      finishReason: "length",
    });

    await agentping.flush({ timeoutMs: 1_000 });

    const eventCall = calls.find((c) => c.url.includes("/events"));
    const body = eventCall!.body as { events: Array<{ type: string; data: Record<string, unknown> }> };
    const finish = body.events.find((e) => e.type === "finish_reason");
    expect(finish).toBeTruthy();
    expect(finish!.data["reason"]).toBe("length");
  });

  it("perStep: true wires onStepFinish that emits per-step events", async () => {
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

    const run = agentping.run("agent");
    const helper = withAgentPing(run, { provider: "openai", model: "gpt-4o", perStep: true });
    expect(helper.onStepFinish).toBeTypeOf("function");

    helper.onStepFinish!({ usage: { inputTokens: 10, outputTokens: 4 } });
    helper.onStepFinish!({ usage: { inputTokens: 20, outputTokens: 9 } });
    helper.onFinish({ usage: { inputTokens: 30, outputTokens: 13 } });

    await agentping.flush({ timeoutMs: 1_000 });

    const eventBodies = calls
      .filter((c) => c.url.includes("/events"))
      .flatMap((c) => (c.body as { events: Array<{ type: string }> }).events);
    const llms = eventBodies.filter((e) => e.type === "llm_call");
    expect(llms).toHaveLength(3); // 2 steps + final
  });

  it("auto-detects provider+model from response.modelId (provider/model format)", async () => {
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

    const run = agentping.run("auto");
    const helper = withAgentPing(run); // no provider/model overrides

    helper.onFinish({
      usage: { inputTokens: 100, outputTokens: 40 },
      response: { modelId: "anthropic/claude-sonnet-4-5" },
    });

    await agentping.flush({ timeoutMs: 1_000 });
    const eventCall = calls.find((c) => c.url.includes("/events"));
    const body = eventCall!.body as { events: Array<{ type: string; data: Record<string, unknown> }> };
    const llm = body.events.find((e) => e.type === "llm_call")!;
    expect(llm.data["provider"]).toBe("anthropic");
    expect(llm.data["model"]).toBe("claude-sonnet-4-5");
  });

  it("auto-detects from event.model object when response.modelId is absent", async () => {
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

    const run = agentping.run("auto");
    const helper = withAgentPing(run);

    helper.onFinish({
      usage: { inputTokens: 50, outputTokens: 12 },
      model: { provider: "openai", modelId: "gpt-4o-mini" } as unknown,
    });

    await agentping.flush({ timeoutMs: 1_000 });
    const eventCall = calls.find((c) => c.url.includes("/events"));
    const body = eventCall!.body as { events: Array<{ type: string; data: Record<string, unknown> }> };
    const llm = body.events.find((e) => e.type === "llm_call")!;
    expect(llm.data["provider"]).toBe("openai");
    expect(llm.data["model"]).toBe("gpt-4o-mini");
  });

  it("falls back to vercel-ai/unknown when nothing can be inferred", async () => {
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

    const run = agentping.run("auto");
    const helper = withAgentPing(run);
    helper.onFinish({ usage: { inputTokens: 10, outputTokens: 2 } });

    await agentping.flush({ timeoutMs: 1_000 });
    const eventCall = calls.find((c) => c.url.includes("/events"));
    const body = eventCall!.body as { events: Array<{ type: string; data: Record<string, unknown> }> };
    const llm = body.events.find((e) => e.type === "llm_call")!;
    expect(llm.data["provider"]).toBe("vercel-ai");
    expect(llm.data["model"]).toBe("unknown");
  });

  it("agentPingOnFinish returns just the onFinish callback", async () => {
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

    const run = agentping.run("agent");
    const cb = agentPingOnFinish(run, { provider: "openai", model: "gpt-4o-mini" });
    cb({ usage: { inputTokens: 50, outputTokens: 12 } });

    await agentping.flush({ timeoutMs: 1_000 });

    const eventCall = calls.find((c) => c.url.includes("/events"));
    const body = eventCall!.body as { events: Array<{ type: string; data: Record<string, unknown> }> };
    const llm = body.events.find((e) => e.type === "llm_call")!;
    expect(llm.data["input_tokens"]).toBe(50);
  });
});
