import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import * as agentping from "../src/index.js";
import { AgentPingHooks } from "../src/instrumentation/openai-agents.js";
import { _resetWarningsForTests } from "../src/warnings.js";

const VALID_KEY = `apk_eu_${"a".repeat(32)}`;

describe("AgentPingHooks (OpenAI Agents SDK)", () => {
  beforeEach(() => {
    _resetWarningsForTests();
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
  });

  afterEach(() => {
    agentping.shutdown();
    vi.restoreAllMocks();
  });

  it("onLLMEnd emits llm_call with usage and model", async () => {
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

    const run = agentping.run("triage");
    const hooks = new AgentPingHooks(run);
    const ctx = {};
    const agent = { name: "Triage", model: "gpt-4o-mini" };
    const response = { usage: { inputTokens: 412, outputTokens: 88 } };

    await hooks.onLLMStart(ctx, agent);
    await hooks.onLLMEnd(ctx, agent, response);

    await agentping.flush({ timeoutMs: 1_000 });

    const eventCall = calls.find((c) => c.url.includes("/events"));
    const body = eventCall!.body as { events: Array<{ type: string; data: Record<string, unknown> }> };
    const llm = body.events.find((e) => e.type === "llm_call")!;
    expect(llm.data["provider"]).toBe("openai");
    expect(llm.data["model"]).toBe("gpt-4o-mini");
    expect(llm.data["input_tokens"]).toBe(412);
    expect(llm.data["output_tokens"]).toBe(88);
    expect(llm.data["latency_ms"]).toBeTypeOf("number");
  });

  it("onToolStart emits tool_call event", async () => {
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

    const run = agentping.run("triage");
    const hooks = new AgentPingHooks(run);
    await hooks.onToolStart({}, { name: "Triage" }, { name: "fetch_orders" });

    await agentping.flush({ timeoutMs: 1_000 });

    const eventCall = calls.find((c) => c.url.includes("/events"));
    const body = eventCall!.body as { events: Array<{ type: string; data: Record<string, unknown> }> };
    const tool = body.events.find((e) => e.type === "tool_call")!;
    expect(tool.data["tool"]).toBe("fetch_orders");
  });

  it("onHandoff emits handoff event with agent names", async () => {
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

    const run = agentping.run("triage");
    const hooks = new AgentPingHooks(run);
    await hooks.onHandoff({}, { name: "Triage" }, { name: "Billing" });

    await agentping.flush({ timeoutMs: 1_000 });

    const eventCall = calls.find((c) => c.url.includes("/events"));
    const body = eventCall!.body as { events: Array<{ type: string; data: Record<string, unknown> }> };
    const handoff = body.events.find((e) => e.type === "handoff")!;
    expect(handoff.data["from"]).toBe("Triage");
    expect(handoff.data["to"]).toBe("Billing");
  });

  it("handles model objects without a model field", async () => {
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

    const run = agentping.run("triage");
    const hooks = new AgentPingHooks(run);
    await hooks.onLLMEnd({}, { name: "agent", model: { provider: "openai" } }, { usage: {} });

    await agentping.flush({ timeoutMs: 1_000 });

    const eventCall = calls.find((c) => c.url.includes("/events"));
    const body = eventCall!.body as { events: Array<{ type: string; data: Record<string, unknown> }> };
    const llm = body.events.find((e) => e.type === "llm_call")!;
    expect(llm.data["model"]).toBe("unknown");
  });
});
