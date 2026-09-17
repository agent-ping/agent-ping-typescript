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

  it("emits one tool_call when a tool finishes, joined to its start by callId", async () => {
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
    const agent = { name: "Triage" };
    const tool = { name: "fetch_orders" };
    const details = {
      toolCall: { type: "function_call", callId: "call_42", name: "fetch_orders", arguments: '{"customer":"c_1"}' },
    };

    await hooks.onToolStart(ctx, agent, tool, details);
    await hooks.onToolEnd(ctx, agent, tool, '{"orders":2}', details);

    await agentping.flush({ timeoutMs: 1_000 });

    const eventCall = calls.find((c) => c.url.includes("/events"));
    const body = eventCall!.body as { events: Array<{ type: string; data: Record<string, unknown> }> };
    const toolCalls = body.events.filter((e) => e.type === "tool_call");
    expect(toolCalls).toHaveLength(1);
    expect(toolCalls[0]!.data["tool"]).toBe("fetch_orders");
    expect(toolCalls[0]!.data["status"]).toBe("success");
    expect(toolCalls[0]!.data["tool_invocation_id"]).toBe("call_42");
    expect(toolCalls[0]!.data["input"]).toBe('{"customer":"c_1"}');
    expect(toolCalls[0]!.data["output"]).toBe('{"orders":2}');
    expect(toolCalls[0]!.data["latency_ms"]).toBeTypeOf("number");
  });

  it("attach subscribes to a Runner-style emitter", async () => {
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

    const listeners = new Map<string, (...args: unknown[]) => void>();
    const runner = {
      on(event: string, listener: (...args: unknown[]) => void) {
        listeners.set(event, listener);
        return runner;
      },
    };

    const run = agentping.run("triage");
    new AgentPingHooks(run, { captureToolPayloads: false }).attach(runner);
    expect([...listeners.keys()].sort()).toEqual([
      "agent_end",
      "agent_handoff",
      "agent_start",
      "agent_tool_end",
      "agent_tool_start",
    ]);

    const ctx = {};
    const triage = { name: "Triage", model: "gpt-4o-mini" };
    const billing = { name: "Billing", model: "gpt-4o-mini" };
    const details = { toolCall: { callId: "call_1", name: "lookup", arguments: "{}" } };
    listeners.get("agent_start")!(ctx, triage);
    listeners.get("agent_tool_start")!(ctx, triage, { name: "lookup" }, details);
    listeners.get("agent_tool_end")!(ctx, triage, { name: "lookup" }, "found", details);
    listeners.get("agent_handoff")!(ctx, triage, billing);
    listeners.get("agent_end")!(ctx, billing, "done");

    await agentping.flush({ timeoutMs: 1_000 });

    const eventCall = calls.find((c) => c.url.includes("/events"));
    const body = eventCall!.body as { events: Array<{ type: string; data: Record<string, unknown> }> };
    const types = body.events.map((e) => e.type);
    expect(types).toEqual(["tool_call", "step", "step"]);
    const tool = body.events[0]!;
    expect(tool.data["tool"]).toBe("lookup");
    expect("input" in tool.data).toBe(false);
    expect("output" in tool.data).toBe(false);
    expect(body.events[1]!.data).toMatchObject({ kind: "handoff", from: "Triage", to: "Billing" });
    expect(body.events[2]!.data).toMatchObject({ kind: "agent", agent: "Billing" });
  });

  it("onHandoff emits a handoff step with agent names", async () => {
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
    const handoff = body.events.find((e) => e.type === "step")!;
    expect(handoff.data["kind"]).toBe("handoff");
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
