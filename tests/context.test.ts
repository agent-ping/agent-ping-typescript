import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import * as agentping from "../src/index.js";
import { instrumentOpenAI } from "../src/instrumentation/openai.js";
import { AgentPingHooks } from "../src/instrumentation/openai-agents.js";
import { withAgentPing } from "../src/instrumentation/vercel-ai.js";
import { _resetWarningsForTests } from "../src/warnings.js";

const VALID_KEY = `apk_eu_${"a".repeat(32)}`;

describe("AsyncLocalStorage active-run scope", () => {
  beforeEach(() => {
    _resetWarningsForTests();
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
  });

  afterEach(() => {
    agentping.shutdown();
    vi.restoreAllMocks();
  });

  it("instrumentOpenAI resolves run from runScope when none is passed", async () => {
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

    const fakeClient = {
      chat: {
        completions: {
          create: vi.fn(async (_args?: unknown) => ({
            model: "gpt-4o-mini",
            usage: { prompt_tokens: 100, completion_tokens: 50 },
          })),
        },
      },
    };

    const run = agentping.run("scoped");
    const wrapped = instrumentOpenAI(fakeClient); // no { run } passed!

    await agentping.runScopeAsync(run, async () => {
      await wrapped.chat.completions.create({ model: "gpt-4o-mini", messages: [] });
    });

    await agentping.flush({ timeoutMs: 1_000 });
    const eventCall = calls.find((c) => c.url.includes("/events"));
    expect(eventCall).toBeTruthy();
    const body = eventCall!.body as { events: Array<{ type: string; data: Record<string, unknown> }> };
    const llm = body.events.find((e) => e.type === "llm_call")!;
    expect(llm.data["provider"]).toBe("openai");
    expect(llm.data["input_tokens"]).toBe(100);
  });

  it("two concurrent runScopeAsync blocks do not leak events between each other", async () => {
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

    const fakeClient = (model: string, prompt: number) => ({
      chat: {
        completions: {
          create: vi.fn(async (_args?: unknown) => {
            await new Promise((r) => setTimeout(r, 5));
            return { model, usage: { prompt_tokens: prompt, completion_tokens: 5 } };
          }),
        },
      },
    });

    const wrapped = instrumentOpenAI(fakeClient("any", 0)); // wrapper has no run binding

    const runA = agentping.run("task-a");
    const runB = agentping.run("task-b");

    await Promise.all([
      agentping.runScopeAsync(runA, async () => {
        await instrumentOpenAI(fakeClient("gpt-4o", 100)).chat.completions.create({ model: "gpt-4o", messages: [] });
      }),
      agentping.runScopeAsync(runB, async () => {
        await instrumentOpenAI(fakeClient("gpt-4o-mini", 200)).chat.completions.create({ model: "gpt-4o-mini", messages: [] });
      }),
    ]);

    await agentping.flush({ timeoutMs: 1_000 });

    // Group llm_call events by which run they landed on.
    const byRun: Record<string, Array<Record<string, unknown>>> = {};
    for (const c of calls) {
      const m = c.url.match(/\/v1\/runs\/(run_[^/]+)\/events/);
      const runId = m?.[1];
      if (!runId) continue;
      const events = (c.body as { events: Array<{ type: string; data: Record<string, unknown> }> }).events;
      for (const e of events) {
        if (e.type === "llm_call") {
          const list = byRun[runId] ?? [];
          list.push(e.data);
          byRun[runId] = list;
        }
      }
    }

    const a = byRun[runA.id] ?? [];
    const b = byRun[runB.id] ?? [];
    expect(a).toHaveLength(1);
    expect(b).toHaveLength(1);
    expect(a[0]?.["input_tokens"]).toBe(100);
    expect(b[0]?.["input_tokens"]).toBe(200);
    // suppress unused-var warning
    void wrapped;
  });

  it("AgentPingHooks resolves run from active scope when constructed with no arg", async () => {
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
    const hooks = new AgentPingHooks(); // no run argument!

    await agentping.runScopeAsync(run, async () => {
      await hooks.onLLMEnd({}, { name: "Triage", model: "gpt-4o-mini" }, { usage: { inputTokens: 50, outputTokens: 12 } });
    });

    await agentping.flush({ timeoutMs: 1_000 });
    const eventCall = calls.find((c) => c.url.includes("/events"));
    const body = eventCall!.body as { events: Array<{ type: string; data: Record<string, unknown> }> };
    const llm = body.events.find((e) => e.type === "llm_call")!;
    expect(llm.data["input_tokens"]).toBe(50);
  });

  it("withAgentPing resolves run from active scope when called with options only", async () => {
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

    const run = agentping.run("scoped-vercel");
    await agentping.runScopeAsync(run, async () => {
      const helper = withAgentPing({ provider: "openai", model: "gpt-4o-mini" });
      helper.onFinish({ usage: { inputTokens: 80, outputTokens: 20 } });
    });

    await agentping.flush({ timeoutMs: 1_000 });
    const eventCall = calls.find((c) => c.url.includes("/events"));
    const body = eventCall!.body as { events: Array<{ type: string; data: Record<string, unknown> }> };
    const llm = body.events.find((e) => e.type === "llm_call")!;
    expect(llm.data["provider"]).toBe("openai");
    expect(llm.data["input_tokens"]).toBe(80);
  });
});
