import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import * as agentping from "../src/index.js";
import { instrumentOpenAI } from "../src/instrumentation/openai.js";
import { _resetWarningsForTests } from "../src/warnings.js";

const VALID_KEY = `apk_eu_${"a".repeat(32)}`;

describe("instrumentOpenAI", () => {
  beforeEach(() => {
    _resetWarningsForTests();
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
  });

  afterEach(() => {
    agentping.shutdown();
    vi.restoreAllMocks();
  });

  it("captures token usage with cached subtraction", async () => {
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
            model: "gpt-4o-2024-08-06",
            usage: {
              prompt_tokens: 200,
              completion_tokens: 80,
              prompt_tokens_details: { cached_tokens: 50 },
            },
          })),
        },
      },
    };

    const run = agentping.run("agent");
    const wrapped = instrumentOpenAI(fakeClient, { run });
    await wrapped.chat.completions.create({
      model: "gpt-4o-2024-08-06",
      messages: [{ role: "user", content: "hi" }],
    });

    await agentping.flush({ timeoutMs: 1_000 });

    const eventCall = calls.find((c) => c.url.includes("/events"));
    expect(eventCall).toBeTruthy();
    const body = eventCall!.body as { events: Array<{ type: string; data: Record<string, unknown> }> };
    const llm = body.events.find((e) => e.type === "llm_call");
    expect(llm).toBeTruthy();
    expect(llm!.data["provider"]).toBe("openai");
    expect(llm!.data["model"]).toBe("gpt-4o-2024-08-06");
    expect(llm!.data["input_tokens"]).toBe(150);
    expect(llm!.data["cached_input_tokens"]).toBe(50);
    expect(llm!.data["output_tokens"]).toBe(80);
    expect("cost_usd" in llm!.data).toBe(false);
  });

  it("emits batch mode when configured", async () => {
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
            usage: { prompt_tokens: 10, completion_tokens: 5 },
          })),
        },
      },
    };

    const run = agentping.run("agent");
    const wrapped = instrumentOpenAI(fakeClient, { run, mode: "batch" });
    await wrapped.chat.completions.create({ model: "gpt-4o-mini", messages: [] });
    await agentping.flush({ timeoutMs: 1_000 });

    const eventCall = calls.find((c) => c.url.includes("/events"));
    const body = eventCall!.body as { events: Array<{ type: string; data: Record<string, unknown> }> };
    const llm = body.events.find((e) => e.type === "llm_call");
    expect(llm!.data["mode"]).toBe("batch");
  });

  it("propagates errors without swallowing them", async () => {
    agentping.init({
      apiKey: VALID_KEY,
      baseUrl: "https://api.example.com",
      flushIntervalMs: 60_000,
      fetchImpl: (() => new Promise(() => undefined)) as unknown as typeof fetch,
    });

    const fakeClient = {
      chat: {
        completions: {
          create: vi.fn(async (_args?: unknown) => {
            throw new Error("rate limited");
          }),
        },
      },
    };
    const run = agentping.run("agent");
    const wrapped = instrumentOpenAI(fakeClient, { run });
    await expect(
      wrapped.chat.completions.create({ model: "x", messages: [] }),
    ).rejects.toThrow("rate limited");
  });

  it("wraps streaming chunks and emits usage from the final chunk", async () => {
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

    async function* gen(): AsyncGenerator<unknown> {
      yield { model: "gpt-4o-mini", choices: [{ delta: { content: "h" } }] };
      yield { model: "gpt-4o-mini", choices: [{ delta: { content: "i" } }] };
      yield {
        model: "gpt-4o-mini",
        choices: [{ delta: {}, finish_reason: "stop" }],
        usage: { prompt_tokens: 8, completion_tokens: 2 },
      };
    }
    const stream = gen();

    const fakeClient = {
      chat: {
        completions: {
          create: vi.fn(async (_args?: unknown) => stream),
        },
      },
    };

    const run = agentping.run("agent");
    const wrapped = instrumentOpenAI(fakeClient, { run });
    const result = (await wrapped.chat.completions.create({
      model: "gpt-4o-mini",
      stream: true,
      messages: [],
    })) as AsyncIterable<unknown>;

    const chunks: unknown[] = [];
    for await (const c of result) chunks.push(c);
    expect(chunks.length).toBe(3);

    await agentping.flush({ timeoutMs: 1_000 });

    const eventCall = calls.find((c) => c.url.includes("/events"));
    expect(eventCall).toBeTruthy();
    const body = eventCall!.body as { events: Array<{ type: string; data: Record<string, unknown> }> };
    const llm = body.events.find((e) => e.type === "llm_call");
    expect(llm!.data["model"]).toBe("gpt-4o-mini");
    expect(llm!.data["input_tokens"]).toBe(8);
    expect(llm!.data["output_tokens"]).toBe(2);
  });
});
