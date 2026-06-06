import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import * as agentping from "../src/index.js";
import { instrumentAnthropic } from "../src/instrumentation/anthropic.js";
import { _resetWarningsForTests } from "../src/warnings.js";

const VALID_KEY = `apk_eu_${"a".repeat(32)}`;

describe("instrumentAnthropic", () => {
  beforeEach(() => {
    _resetWarningsForTests();
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
  });

  afterEach(() => {
    agentping.shutdown();
    vi.restoreAllMocks();
  });

  it("emits llm_call event with token usage on non-streaming response", async () => {
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
      messages: {
        create: vi.fn(async (_args?: unknown) => ({
          model: "claude-sonnet-4-5",
          usage: {
            input_tokens: 100,
            output_tokens: 50,
            cache_read_input_tokens: 25,
            cache_creation_input_tokens: 10,
          },
        })),
      },
    };

    const run = agentping.run("agent");
    const wrapped = instrumentAnthropic(fakeClient, { run });
    await wrapped.messages.create({
      model: "claude-sonnet-4-5",
      max_tokens: 100,
      messages: [{ role: "user", content: "hi" }],
    });

    await agentping.flush({ timeoutMs: 1_000 });

    const eventCall = calls.find((c) => c.url.includes("/events"));
    expect(eventCall).toBeTruthy();
    const body = eventCall!.body as { events: Array<{ type: string; data: Record<string, unknown> }> };
    const llm = body.events.find((e) => e.type === "llm_call");
    expect(llm).toBeTruthy();
    expect(llm!.data["provider"]).toBe("anthropic");
    expect(llm!.data["model"]).toBe("claude-sonnet-4-5");
    expect(llm!.data["input_tokens"]).toBe(100);
    expect(llm!.data["output_tokens"]).toBe(50);
    expect(llm!.data["cached_input_tokens"]).toBe(25);
    expect(llm!.data["cache_creation_input_tokens"]).toBe(10);
    expect(typeof llm!.data["latency_ms"]).toBe("number");
    expect("cost_usd" in llm!.data).toBe(false);
  });

  it("does not crash if the wrapped client throws", async () => {
    agentping.init({
      apiKey: VALID_KEY,
      baseUrl: "https://api.example.com",
      flushIntervalMs: 60_000,
      fetchImpl: (() => new Promise(() => undefined)) as unknown as typeof fetch,
    });

    const fakeClient = {
      messages: {
        create: vi.fn(async (_args?: unknown) => {
          throw new Error("provider down");
        }),
      },
    };

    const run = agentping.run("agent");
    const wrapped = instrumentAnthropic(fakeClient, { run });
    await expect(wrapped.messages.create({})).rejects.toThrow("provider down");
  });

  it("wraps streaming responses and emits event after completion", async () => {
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

    const chunks = [
      { type: "message_start", message: { model: "claude-haiku-4", usage: { input_tokens: 12, output_tokens: 0 } } },
      { type: "content_block_delta", usage: { output_tokens: 3 } },
      { type: "content_block_delta", usage: { output_tokens: 4 } },
      { type: "message_stop", usage: { output_tokens: 1 } },
    ];

    async function* gen(): AsyncGenerator<unknown> {
      for (const c of chunks) yield c;
    }

    const stream = gen();

    const fakeClient = {
      messages: {
        create: vi.fn(async (_args?: unknown) => stream),
      },
    };

    const run = agentping.run("agent");
    const wrapped = instrumentAnthropic(fakeClient, { run });
    const result = (await wrapped.messages.create({
      model: "claude-haiku-4",
      stream: true,
      messages: [],
    })) as AsyncIterable<unknown>;

    const received: unknown[] = [];
    for await (const c of result) {
      received.push(c);
    }
    expect(received.length).toBe(chunks.length);

    await agentping.flush({ timeoutMs: 1_000 });

    const eventCall = calls.find((c) => c.url.includes("/events"));
    expect(eventCall).toBeTruthy();
    const body = eventCall!.body as { events: Array<{ type: string; data: Record<string, unknown> }> };
    const llm = body.events.find((e) => e.type === "llm_call");
    expect(llm).toBeTruthy();
    expect(llm!.data["model"]).toBe("claude-haiku-4");
    expect(llm!.data["input_tokens"]).toBe(12);
    expect(llm!.data["output_tokens"]).toBe(8);
  });

  it("returns the original client unchanged if shape is unexpected", () => {
    const ugly = {} as { messages: { create: () => unknown } };
    const result = instrumentAnthropic(ugly);
    expect(result).toBe(ugly);
  });
});
