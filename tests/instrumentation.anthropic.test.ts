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
          stop_reason: "tool_use",
          content: [
            { type: "text", text: "Let me check." },
            { type: "tool_use", id: "toolu_1", name: "lookup", input: {} },
          ],
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
    // Anthropic reports input_tokens net of the cache; ingest wants gross.
    expect(llm!.data["input_tokens"]).toBe(135);
    expect(llm!.data["output_tokens"]).toBe(50);
    expect(llm!.data["cached_input_tokens"]).toBe(25);
    expect(llm!.data["cache_creation_input_tokens"]).toBe(10);
    expect(llm!.data["finish_reason"]).toBe("tool_use");
    expect(llm!.data["tool_calls"]).toBe(1);
    expect("stream" in llm!.data).toBe(false);
    expect(typeof llm!.data["latency_ms"]).toBe("number");
    expect("cost_usd" in llm!.data).toBe(false);
  });

  it("emits an errored llm_call when the provider rejects", async () => {
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

    class RateLimitError extends Error {}
    const fakeClient = {
      messages: {
        create: vi.fn(async (_args?: unknown) => {
          throw new RateLimitError("rate limited");
        }),
      },
    };

    const run = agentping.run("agent");
    const wrapped = instrumentAnthropic(fakeClient, { run });
    await expect(wrapped.messages.create({ model: "claude-sonnet-4-5" })).rejects.toThrow("rate limited");

    await agentping.flush({ timeoutMs: 1_000 });

    const eventCall = calls.find((c) => c.url.includes("/events"));
    const body = eventCall!.body as { events: Array<{ type: string; data: Record<string, unknown> }> };
    const llm = body.events.find((e) => e.type === "llm_call")!;
    expect(llm.data["provider"]).toBe("anthropic");
    expect(llm.data["model"]).toBe("claude-sonnet-4-5");
    expect(llm.data["status"]).toBe("error");
    expect(llm.data["error"]).toBe("rate limited");
    expect(llm.data["exception"]).toBe("RateLimitError");
    expect(typeof llm.data["latency_ms"]).toBe("number");
    expect("input_tokens" in llm.data).toBe(false);
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

    // Real event shapes: usage arrives on message_start (input side) and
    // message_delta (cumulative output side); stop_reason rides message_delta.
    const chunks = [
      {
        type: "message_start",
        message: {
          model: "claude-haiku-4",
          usage: { input_tokens: 12, output_tokens: 1, cache_read_input_tokens: 4 },
        },
      },
      { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } },
      { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "Hi" } },
      { type: "content_block_start", index: 1, content_block: { type: "tool_use", id: "toolu_1", name: "lookup" } },
      { type: "message_delta", delta: { stop_reason: "tool_use" }, usage: { output_tokens: 8 } },
      { type: "message_stop" },
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
    expect(llm!.data["input_tokens"]).toBe(16); // 12 net + 4 cache read
    expect(llm!.data["cached_input_tokens"]).toBe(4);
    expect(llm!.data["output_tokens"]).toBe(8); // last message_delta wins, not a sum
    expect(llm!.data["finish_reason"]).toBe("tool_use");
    expect(llm!.data["tool_calls"]).toBe(1);
    expect(llm!.data["stream"]).toBe(true);
  });

  it("returns the original client unchanged if shape is unexpected", () => {
    const ugly = {} as { messages: { create: () => unknown } };
    const result = instrumentAnthropic(ugly);
    expect(result).toBe(ugly);
  });
});
