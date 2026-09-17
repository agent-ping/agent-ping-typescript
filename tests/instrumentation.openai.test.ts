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

  it("captures gross token usage with the cached and reasoning subsets", async () => {
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
            choices: [
              {
                finish_reason: "tool_calls",
                message: {
                  role: "assistant",
                  tool_calls: [
                    { id: "call_1", type: "function", function: { name: "lookup", arguments: "{}" } },
                    { id: "call_2", type: "function", function: { name: "lookup", arguments: "{}" } },
                  ],
                },
              },
            ],
            usage: {
              prompt_tokens: 200,
              completion_tokens: 80,
              prompt_tokens_details: { cached_tokens: 50 },
              completion_tokens_details: { reasoning_tokens: 20 },
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
    // prompt_tokens is already gross; cached_tokens is the subset ingest discounts.
    expect(llm!.data["input_tokens"]).toBe(200);
    expect(llm!.data["cached_input_tokens"]).toBe(50);
    expect(llm!.data["output_tokens"]).toBe(80);
    expect(llm!.data["reasoning_tokens"]).toBe(20);
    expect(llm!.data["finish_reason"]).toBe("tool_calls");
    expect(llm!.data["tool_calls"]).toBe(2);
    expect("stream" in llm!.data).toBe(false);
    expect("cost_usd" in llm!.data).toBe(false);
  });

  it("instruments responses.create", async () => {
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
      responses: {
        create: vi.fn(async (_args?: unknown) => ({
          model: "gpt-5-mini",
          status: "completed",
          output: [
            { type: "reasoning", summary: [] },
            { type: "function_call", call_id: "call_1", name: "lookup", arguments: "{}" },
          ],
          usage: {
            input_tokens: 120,
            output_tokens: 60,
            input_tokens_details: { cached_tokens: 40 },
            output_tokens_details: { reasoning_tokens: 25 },
          },
        })),
      },
    };

    const run = agentping.run("agent");
    const wrapped = instrumentOpenAI(fakeClient, { run });
    await wrapped.responses.create({ model: "gpt-5-mini", input: "hi" });
    await agentping.flush({ timeoutMs: 1_000 });

    const eventCall = calls.find((c) => c.url.includes("/events"));
    const body = eventCall!.body as { events: Array<{ type: string; data: Record<string, unknown> }> };
    const llm = body.events.find((e) => e.type === "llm_call")!;
    expect(llm.data["provider"]).toBe("openai");
    expect(llm.data["model"]).toBe("gpt-5-mini");
    expect(llm.data["input_tokens"]).toBe(120);
    expect(llm.data["cached_input_tokens"]).toBe(40);
    expect(llm.data["output_tokens"]).toBe(60);
    expect(llm.data["reasoning_tokens"]).toBe(25);
    expect(llm.data["finish_reason"]).toBe("tool_calls");
    expect(llm.data["tool_calls"]).toBe(1);
  });

  it("reads the terminal event of a responses.create stream", async () => {
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
      yield { type: "response.created", response: { model: "gpt-5-mini", status: "in_progress" } };
      yield { type: "response.output_text.delta", delta: "hi" };
      yield {
        type: "response.incomplete",
        response: {
          model: "gpt-5-mini",
          status: "incomplete",
          incomplete_details: { reason: "max_output_tokens" },
          output: [{ type: "message", role: "assistant" }],
          usage: { input_tokens: 30, output_tokens: 10 },
        },
      };
    }

    const fakeClient = {
      responses: {
        create: vi.fn(async (_args?: unknown) => gen()),
      },
    };

    const run = agentping.run("agent");
    const wrapped = instrumentOpenAI(fakeClient, { run });
    const result = (await wrapped.responses.create({
      model: "gpt-5-mini",
      input: "hi",
      stream: true,
    })) as AsyncIterable<unknown>;
    const events: unknown[] = [];
    for await (const e of result) events.push(e);
    expect(events.length).toBe(3);

    await agentping.flush({ timeoutMs: 1_000 });

    const eventCall = calls.find((c) => c.url.includes("/events"));
    const body = eventCall!.body as { events: Array<{ type: string; data: Record<string, unknown> }> };
    const llm = body.events.find((e) => e.type === "llm_call")!;
    expect(llm.data["model"]).toBe("gpt-5-mini");
    expect(llm.data["input_tokens"]).toBe(30);
    expect(llm.data["output_tokens"]).toBe(10);
    expect(llm.data["finish_reason"]).toBe("max_output_tokens");
    expect(llm.data["stream"]).toBe(true);
    expect("tool_calls" in llm.data).toBe(false);
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
      // Tool call arguments arrive in pieces that share an index; count distinct calls.
      yield { model: "gpt-4o-mini", choices: [{ delta: { tool_calls: [{ index: 0, id: "call_1" }] } }] };
      yield { model: "gpt-4o-mini", choices: [{ delta: { tool_calls: [{ index: 0 }] } }] };
      yield {
        model: "gpt-4o-mini",
        choices: [{ delta: {}, finish_reason: "tool_calls" }],
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
    expect(chunks.length).toBe(5);

    await agentping.flush({ timeoutMs: 1_000 });

    const eventCall = calls.find((c) => c.url.includes("/events"));
    expect(eventCall).toBeTruthy();
    const body = eventCall!.body as { events: Array<{ type: string; data: Record<string, unknown> }> };
    const llm = body.events.find((e) => e.type === "llm_call");
    expect(llm!.data["model"]).toBe("gpt-4o-mini");
    expect(llm!.data["input_tokens"]).toBe(8);
    expect(llm!.data["output_tokens"]).toBe(2);
    expect(llm!.data["finish_reason"]).toBe("tool_calls");
    expect(llm!.data["tool_calls"]).toBe(1);
    expect(llm!.data["stream"]).toBe(true);
  });
});
