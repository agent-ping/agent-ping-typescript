import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import * as agentping from "../src/index.js";
import { instrumentAnthropic } from "../src/instrumentation/anthropic.js";
import { instrumentOpenAI } from "../src/instrumentation/openai.js";
import { _resetWarningsForTests } from "../src/warnings.js";

const VALID_KEY = `apk_eu_${"a".repeat(32)}`;

type Call = { url: string; body: Record<string, unknown> };
type LlmCall = Record<string, unknown>;

function capture(): { calls: Call[]; fetchImpl: typeof fetch } {
  const calls: Call[] = [];
  const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
    calls.push({ url, body: JSON.parse(init!.body as string) as Record<string, unknown> });
    return new Response("{}", { status: 202 });
  });
  return { calls, fetchImpl: fetchMock as unknown as typeof fetch };
}

function init(fetchImpl: typeof fetch, extra: Record<string, unknown> = {}): void {
  agentping.init({
    apiKey: VALID_KEY,
    baseUrl: "https://api.example.com",
    flushIntervalMs: 5,
    batchSize: 10,
    fetchImpl,
    ...extra,
  });
}

function runStart(calls: Call[], runId: string): Record<string, unknown> {
  return calls.find((c) => c.url.endsWith("/v1/runs") && c.body["id"] === runId)!.body;
}

function llmCalls(calls: Call[]): LlmCall[] {
  return calls
    .filter((c) => c.url.includes("/events"))
    .flatMap((c) => (c.body["events"] as Array<{ type: string; data: LlmCall }>))
    .filter((e) => e.type === "llm_call")
    .map((e) => e.data);
}

describe("run version", () => {
  beforeEach(() => {
    _resetWarningsForTests();
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
  });
  afterEach(() => {
    agentping.shutdown();
    vi.restoreAllMocks();
    delete process.env["AGENTPING_VERSION"];
  });

  it("defaults from init() and a run can override it", async () => {
    const { calls, fetchImpl } = capture();
    init(fetchImpl, { version: " 2026.10.27 " });

    const byDefault = agentping.run("triage");
    const override = agentping.run("triage", { version: "prompt-v7" });
    await agentping.flush({ timeoutMs: 1_000 });

    expect(runStart(calls, byDefault.id)["version"]).toBe("2026.10.27");
    expect(runStart(calls, override.id)["version"]).toBe("prompt-v7");
  });

  it("comes from AGENTPING_VERSION and is omitted when nothing sets it", async () => {
    const { calls, fetchImpl } = capture();
    process.env["AGENTPING_VERSION"] = "git-abc123";
    init(fetchImpl);
    const fromEnv = agentping.run("triage");
    await agentping.flush({ timeoutMs: 1_000 });
    agentping.shutdown();

    delete process.env["AGENTPING_VERSION"];
    init(fetchImpl);
    const unset = agentping.run("triage");
    const chat = agentping.conversation("support-chat", agentping.newSessionId(), { version: "v3" });
    await agentping.flush({ timeoutMs: 1_000 });

    expect(runStart(calls, fromEnv.id)["version"]).toBe("git-abc123");
    expect("version" in runStart(calls, unset.id)).toBe(false);
    expect(runStart(calls, chat.id)["version"]).toBe("v3");
  });
});

describe("prompt capture and time to first token", () => {
  beforeEach(() => {
    _resetWarningsForTests();
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
  });
  afterEach(() => {
    agentping.shutdown();
    vi.restoreAllMocks();
  });

  const anthropicResponse = {
    model: "claude-haiku-5-5",
    stop_reason: "end_turn",
    content: [
      { type: "text", text: "Route to billing." },
      { type: "tool_use", id: "toolu_1", name: "lookup", input: {} },
    ],
    usage: { input_tokens: 10, output_tokens: 5 },
  };

  it("sends nothing about the prompt by default", async () => {
    const { calls, fetchImpl } = capture();
    init(fetchImpl);
    const client = instrumentAnthropic(
      { messages: { create: vi.fn(async (_args?: unknown) => anthropicResponse) } },
      { run: agentping.run("triage") },
    );
    await client.messages.create({
      model: "claude-haiku-5-5",
      system: "Be terse.",
      messages: [{ role: "user", content: "hi" }],
    });
    await agentping.flush({ timeoutMs: 1_000 });

    const [data] = llmCalls(calls);
    expect(data).toBeTruthy();
    expect("input" in data!).toBe(false);
    expect("output" in data!).toBe(false);
    expect("system_instructions" in data!).toBe(false);
  });

  it("Anthropic capture sends messages, system and text blocks, clipped", async () => {
    const { calls, fetchImpl } = capture();
    init(fetchImpl, { captureLlmContent: true, llmContentMaxChars: 40 });
    const client = instrumentAnthropic(
      { messages: { create: vi.fn(async (_args?: unknown) => anthropicResponse) } },
      { run: agentping.run("triage") },
    );
    await client.messages.create({
      model: "claude-haiku-5-5",
      system: [{ type: "text", text: "Be terse." }],
      messages: [
        {
          role: "user",
          content: [{ type: "text", text: "Refund request from Acme ".repeat(10) }, { type: "image", source: {} }],
        },
        { role: "assistant", content: "Checking." },
      ],
    });
    await agentping.flush({ timeoutMs: 1_000 });

    const [data] = llmCalls(calls);
    const input = data!["input"] as Array<{ role: string; content: string }>;
    expect(data!["system_instructions"]).toBe("Be terse.");
    expect(input.map((m) => m.role)).toEqual(["user", "assistant"]);
    expect(input[0]!.content.endsWith("...")).toBe(true);
    expect(input[0]!.content.length).toBe(43);
    expect(input[1]!.content).toBe("Checking.");
    expect(data!["output"]).toBe("Route to billing.");
  });

  it("Anthropic streams report ttft_ms from the first content block and capture deltas", async () => {
    const { calls, fetchImpl } = capture();
    init(fetchImpl, { captureLlmContent: true });
    async function* gen(): AsyncGenerator<unknown> {
      yield { type: "message_start", message: { model: "claude-haiku-5-5", usage: { input_tokens: 8, output_tokens: 1 } } };
      await new Promise((resolve) => setTimeout(resolve, 15));
      yield { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } };
      yield { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "Route " } };
      yield { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "to billing." } };
      yield { type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 4 } };
      yield { type: "message_stop" };
    }
    const client = instrumentAnthropic(
      { messages: { create: vi.fn(async (_args?: unknown) => gen()) } },
      { run: agentping.run("triage") },
    );
    const stream = (await client.messages.create({
      model: "claude-haiku-5-5",
      stream: true,
      messages: [{ role: "user", content: "hi" }],
    })) as AsyncIterable<unknown>;
    for await (const _chunk of stream) {
      // drain
    }
    await agentping.flush({ timeoutMs: 1_000 });

    const [data] = llmCalls(calls);
    expect(data!["stream"]).toBe(true);
    expect(typeof data!["ttft_ms"]).toBe("number");
    expect(data!["ttft_ms"] as number).toBeGreaterThanOrEqual(10);
    expect(data!["ttft_ms"] as number).toBeLessThanOrEqual(data!["latency_ms"] as number);
    expect(data!["output"]).toBe("Route to billing.");
    expect(data!["input"]).toEqual([{ role: "user", content: "hi" }]);
  });

  it("OpenAI chat streams report ttft_ms and captured deltas with the system message", async () => {
    const { calls, fetchImpl } = capture();
    init(fetchImpl, { captureLlmContent: true });
    async function* gen(): AsyncGenerator<unknown> {
      yield { model: "gpt-5", choices: [{ delta: { content: "" } }] };
      yield { model: "gpt-5", choices: [{ delta: { content: "Route " } }] };
      yield { model: "gpt-5", choices: [{ delta: { content: "to billing." }, finish_reason: "stop" }] };
      yield { model: "gpt-5", choices: [], usage: { prompt_tokens: 30, completion_tokens: 12 } };
    }
    const client = instrumentOpenAI(
      { chat: { completions: { create: vi.fn(async (_args?: unknown) => gen()) } } },
      { run: agentping.run("triage") },
    );
    const stream = (await client.chat.completions.create({
      model: "gpt-5",
      stream: true,
      messages: [{ role: "system", content: "Be terse." }, { role: "user", content: "hi" }],
    })) as AsyncIterable<unknown>;
    for await (const _chunk of stream) {
      // drain
    }
    await agentping.flush({ timeoutMs: 1_000 });

    const [data] = llmCalls(calls);
    expect(typeof data!["ttft_ms"]).toBe("number");
    expect(data!["output"]).toBe("Route to billing.");
    expect(data!["input"]).toEqual([
      { role: "system", content: "Be terse." },
      { role: "user", content: "hi" },
    ]);
  });

  it("OpenAI Responses capture input, instructions and output_text, with no ttft off-stream", async () => {
    const { calls, fetchImpl } = capture();
    init(fetchImpl, { captureLlmContent: true });
    const client = instrumentOpenAI(
      {
        responses: {
          create: vi.fn(async (_args?: unknown) => ({
            model: "gpt-5",
            status: "completed",
            output: [],
            output_text: "Route to billing.",
            usage: { input_tokens: 10, output_tokens: 5 },
          })),
        },
      },
      { run: agentping.run("triage") },
    );
    await client.responses.create({ model: "gpt-5", instructions: "Be terse.", input: "Refund request from Acme" });
    await agentping.flush({ timeoutMs: 1_000 });

    const [data] = llmCalls(calls);
    expect(data!["system_instructions"]).toBe("Be terse.");
    expect(data!["input"]).toEqual([{ role: "user", content: "Refund request from Acme" }]);
    expect(data!["output"]).toBe("Route to billing.");
    expect("ttft_ms" in data!).toBe(false);
  });
});
