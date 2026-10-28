/**
 * Time to first token and opt-in prompt capture for the Gemini, Mistral,
 * Cohere and Bedrock wrappers. Anthropic and OpenAI are covered in
 * version-and-content.test.ts.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import * as agentping from "../src/index.js";
import { instrumentBedrock } from "../src/instrumentation/bedrock.js";
import { instrumentCohere } from "../src/instrumentation/cohere.js";
import { instrumentGemini } from "../src/instrumentation/gemini.js";
import { instrumentMistral } from "../src/instrumentation/mistral.js";
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

function llmCall(calls: Call[]): LlmCall {
  const found = calls
    .filter((c) => c.url.includes("/events"))
    .flatMap((c) => c.body["events"] as Array<{ type: string; data: LlmCall }>)
    .filter((e) => e.type === "llm_call")
    .map((e) => e.data);
  expect(found).toHaveLength(1);
  return found[0]!;
}

async function* delayed<T>(items: T[], delayMs = 12): AsyncGenerator<T> {
  let first = true;
  for (const item of items) {
    if (first) {
      await new Promise((resolve) => setTimeout(resolve, delayMs));
      first = false;
    }
    yield item;
  }
}

async function drain(stream: AsyncIterable<unknown>): Promise<void> {
  for await (const _chunk of stream) {
    // drain
  }
}

function expectTtft(data: LlmCall): void {
  expect(data["stream"]).toBe(true);
  expect(typeof data["ttft_ms"]).toBe("number");
  expect(data["ttft_ms"] as number).toBeGreaterThanOrEqual(10);
  expect(data["ttft_ms"] as number).toBeLessThanOrEqual(data["latency_ms"] as number);
}

describe("provider prompt capture and time to first token", () => {
  beforeEach(() => {
    _resetWarningsForTests();
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
  });
  afterEach(() => {
    agentping.shutdown();
    vi.restoreAllMocks();
  });

  // -- Gemini ----------------------------------------------------------------

  const geminiResponse = (text: string, extra: Record<string, unknown> = {}) => ({
    modelVersion: "gemini-2.0-flash",
    candidates: [{ finishReason: "STOP", content: { parts: [{ text }] } }],
    usageMetadata: { promptTokenCount: 5, candidatesTokenCount: 2 },
    ...extra,
  });

  it("Gemini sends nothing about the prompt by default", async () => {
    const { calls, fetchImpl } = capture();
    init(fetchImpl);
    const client = instrumentGemini(
      { models: { generateContent: vi.fn(async (_args?: unknown) => geminiResponse("Paris.")) } },
      { run: agentping.run("research") },
    );
    await client.models.generateContent({ model: "gemini-2.0-flash", contents: "Capital of France?" });
    await agentping.flush({ timeoutMs: 1_000 });

    const data = llmCall(calls);
    expect("input" in data).toBe(false);
    expect("output" in data).toBe(false);
    expect("system_instructions" in data).toBe(false);
  });

  it("Gemini capture maps contents roles and the system instruction", async () => {
    const { calls, fetchImpl } = capture();
    init(fetchImpl, { captureLlmContent: true });
    const client = instrumentGemini(
      { models: { generateContent: vi.fn(async (_args?: unknown) => geminiResponse("Paris.")) } },
      { run: agentping.run("research") },
    );
    await client.models.generateContent({
      model: "gemini-2.0-flash",
      contents: [
        { role: "user", parts: [{ text: "Capital of France?" }] },
        { role: "model", parts: [{ text: "Thinking..." }, { functionCall: { name: "lookup", args: {} } }] },
        { role: "user", parts: [{ text: "Well?" }] },
      ],
      config: { systemInstruction: "Answer briefly." },
    });
    await agentping.flush({ timeoutMs: 1_000 });

    const data = llmCall(calls);
    expect(data["system_instructions"]).toBe("Answer briefly.");
    expect(data["input"]).toEqual([
      { role: "user", content: "Capital of France?" },
      { role: "assistant", content: "Thinking...\n[functionCall lookup]" },
      { role: "user", content: "Well?" },
    ]);
    expect(data["output"]).toBe("Paris.");
    expect("ttft_ms" in data).toBe(false);
  });

  it("Gemini treats a bare string or part list as one user message", async () => {
    const { calls, fetchImpl } = capture();
    init(fetchImpl, { captureLlmContent: true });
    const client = instrumentGemini(
      { models: { generateContent: vi.fn(async (_args?: unknown) => geminiResponse("ok")) } },
      { run: agentping.run("research") },
    );
    await client.models.generateContent({ model: "gemini-2.0-flash", contents: ["Summarise:", { text: "the report" }] });
    await agentping.flush({ timeoutMs: 1_000 });

    expect(llmCall(calls)["input"]).toEqual([{ role: "user", content: "Summarise:\nthe report" }]);
  });

  it("Gemini streams report ttft_ms from the first text part and join the text", async () => {
    const { calls, fetchImpl } = capture();
    init(fetchImpl, { captureLlmContent: true });
    const chunks = [
      { candidates: [] },
      { candidates: [{ content: { parts: [{ text: "Par" }] } }] },
      geminiResponse("is."),
    ];
    const client = instrumentGemini(
      {
        models: {
          generateContent: vi.fn(async (_args?: unknown) => geminiResponse("unused")),
          generateContentStream: vi.fn(async (_args?: unknown) => delayed(chunks)),
        },
      },
      { run: agentping.run("research") },
    );
    const stream = (await client.models.generateContentStream({ model: "gemini-2.0-flash", contents: "hi" })) as AsyncIterable<unknown>;
    await drain(stream);
    await agentping.flush({ timeoutMs: 1_000 });

    const data = llmCall(calls);
    expectTtft(data);
    expect(data["output"]).toBe("Paris.");
    expect(data["input"]).toEqual([{ role: "user", content: "hi" }]);
    expect(data["input_tokens"]).toBe(5);
  });

  // -- Mistral ---------------------------------------------------------------

  it("Mistral capture sends messages and the first choice's content", async () => {
    const { calls, fetchImpl } = capture();
    init(fetchImpl, { captureLlmContent: true });
    const client = instrumentMistral(
      {
        chat: {
          complete: vi.fn(async (_args?: unknown) => ({
            model: "mistral-large-latest",
            usage: { promptTokens: 20, completionTokens: 4 },
            choices: [
              {
                finishReason: "stop",
                message: { content: [{ type: "text", text: "Bonjour" }, { type: "reference", referenceIds: [1] }] },
              },
            ],
          })),
        },
      },
      { run: agentping.run("translate") },
    );
    await client.chat.complete({
      model: "mistral-large-latest",
      messages: [
        { role: "system", content: "Translate to French." },
        { role: "user", content: "Hello" },
      ],
    });
    await agentping.flush({ timeoutMs: 1_000 });

    const data = llmCall(calls);
    expect(data["input"]).toEqual([
      { role: "system", content: "Translate to French." },
      { role: "user", content: "Hello" },
    ]);
    expect(data["output"]).toBe("Bonjour\n[reference]");
    expect("ttft_ms" in data).toBe(false);
  });

  it("Mistral streams report ttft_ms from the first content delta and capture deltas", async () => {
    const { calls, fetchImpl } = capture();
    init(fetchImpl, { captureLlmContent: true });
    const event = (content: string, usage?: Record<string, number>) => ({
      data: {
        model: "mistral-large-latest",
        usage,
        choices: [{ delta: { role: "assistant", content } }],
      },
    });
    const client = instrumentMistral(
      {
        chat: {
          complete: vi.fn(async (_args?: unknown) => ({})),
          stream: vi.fn(async (_args?: unknown) =>
            delayed([event(""), event("Bon"), event("jour", { promptTokens: 20, completionTokens: 4 })]),
          ),
        },
      },
      { run: agentping.run("translate") },
    );
    const stream = (await client.chat.stream({
      model: "mistral-large-latest",
      messages: [{ role: "user", content: "Hello" }],
    })) as AsyncIterable<unknown>;
    await drain(stream);
    await agentping.flush({ timeoutMs: 1_000 });

    const data = llmCall(calls);
    expectTtft(data);
    expect(data["output"]).toBe("Bonjour");
    expect(data["input"]).toEqual([{ role: "user", content: "Hello" }]);
    expect(data["input_tokens"]).toBe(20);
  });

  // -- Cohere ----------------------------------------------------------------

  it("Cohere capture sends messages and the response text", async () => {
    const { calls, fetchImpl } = capture();
    init(fetchImpl, { captureLlmContent: true });
    const client = instrumentCohere(
      {
        chat: vi.fn(async (_args?: unknown) => ({
          usage: { billedUnits: { inputTokens: 30, outputTokens: 3 } },
          finishReason: "COMPLETE",
          message: { content: [{ type: "text", text: "Yes, in scope." }] },
        })),
      },
      { run: agentping.run("rfp") },
    );
    await client.chat({ model: "command-a-03-2025", messages: [{ role: "user", content: "Is SSO in scope?" }] });
    await agentping.flush({ timeoutMs: 1_000 });

    const data = llmCall(calls);
    expect(data["input"]).toEqual([{ role: "user", content: "Is SSO in scope?" }]);
    expect(data["output"]).toBe("Yes, in scope.");
    expect(data["input_tokens"]).toBe(30);
  });

  it("Cohere streams report ttft_ms from content-start and join content deltas", async () => {
    const { calls, fetchImpl } = capture();
    init(fetchImpl, { captureLlmContent: true });
    const delta = (text: string) => ({ type: "content-delta", delta: { message: { content: { text } } } });
    const client = instrumentCohere(
      {
        chat: vi.fn(async (_args?: unknown) => ({})),
        chatStream: vi.fn(async (_args?: unknown) =>
          delayed([
            { type: "message-start" },
            { type: "content-start" },
            delta("Yes, "),
            delta("in scope."),
            { type: "content-end" },
            {
              type: "message-end",
              delta: { usage: { billedUnits: { inputTokens: 30, outputTokens: 3 } }, finishReason: "COMPLETE" },
            },
          ]),
        ),
      },
      { run: agentping.run("rfp") },
    );
    const stream = (await client.chatStream({
      model: "command-a-03-2025",
      messages: [{ role: "user", content: "Is SSO in scope?" }],
    })) as AsyncIterable<unknown>;
    await drain(stream);
    await agentping.flush({ timeoutMs: 1_000 });

    const data = llmCall(calls);
    expectTtft(data);
    expect(data["output"]).toBe("Yes, in scope.");
    expect(data["finish_reason"]).toBe("COMPLETE");
    expect(data["input"]).toEqual([{ role: "user", content: "Is SSO in scope?" }]);
  });

  // -- Bedrock ---------------------------------------------------------------

  class ConverseCommand {
    constructor(public input: Record<string, unknown>) {}
  }
  class ConverseStreamCommand {
    constructor(public input: Record<string, unknown>) {}
  }
  class InvokeModelCommand {
    constructor(public input: Record<string, unknown>) {}
  }

  const bedrockInput = {
    modelId: "eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
    system: [{ text: "Be terse." }],
    messages: [
      { role: "user", content: [{ text: "Refund?" }, { image: { format: "png" } }] },
      { role: "assistant", content: [{ toolUse: { toolUseId: "t1", name: "lookup_order", input: {} } }] },
      { role: "user", content: [{ toolResult: { toolUseId: "t1", content: [{ text: "ok" }] } }] },
    ],
  };

  it("Bedrock Converse capture renders content blocks and the system prompt", async () => {
    const { calls, fetchImpl } = capture();
    init(fetchImpl, { captureLlmContent: true });
    const client = instrumentBedrock(
      {
        send: vi.fn(async (_command: unknown) => ({
          output: { message: { role: "assistant", content: [{ text: "Refund approved." }] } },
          stopReason: "end_turn",
          usage: { inputTokens: 40, outputTokens: 3 },
        })),
      },
      { run: agentping.run("support") },
    );
    await client.send(new ConverseCommand(bedrockInput));
    await agentping.flush({ timeoutMs: 1_000 });

    const data = llmCall(calls);
    expect(data["system_instructions"]).toBe("Be terse.");
    expect(data["input"]).toEqual([
      { role: "user", content: "Refund?\n[image]" },
      { role: "assistant", content: "[toolUse lookup_order]" },
      { role: "user", content: "[toolResult]" },
    ]);
    expect(data["output"]).toBe("Refund approved.");
    expect("ttft_ms" in data).toBe(false);
  });

  it("Bedrock ConverseStream reports ttft_ms and captures the text deltas", async () => {
    const { calls, fetchImpl } = capture();
    init(fetchImpl, { captureLlmContent: true });
    const client = instrumentBedrock(
      {
        send: vi.fn(async (_command: unknown) => ({
          stream: delayed([
            { messageStart: { role: "assistant" } },
            { contentBlockStart: { start: {} } },
            { contentBlockDelta: { delta: { text: "Refund " } } },
            { contentBlockDelta: { delta: { text: "approved." } } },
            { messageStop: { stopReason: "end_turn" } },
            { metadata: { usage: { inputTokens: 40, outputTokens: 3 } } },
          ]),
        })),
      },
      { run: agentping.run("support") },
    );
    const response = (await client.send(new ConverseStreamCommand(bedrockInput))) as { stream: AsyncIterable<unknown> };
    await drain(response.stream);
    await agentping.flush({ timeoutMs: 1_000 });

    const data = llmCall(calls);
    expectTtft(data);
    expect(data["output"]).toBe("Refund approved.");
    expect(data["system_instructions"]).toBe("Be terse.");
    expect((data["input"] as unknown[])[0]).toEqual({ role: "user", content: "Refund?\n[image]" });
    expect(data["finish_reason"]).toBe("end_turn");
  });

  it("Bedrock InvokeModel is not captured", async () => {
    const { calls, fetchImpl } = capture();
    init(fetchImpl, { captureLlmContent: true });
    const client = instrumentBedrock(
      {
        send: vi.fn(async (_command: unknown) => ({
          body: new Uint8Array(),
          $metadata: {
            httpHeaders: { "x-amzn-bedrock-input-token-count": "7", "x-amzn-bedrock-output-token-count": "2" },
          },
        })),
      },
      { run: agentping.run("support") },
    );
    await client.send(new InvokeModelCommand({ modelId: "amazon.titan-text-lite-v1", body: "{}" }));
    await agentping.flush({ timeoutMs: 1_000 });

    const data = llmCall(calls);
    expect(data["input_tokens"]).toBe(7);
    expect("input" in data).toBe(false);
    expect("output" in data).toBe(false);
  });
});
