# @agentping/sdk

Production observability for AI agents. Spend, Pulse, Verify.
[agentping.io](https://agentping.io) ·
[docs](https://agentping.io/docs/sdks/typescript).

## Install

```bash
npm install @agentping/sdk
```

Node 18 or later. ESM only.

## Quickstart

```ts
import * as agentping from "@agentping/sdk";

agentping.init({ apiKey: process.env.AGENTPING_API_KEY });

const run = agentping.run("support-triage", { customerId: "cust_123" });
run.event("log", { message: "classified ticket" });
run.event("llm_call", {
  provider: "anthropic",
  model: "claude-sonnet-4-5",
  input_tokens: 1024,
  output_tokens: 312,
  latency_ms: 1430,
});
await run.finish({ status: "success", scores: { confidence: 0.93 } });
```

`run.id` is populated synchronously before any network call. Safe to
log, store, and return to your caller immediately.

## Heartbeats

For cron jobs and one-shot scripts:

```ts
agentping.heartbeat("daily-summary", {
  status: "ok",
  costUsd: 0.084,
  durationMs: 12_300,
  metadata: { rows: 421 },
});
```

## Guard (spend safety net)

Put a guard at the top of a scheduled or autonomous script. It makes one awaited
call to AgentPing, which checks your configured rules against your real
cumulative spend, and refuses to start the run if a rule is tripped (or if an
operator has paused the agent from the dashboard). Guard is on the Team and
Business plans.

```ts
// Hard mode (default): throws on block, so the run does not proceed.
await agentping.guard.check({
  customerRef: "acme-corp",
  agent: "nightly-summariser",
  function: "run",
});
```

By default a guard fails closed: `mode: "hard"` throws `agentping.Paused` on a
block, and `onUnreachable: "block"` means the script does not run unless the gate
confirms it is safe. That couples your scheduled runs to AgentPing's uptime; pass
`onUnreachable: "allow"` to run when the gate is unreachable.

```ts
try {
  await agentping.guard.check({ agent: "nightly-summariser" });
} catch (e) {
  if (e instanceof agentping.Paused) {
    console.warn("blocked by", e.verdict.blockedBy, e.verdict.rules);
  }
  throw e;
}

// Soft mode: branch on the verdict instead of throwing.
const v = await agentping.guard.check({ agent: "nightly-summariser", mode: "soft" });
if (v.blocked) process.exit(0);
```

A guard at the top of a script protects against the *next invocation* starting,
not against one run that passes the check and then loops forever. Drop
`guard.check(...)` at several points to turn it into a series of gates.
Thresholds are set in USD.

## Auto-instrumentation

Wrap a provider client to capture `llm_call` events automatically. No
globals are monkey-patched.

**LLM providers:**

```ts
import Anthropic from "@anthropic-ai/sdk";
import OpenAI from "openai";
import { GoogleGenAI } from "@google/genai";
import { Mistral } from "@mistralai/mistralai";
import { CohereClientV2 } from "cohere-ai";
import { BedrockRuntimeClient } from "@aws-sdk/client-bedrock-runtime";

const run = agentping.run("answer-bot");
const anthropic = agentping.instrumentAnthropic(new Anthropic(),         { run });
const openai    = agentping.instrumentOpenAI(   new OpenAI(),            { run });
const gemini    = agentping.instrumentGemini(   new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY }), { run });
const mistral   = agentping.instrumentMistral(  new Mistral({ apiKey: process.env.MISTRAL_API_KEY }),    { run });
const cohere    = agentping.instrumentCohere(   new CohereClientV2({ token: process.env.COHERE_API_KEY }), { run });
const bedrock   = agentping.instrumentBedrock(  new BedrockRuntimeClient({ region: "eu-west-2" }),       { run });
```

Streams are supported across every wrapper. Token counts accumulate as
chunks arrive; the `llm_call` event is emitted when the stream ends. The
wrapper never delays yielding chunks to your code.

**Agent frameworks:**

```ts
// LangChain.js (callback handler)
import { ChatOpenAI } from "@langchain/openai";
const llm = new ChatOpenAI({
  callbacks: [new agentping.AgentPingLangChainCallbackHandler()],
});

// Vercel AI SDK (spread into streamText / generateText)
import { generateText } from "ai";
import { openai } from "@ai-sdk/openai";
const { text } = await generateText({
  model: openai("gpt-4o-mini"),
  prompt: "summarise this report",
  ...agentping.withAgentPing(run),
});

// OpenAI Agents SDK (Python-compatible hook shape)
import { Runner } from "@openai/agents";
const result = await Runner.run(triageAgent, "billing question", {
  hooks: new agentping.AgentPingHooks(run),
});
```

Per-integration docs:
[Anthropic](https://agentping.io/docs/providers/anthropic) ·
[OpenAI](https://agentping.io/docs/providers/openai) ·
[Gemini](https://agentping.io/docs/providers/gemini) ·
[Mistral](https://agentping.io/docs/providers/mistral) ·
[Cohere](https://agentping.io/docs/providers/cohere) ·
[Bedrock](https://agentping.io/docs/providers/bedrock) ·
[LangChain.js](https://agentping.io/docs/frameworks/langchain) ·
[Vercel AI SDK](https://agentping.io/docs/frameworks/vercel-ai) ·
[OpenAI Agents](https://agentping.io/docs/frameworks/openai-agents).

## Contract guarantees

The SDK is built to never block, crash, or leak resources in your
agent's hot path.

- **Non-blocking.** All HTTP is fire-and-forget through a background queue.
- **Hard 2s timeout** on every request via `AbortController`.
- **Bounded queue.** Default 1000 envelopes. When full, the oldest is dropped and counted.
- **Batched flushes.** Up to 50 envelopes per request, every 2 seconds or when full.
- **Exponential back-off** on retryable failures; honours `Retry-After`. Gives up after 5 attempts.
- **One warning per error class** (`auth_error`, `network_error`, `server_error`).
- **Process exit hook** flushes pending events with a 5-second deadline.

## Status and flush

```ts
const s = agentping.status();
// { queueSize, droppedCount, lastFlushTs, lastError }

await agentping.flush({ timeoutMs: 5000 });
```

## Environment variables

- `AGENTPING_API_KEY`, team API key, format `apk_<region>_<32 hex>`.
- `AGENTPING_BASE_URL`, override the default ingest URL. The default is
  region-aware: `apk_eu_*` routes to `https://eu.ingest.agentping.io`,
  `apk_us_*` routes to `https://us.ingest.agentping.io`. Override only
  when self-hosting or testing.
- `AGENTPING_CONTROL_URL`, override the control-plane URL used by
  `guard.check`. Region-derived (EU: `https://agentping.io`).
- `AGENTPING_PARENT_RUN`, parent run id for nested agents.

## License

MIT.
