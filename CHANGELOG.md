# Changelog

All notable changes to the AgentPing TypeScript SDK are documented
here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and this project
adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0] - 2026-09-17

### Fixed

- `input_tokens` is now the gross prompt size, including any cached
  prefix. Anthropic and Bedrock Converse report `input_tokens` net of
  the cache, and earlier releases forwarded that number as-is. Ingest
  subtracts `cached_input_tokens` itself when it prices a call, so
  cached calls were priced too low. Every wrapper now reports gross.
- `withAgentPing()` no longer emits a stray `finish_reason` event. The
  finish reason is a field on `llm_call`.
- A stream that fails part way through emits one errored `llm_call`
  only. Earlier builds also emitted a success-shaped `llm_call` when
  the broken stream was closed.

### Added

- `llm_call` carries `finish_reason`, `tool_calls` (count),
  `reasoning_tokens` and `cache_creation_input_tokens` where the
  provider reports them, and `stream: true` on streamed calls.
- Provider errors are recorded. A rejected call emits `llm_call` with
  `status: "error"`, `error` (message), `exception` (class name) and
  `latency_ms`, then rethrows.
- `instrumentOpenAI()` wraps `responses.create`, streamed and not,
  reading usage, `status` and `incomplete_details` from the terminal
  event.
- `withAgentPing()` emits one `tool_call` per tool, with `status`,
  `tool_invocation_id`, `latency_ms`, `input` and `output`, and records
  tool errors from `tool-error` content parts. Multi-step calls sum
  usage over `steps[]` once per call.
- `AgentPingHooks.attach(emitter)` subscribes to the Runner or Agent
  events of `@openai/agents` (`agent_start`, `agent_end`,
  `agent_handoff`, `agent_tool_start`, `agent_tool_end`). Tools emit
  `tool_call` joined to their start by `callId`; handoffs and agent
  turns emit `step` events with `kind`.
- `AgentPingLangChainCallbackHandler` reads `usage_metadata` (gross
  input, cached, cache creation and reasoning tokens), emits
  `tool_call` on `handleToolEnd` and `handleToolError`, and records
  `handleLLMError` as an errored `llm_call`. Provider is inferred from
  the model class name before the model id.
- `captureToolPayloads` and `toolPayloadMaxChars` options on the
  LangChain handler and the Agents hooks. Payloads are stringified and
  capped at 4000 characters by default.
- `InstrumentOptions` and `ToolPayloadOptions` are exported.

### Changed

- `AgentPingHooks.onToolStart` no longer emits an event; `onToolEnd`
  emits the `tool_call`. `onHandoff` emits `step` with
  `kind: "handoff"` instead of a `handoff` event.
- The LangChain handler no longer emits `llm_call_error` or the
  `args_preview` field on tool events.

## [0.1.0] - 2026-05-17

Initial public release.

### Added

- Core run lifecycle. `agentping.init(options)`, `agentping.run(name,
  opts?)`, `run.event(type, payload)`, `run.finish({ status, scores })`.
- `runScope(name, fn)` and `runScopeAsync(name, fn)` for callers that
  prefer scoped execution over manual `finish()`.
- Client-generated UUIDv7 run IDs. `run.id` is populated synchronously
  before any network call.
- `agentping.heartbeat(agent, { status, costUsd, durationMs, metadata })`
  for cron-shaped jobs.
- Bounded background queue (default 1000), drop-oldest on overflow,
  exposed via `agentping.status()`.
- `atexit` flush with a 5-second deadline. `agentping.flush({ timeoutMs })`
  for explicit drains.
- Region-aware default base URL. `apk_eu_*` keys route to
  `https://eu.ingest.agentping.io`; `apk_us_*` keys route to
  `https://us.ingest.agentping.io`. Override via `AGENTPING_BASE_URL`
  or the `baseUrl` init option.

### Auto-instrumentation

- LLM providers: `instrumentAnthropic()`, `instrumentOpenAI()`,
  `instrumentGemini()`, `instrumentMistral()`, `instrumentCohere()`,
  `instrumentBedrock()`. Streaming, embeddings, and prompt-cache
  attribution are captured. AsyncIterable wraps preserve the original
  stream's chunk-by-chunk delivery. Bedrock covers Converse,
  ConverseStream, InvokeModel, and InvokeModelWithResponseStream.
- Frameworks: `AgentPingLangChainCallbackHandler`, `withAgentPing()`
  for Vercel AI SDK, `AgentPingHooks` for the OpenAI Agents SDK.

### Distribution

- Packaged as `@agentping/sdk`. ESM only. Node 18 or later.
- The `prepack` and `prepublishOnly` scripts build `dist/` before
  packing. Do not remove them; without these the published tarball
  ships no compiled JavaScript.

## Notes on stability

The 0.x line is pre-1.0. Public API may change before 1.0.0. We do not
break the wire format between SDK and ingest without a version bump and
a migration note here.

[Unreleased]: https://github.com/agent-ping/agent-ping-typescript/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/agent-ping/agent-ping-typescript/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/agent-ping/agent-ping-typescript/releases/tag/v0.1.0
