# Changelog

All notable changes to the AgentPing TypeScript SDK are documented
here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and this project
adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

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

- Published as `@agentping/sdk` on npm. ESM only. Node 18 or later.
- The `prepack` and `prepublishOnly` scripts build `dist/` before
  packing. Do not remove them; without these the published tarball
  ships no compiled JavaScript.

## Notes on stability

The 0.x line is pre-1.0. Public API may change before 1.0.0. We do not
break the wire format between SDK and ingest without a version bump and
a migration note here.

[Unreleased]: https://github.com/agent-ping/agent-ping-typescript/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/agent-ping/agent-ping-typescript/releases/tag/v0.1.0
