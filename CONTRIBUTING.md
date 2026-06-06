# Contributing

Thanks for considering a contribution to the AgentPing TypeScript SDK.

## Reporting a bug

Open a GitHub issue. Useful issues include:

- The version of `@agentping/sdk` you're running.
- Your Node version and OS (`node --version`).
- A minimal reproduction. Snip the surrounding application code; we
  only need the AgentPing calls and the provider client.
- What you expected to happen, what actually happened.
- Anything visible from `agentping.status()` if the bug is about
  telemetry not landing.

Security issues should not go on the public tracker. Email
`security@agentping.io` instead.

## Proposing a feature

Open an issue before opening a PR. Most feature work touches the wire
contract with the AgentPing ingest server, so we want to design the
interface together before you spend time implementing it.

If you're adding auto-instrumentation for a provider we don't cover
yet, see [Auto-instrumentation guidelines](#auto-instrumentation-guidelines)
below.

## Local development

```bash
git clone https://github.com/agent-ping/agent-ping-typescript
cd agentping-typescript
npm install
npm test
```

Node 18 or later. The SDK is ESM only; CommonJS consumers should use a
dynamic `import()`.

## Tests

```bash
npm test              # full suite (vitest)
npm test -- run.test  # filter by file name
npx tsc --noEmit      # type-check without emitting
```

We expect every PR to add or update tests. The HTTP layer is mocked, so
tests do not require credentials or a live ingest endpoint.

## Build

```bash
npm run build
ls dist/
```

The `prepack` hook runs this automatically before `npm pack` or
`npm publish`. Do not remove the hook; without it the published
tarball ships no compiled JavaScript.

## Style

- Strict TypeScript. No `any`, no `// @ts-ignore` except with a written
  reason on the line above.
- Format with `prettier`. Lint with `eslint`. Both run in CI.
- No em-dashes in source, JSDoc, comments, or commit messages. Use
  commas, semicolons, or sentence breaks. This is a project-wide rule.

## Auto-instrumentation guidelines

When adding an `instrument*` function for a new provider:

1. The patch must be idempotent. Calling it twice should be a no-op.
2. Capture token counts from the actual response, not from request
   parameters. Tokens reported pre-call are estimates and produce
   wrong spend numbers.
3. AsyncIterable wrappers must preserve the original iterator's
   behaviour: each chunk flows to the caller without buffering, and
   the `llm_call` event is emitted once when the stream closes with
   the accumulated counts.
4. Prompt-cache token attribution: split cached and uncached input
   tokens into distinct fields if the provider exposes them. The
   server rate card prices the two differently.
5. Embedding calls and chat calls go to the same `llm_call` event
   type. The `model` field distinguishes them downstream.

## Commit messages and PRs

- Subject in imperative mood (`add Vercel AI SDK helper`, not
  `added Vercel AI SDK helper`).
- One topic per PR. Bug fixes, refactors, and new instrumentation
  should not be bundled.
- If the PR changes the wire contract with ingest, link the matching
  issue in the AgentPing platform tracker.

## Releasing

Maintainers only. Release flow lives in
`agent-ping/LAUNCH.md` in the AgentPing platform monorepo.
