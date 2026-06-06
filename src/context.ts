/**
 * Active-run context, propagated via Node's AsyncLocalStorage.
 *
 * AsyncLocalStorage is the Node equivalent of Python's ContextVar: it
 * isolates value propagation across `await` boundaries and concurrent
 * async tasks, so two concurrent requests in a web server each see their
 * own active run.
 *
 * Use `runScope(run, fn)` to wrap a chunk of work with an active run.
 * Instrumentation wrappers fall back to the active run when no `run`
 * option was passed to them.
 */

import { AsyncLocalStorage } from "node:async_hooks";

export interface RunLike {
  event: (type: string, data: Record<string, unknown>) => void;
}

const storage = new AsyncLocalStorage<RunLike>();

/** Returns the currently-scoped active run, or undefined if none. */
export function getActiveRun(): RunLike | undefined {
  return storage.getStore();
}

/** Run `fn` inside a scope where `run` is the active run. */
export function runScope<T>(run: RunLike, fn: () => T): T {
  return storage.run(run, fn);
}

/** Async variant: returns the awaited value of `fn` with `run` active for its duration. */
export async function runScopeAsync<T>(run: RunLike, fn: () => Promise<T>): Promise<T> {
  return storage.run(run, fn);
}
