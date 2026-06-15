import { getState } from "./state.js";
import { warnOnce } from "./warnings.js";

/**
 * The guard gate: a top-of-script safety net (guard-checks-spec).
 *
 * One awaited call you put at the top of a script. It asks AgentPing whether to
 * run, given your configured rules and your real cumulative spend, and either
 * returns a verdict (soft mode) or throws {@link Paused} (hard mode, default).
 *
 * Defaults fail closed on purpose: `mode` defaults to "hard" (throw on block)
 * and `onUnreachable` to "block" (do not run unless confirmed safe). That
 * couples your scheduled runs to AgentPing's uptime; set onUnreachable:"allow"
 * to run when the gate is unreachable.
 */

const PATH = "/v1/guard/check";

export interface GuardVerdict {
  decision: "allow" | "block";
  blocked: boolean;
  blockedBy: string[];
  rules: Array<Record<string, unknown>>;
  staleAsOf: string | null;
  active: boolean;
  inactiveReason: string | null;
  guardCheckId: string | null;
  checkedAt: string | null;
}

export interface GuardCheckOptions {
  customerRef?: string;
  agent?: string;
  function?: string;
  environment?: string;
  mode?: "hard" | "soft";
  onUnreachable?: "block" | "allow";
}

/** Thrown by {@link check} in hard mode when the gate blocks. Carries the full verdict. */
export class Paused extends Error {
  readonly verdict: GuardVerdict;

  constructor(verdict: GuardVerdict) {
    const reason = verdict.blockedBy.join(", ") || "a guard rule";
    super(`AgentPing guard blocked this run (${reason}).`);
    this.name = "Paused";
    this.verdict = verdict;
  }
}

function verdictFrom(data: Record<string, unknown>): GuardVerdict {
  const decision = data["decision"] === "block" ? "block" : "allow";
  return {
    decision,
    blocked: decision === "block",
    blockedBy: Array.isArray(data["blocked_by"])
      ? (data["blocked_by"] as string[])
      : [],
    rules: Array.isArray(data["rules"])
      ? (data["rules"] as Array<Record<string, unknown>>)
      : [],
    staleAsOf: (data["stale_as_of"] as string | null) ?? null,
    active: data["active"] !== false,
    inactiveReason: (data["inactive_reason"] as string | null) ?? null,
    guardCheckId: (data["guard_check_id"] as string | null) ?? null,
    checkedAt: (data["checked_at"] as string | null) ?? null,
  };
}

function synthetic(
  decision: "allow" | "block",
  blockedBy: string[] = [],
): GuardVerdict {
  return {
    decision,
    blocked: decision === "block",
    blockedBy,
    rules: [],
    staleAsOf: null,
    active: true,
    inactiveReason: null,
    guardCheckId: null,
    checkedAt: null,
  };
}

function unreachable(
  reason: string,
  mode: string,
  onUnreachable: string,
): GuardVerdict {
  warnOnce(
    "guard_unreachable",
    `guard gate unreachable (${reason}); applying onUnreachable=${onUnreachable}.`,
  );
  const verdict =
    onUnreachable === "block"
      ? synthetic("block", ["unreachable"])
      : synthetic("allow");
  if (verdict.blocked && mode === "hard") throw new Paused(verdict);
  return verdict;
}

/**
 * Ask the gate whether to run. In hard mode (default) throws {@link Paused} on a
 * block and resolves to the verdict on an allow; in soft mode always resolves
 * to the verdict (check `.blocked`).
 *
 * A rate-limited gate (HTTP 429) is our condition, not a safety signal: retried
 * once, then leans allow even under onUnreachable:"block". An unreachable gate
 * follows onUnreachable. An inactive plan is a no-op allow plus a loud one-time
 * warning that the script is running unguarded.
 */
export async function check(
  options: GuardCheckOptions = {},
): Promise<GuardVerdict> {
  const mode = options.mode ?? "hard";
  const onUnreachable = options.onUnreachable ?? "block";

  const body: Record<string, string> = {};
  if (options.customerRef) body["customer_ref"] = options.customerRef;
  if (options.agent) body["agent"] = options.agent;
  if (options.function) body["function"] = options.function;
  if (options.environment) body["environment"] = options.environment;

  const state = getState();
  if (!state || !state.apiKey) {
    return unreachable("not_initialized", mode, onUnreachable);
  }

  let result = await state.controlClient.postRaw(PATH, body);

  if (result.status === 429) {
    await backoff(result.retryAfterMs);
    result = await state.controlClient.postRaw(PATH, body);
    if (result.status === 429 || result.status === 0) {
      // A throttle is our condition, not a safety signal: lean allow.
      return synthetic("allow");
    }
  }

  if (!result.ok || result.status !== 200) {
    return unreachable(`http_${result.status}`, mode, onUnreachable);
  }
  if (
    typeof result.body !== "object" ||
    result.body === null ||
    !("decision" in result.body)
  ) {
    return unreachable("bad_response", mode, onUnreachable);
  }

  const verdict = verdictFrom(result.body as Record<string, unknown>);

  if (!verdict.active) {
    warnOnce(
      "guard_inactive_on_plan",
      "guard is inactive on your plan; this script is running UNGUARDED. " +
        "Upgrade to the Team or Business plan to enable it.",
    );
  }

  if (verdict.blocked && mode === "hard") throw new Paused(verdict);
  return verdict;
}

function backoff(retryAfterMs: number | undefined): Promise<void> {
  const ms = Math.min(retryAfterMs ?? 500, 1000);
  return new Promise((resolve) => setTimeout(resolve, ms));
}
