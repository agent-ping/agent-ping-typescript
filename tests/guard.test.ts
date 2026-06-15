import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import * as agentping from "../src/index.js";
import { _resetWarningsForTests } from "../src/warnings.js";

const VALID_KEY = "apk_eu_" + "a".repeat(32);

function jsonResponse(
  status: number,
  body: unknown,
  headers: Record<string, string> = {},
): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json", ...headers },
  });
}

function verdict(
  decision: "allow" | "block",
  extra: Record<string, unknown> = {},
) {
  return {
    decision,
    checked_at: "2026-06-15T09:14:22Z",
    blocked_by: decision === "block" ? ["grl_eu_x"] : [],
    guard_check_id: decision === "block" ? "gck_eu_1" : null,
    rules: [],
    stale_as_of: "2026-06-15T09:13:50Z",
    active: true,
    inactive_reason: null,
    ...extra,
  };
}

function initWith(fetchMock: ReturnType<typeof vi.fn>) {
  agentping.init({
    apiKey: VALID_KEY,
    baseUrl: "https://ingest.test",
    controlUrl: "https://control.test",
    fetchImpl: fetchMock as unknown as typeof fetch,
  });
}

describe("guard.check", () => {
  beforeEach(() => {
    _resetWarningsForTests();
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
  });
  afterEach(() => {
    agentping.shutdown();
    vi.restoreAllMocks();
  });

  it("returns a verdict on allow", async () => {
    initWith(vi.fn().mockResolvedValue(jsonResponse(200, verdict("allow"))));
    const v = await agentping.guard.check({ agent: "nightly" });
    expect(v.decision).toBe("allow");
    expect(v.blocked).toBe(false);
  });

  it("throws Paused in hard mode on block", async () => {
    initWith(vi.fn().mockResolvedValue(jsonResponse(200, verdict("block"))));
    await expect(
      agentping.guard.check({ agent: "nightly" }),
    ).rejects.toBeInstanceOf(agentping.Paused);
  });

  it("returns the block verdict in soft mode", async () => {
    initWith(vi.fn().mockResolvedValue(jsonResponse(200, verdict("block"))));
    const v = await agentping.guard.check({ agent: "nightly", mode: "soft" });
    expect(v.blocked).toBe(true);
    expect(v.guardCheckId).toBe("gck_eu_1");
  });

  it("targets the control plane with the dimensions", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(jsonResponse(200, verdict("allow")));
    initWith(fetchMock);
    await agentping.guard.check({
      customerRef: "cus_abc",
      agent: "nightly",
      function: "run",
    });
    const call = fetchMock.mock.calls[0]!;
    expect(call[0]).toBe("https://control.test/v1/guard/check");
    const body = JSON.parse((call[1] as RequestInit).body as string);
    expect(body).toMatchObject({
      customer_ref: "cus_abc",
      agent: "nightly",
      function: "run",
    });
  });

  it("an inactive plan warns and allows", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    initWith(
      vi
        .fn()
        .mockResolvedValue(
          jsonResponse(
            200,
            verdict("allow", { active: false, inactive_reason: "tier" }),
          ),
        ),
    );
    const v = await agentping.guard.check({ agent: "nightly" });
    expect(v.active).toBe(false);
    expect(v.decision).toBe("allow");
    expect(
      warn.mock.calls.some((c) =>
        String(c.join(" ")).toLowerCase().includes("unguarded"),
      ),
    ).toBe(true);
  });

  it("fails closed when unreachable by default", async () => {
    initWith(vi.fn().mockRejectedValue(new Error("boom")));
    await expect(
      agentping.guard.check({ agent: "nightly" }),
    ).rejects.toBeInstanceOf(agentping.Paused);
  });

  it("can fail open when unreachable", async () => {
    initWith(vi.fn().mockRejectedValue(new Error("boom")));
    const v = await agentping.guard.check({
      agent: "nightly",
      onUnreachable: "allow",
    });
    expect(v.decision).toBe("allow");
  });

  it("leans allow on a 429 even when fail-closed", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        jsonResponse(429, { error: "rate_limited" }, { "retry-after": "0" }),
      );
    initWith(fetchMock);
    const v = await agentping.guard.check({
      agent: "nightly",
      onUnreachable: "block",
    });
    expect(v.decision).toBe("allow");
    expect(fetchMock.mock.calls.length).toBe(2); // one retry
  });

  it("follows on_unreachable on a 5xx", async () => {
    initWith(vi.fn().mockResolvedValue(jsonResponse(503, {})));
    await expect(
      agentping.guard.check({ agent: "nightly" }),
    ).rejects.toBeInstanceOf(agentping.Paused);
  });
});
