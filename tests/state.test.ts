import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import * as agentping from "../src/index.js";
import { defaultBaseUrlForRegion, getState, shutdown } from "../src/state.js";
import { _resetWarningsForTests } from "../src/warnings.js";

const FETCH_NOOP = vi.fn(async () => new Response("{}", { status: 202 }));

describe("regional base_url defaults", () => {
  beforeEach(() => {
    _resetWarningsForTests();
    delete process.env["AGENTPING_BASE_URL"];
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
  });

  afterEach(() => {
    shutdown();
    delete process.env["AGENTPING_BASE_URL"];
    vi.restoreAllMocks();
  });

  it("EU key picks eu.ingest.agentping.io", () => {
    agentping.init({
      apiKey: `apk_eu_${"a".repeat(32)}`,
      fetchImpl: FETCH_NOOP as unknown as typeof fetch,
    });
    expect(getState()!.baseUrl).toBe("https://eu.ingest.agentping.io");
    expect(getState()!.region).toBe("eu");
  });

  it("US key picks us.ingest.agentping.io", () => {
    agentping.init({
      apiKey: `apk_us_${"f".repeat(32)}`,
      fetchImpl: FETCH_NOOP as unknown as typeof fetch,
    });
    expect(getState()!.baseUrl).toBe("https://us.ingest.agentping.io");
    expect(getState()!.region).toBe("us");
  });

  it("explicit baseUrl overrides the region default", () => {
    agentping.init({
      apiKey: `apk_us_${"f".repeat(32)}`,
      baseUrl: "https://api.test",
      fetchImpl: FETCH_NOOP as unknown as typeof fetch,
    });
    expect(getState()!.baseUrl).toBe("https://api.test");
  });

  it("AGENTPING_BASE_URL env overrides the region default", () => {
    process.env["AGENTPING_BASE_URL"] = "https://override.example";
    agentping.init({
      apiKey: `apk_us_${"f".repeat(32)}`,
      fetchImpl: FETCH_NOOP as unknown as typeof fetch,
    });
    expect(getState()!.baseUrl).toBe("https://override.example");
  });

  it("missing key falls back to the EU host", () => {
    agentping.init({
      apiKey: "",
      fetchImpl: FETCH_NOOP as unknown as typeof fetch,
    });
    expect(getState()!.baseUrl).toBe("https://eu.ingest.agentping.io");
    expect(getState()!.region).toBe("eu");
  });

  it("helper resolves known regions and falls back for unknown ones", () => {
    expect(defaultBaseUrlForRegion("eu")).toBe("https://eu.ingest.agentping.io");
    expect(defaultBaseUrlForRegion("us")).toBe("https://us.ingest.agentping.io");
    expect(defaultBaseUrlForRegion("zz")).toBe("https://eu.ingest.agentping.io");
  });
});
