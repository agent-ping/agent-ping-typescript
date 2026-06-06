import { describe, expect, it } from "vitest";
import { extractRegion, isValidId, newId, uuid7Hex } from "../src/ids.js";

describe("ids", () => {
  it("uuid7Hex returns 32 lowercase hex chars", () => {
    const v = uuid7Hex();
    expect(v).toMatch(/^[0-9a-f]{32}$/);
  });

  it("uuid7Hex encodes version 7 in the right nibble", () => {
    const v = uuid7Hex();
    expect(v[12]).toBe("7");
  });

  it("uuid7Hex is monotonic when called rapidly", () => {
    const a = uuid7Hex();
    const b = uuid7Hex();
    const c = uuid7Hex();
    expect(a < b || a === b).toBe(true);
    expect(b <= c).toBe(true);
  });

  it("extractRegion pulls eu from an apk key", () => {
    const key = `apk_eu_${"a".repeat(32)}`;
    expect(extractRegion(key)).toBe("eu");
  });

  it("extractRegion pulls us from an apk key", () => {
    const key = `apk_us_${"f".repeat(32)}`;
    expect(extractRegion(key)).toBe("us");
  });

  it("extractRegion falls back to eu on malformed input", () => {
    expect(extractRegion("garbage")).toBe("eu");
    expect(extractRegion("")).toBe("eu");
    expect(extractRegion(null)).toBe("eu");
    expect(extractRegion(undefined)).toBe("eu");
  });

  it("newId composes prefix, region, and hex", () => {
    const id = newId("run", "eu");
    expect(id).toMatch(/^run_eu_[0-9a-f]{32}$/);
  });

  it("isValidId accepts well-formed ids", () => {
    expect(isValidId(`run_eu_${"a".repeat(32)}`)).toBe(true);
    expect(isValidId(`evt_us_${"0".repeat(32)}`)).toBe(true);
  });

  it("isValidId rejects malformed ids", () => {
    expect(isValidId("run_eu_short")).toBe(false);
    expect(isValidId(`RUN_eu_${"a".repeat(32)}`)).toBe(false);
    expect(isValidId(`run_EU_${"a".repeat(32)}`)).toBe(false);
    expect(isValidId(undefined)).toBe(false);
  });
});
