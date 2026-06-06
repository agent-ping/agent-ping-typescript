import { randomBytes } from "node:crypto";

const REGION_PATTERN = /^apk_([a-z]{2})_[0-9a-f]{32}$/;
const ID_PATTERN = /^[a-z]+_[a-z]{2}_[0-9a-f]{32}$/;

let lastMs = 0;
let lastSeq = 0;

export function uuid7Hex(): string {
  let ms = Date.now();
  if (ms <= lastMs) {
    ms = lastMs;
    lastSeq += 1;
  } else {
    lastMs = ms;
    lastSeq = 0;
  }
  const seq = lastSeq & 0x0fff;

  const rand = randomBytes(10);
  // First two bytes will hold version + seq.
  rand[0] = 0x70 | ((seq >> 8) & 0x0f);
  rand[1] = seq & 0xff;
  // Variant bits 10xxxxxx in byte 2 (which corresponds to the high bits of the
  // 62-bit random tail).
  rand[2] = 0x80 | (rand[2]! & 0x3f);

  const msHi = Math.floor(ms / 0x100000000);
  const msLo = ms >>> 0;

  const bytes = Buffer.alloc(16);
  bytes[0] = (msHi >>> 8) & 0xff;
  bytes[1] = msHi & 0xff;
  bytes[2] = (msLo >>> 24) & 0xff;
  bytes[3] = (msLo >>> 16) & 0xff;
  bytes[4] = (msLo >>> 8) & 0xff;
  bytes[5] = msLo & 0xff;
  rand.copy(bytes, 6, 0, 10);

  return bytes.toString("hex");
}

export function extractRegion(apiKey: string | undefined | null): string {
  if (!apiKey) return "eu";
  const m = REGION_PATTERN.exec(apiKey);
  if (!m || !m[1]) return "eu";
  return m[1];
}

export function newId(prefix: string, region: string): string {
  return `${prefix}_${region}_${uuid7Hex()}`;
}

export function isValidId(value: string | undefined | null): boolean {
  if (!value) return false;
  return ID_PATTERN.test(value);
}
