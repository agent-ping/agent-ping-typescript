import { HttpClient, HttpResult } from "./client.js";
import { BoundedQueue } from "./queue.js";
import type { Envelope, RunEventEnvelope } from "./state.js";

export interface WorkerOptions {
  queue: BoundedQueue<Envelope>;
  client: HttpClient;
  flushIntervalMs: number;
  batchSize: number;
}

const MAX_RETRIES = 5;

export class Worker {
  private readonly queue: BoundedQueue<Envelope>;
  private readonly client: HttpClient;
  private readonly flushIntervalMs: number;
  private readonly batchSize: number;
  private timer: ReturnType<typeof setInterval> | null = null;
  private inFlight: Promise<void> | null = null;
  private nextEligibleTs = 0;
  private retryAttempts = 0;

  public lastFlushTs: number | null = null;
  public lastError: string | null = null;

  constructor(opts: WorkerOptions) {
    this.queue = opts.queue;
    this.client = opts.client;
    this.flushIntervalMs = opts.flushIntervalMs;
    this.batchSize = opts.batchSize;
  }

  start(): void {
    if (this.timer) return;
    this.timer = setInterval(() => {
      void this.tick();
    }, this.flushIntervalMs);
    if (typeof this.timer === "object" && this.timer && "unref" in this.timer) {
      (this.timer as { unref: () => void }).unref();
    }
  }

  stop(): void {
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = null;
    }
  }

  notifyEnqueued(): void {
    if (this.queue.size >= this.batchSize) {
      void this.tick();
    }
  }

  private async tick(): Promise<void> {
    if (this.inFlight) {
      await this.inFlight;
      return;
    }
    if (Date.now() < this.nextEligibleTs) return;
    if (this.queue.size === 0) return;

    this.inFlight = this.flushOnce();
    try {
      await this.inFlight;
    } finally {
      this.inFlight = null;
    }
  }

  async flushAll(deadlineMs: number): Promise<void> {
    const deadline = Date.now() + deadlineMs;
    while (this.queue.size > 0 && Date.now() < deadline) {
      if (this.inFlight) {
        await this.inFlight;
        continue;
      }
      const wait = this.nextEligibleTs - Date.now();
      if (wait > 0) {
        const sleep = Math.min(wait, deadline - Date.now());
        if (sleep <= 0) break;
        await new Promise((r) => setTimeout(r, sleep));
        continue;
      }
      this.inFlight = this.flushOnce();
      try {
        await this.inFlight;
      } finally {
        this.inFlight = null;
      }
    }
  }

  private async flushOnce(): Promise<void> {
    const batch = this.queue.drain(this.batchSize);
    if (batch.length === 0) return;

    // Group consecutive run_events for the same run id into a single POST.
    const groups = groupEnvelopes(batch);

    for (const group of groups) {
      const result = await this.sendGroup(group);
      this.lastFlushTs = Date.now();
      if (!result.ok) {
        this.lastError = `status=${result.status}`;
        if (
          result.errorClass === "auth_error" ||
          result.errorClass === "client_error"
        ) {
          // 4xx (other than 429) is permanent: drop these and move on.
          this.queue.incrementDropped(group.length);
          this.retryAttempts = 0;
          continue;
        }
        // Retryable: put back at the head and schedule back-off.
        this.queue.unshiftAll(group.concat(unsentGroups(groups, group)));
        this.scheduleBackoff(result);
        return;
      }
      this.lastError = null;
      this.retryAttempts = 0;
    }
  }

  private async sendGroup(group: Envelope[]): Promise<HttpResult> {
    if (group.length === 0) {
      return { ok: true, status: 202, body: null };
    }
    const head = group[0];
    if (!head) return { ok: true, status: 202, body: null };
    switch (head.kind) {
      case "run_start": {
        // run_start envelopes are always sent individually.
        return this.attempt(() => this.client.post("/v1/runs", head.body));
      }
      case "run_finish": {
        return this.attempt(() =>
          this.client.post(`/v1/runs/${head.runId}/finish`, head.body),
        );
      }
      case "heartbeat": {
        return this.attempt(() => this.client.post("/v1/heartbeats", head.body));
      }
      case "run_events": {
        const events = group.map((e) => (e as RunEventEnvelope).event);
        return this.attempt(() =>
          this.client.post(`/v1/runs/${head.runId}/events`, { events }),
        );
      }
    }
  }

  private async attempt(fn: () => Promise<HttpResult>): Promise<HttpResult> {
    try {
      return await fn();
    } catch (err) {
      this.lastError = err instanceof Error ? err.message : String(err);
      return { ok: false, status: 0, body: null, errorClass: "network_error" };
    }
  }

  private scheduleBackoff(result: HttpResult): void {
    this.retryAttempts += 1;
    if (this.retryAttempts > MAX_RETRIES) {
      // Give up: drain the queue's head-of-line entries.
      const dropped = this.queue.drain(this.queue.size);
      this.queue.incrementDropped(dropped.length);
      this.retryAttempts = 0;
      this.nextEligibleTs = 0;
      return;
    }
    const explicit = result.retryAfterMs;
    const backoff =
      explicit !== undefined
        ? explicit
        : Math.min(30_000, 500 * Math.pow(2, this.retryAttempts - 1));
    this.nextEligibleTs = Date.now() + backoff;
  }
}

function groupEnvelopes(batch: Envelope[]): Envelope[][] {
  const groups: Envelope[][] = [];
  let current: Envelope[] = [];
  let currentRunId: string | null = null;

  for (const env of batch) {
    if (env.kind === "run_events") {
      if (currentRunId === env.runId) {
        current.push(env);
      } else {
        if (current.length > 0) groups.push(current);
        current = [env];
        currentRunId = env.runId;
      }
    } else {
      if (current.length > 0) {
        groups.push(current);
        current = [];
        currentRunId = null;
      }
      groups.push([env]);
    }
  }
  if (current.length > 0) groups.push(current);
  return groups;
}

function unsentGroups(all: Envelope[][], from: Envelope[]): Envelope[] {
  const idx = all.indexOf(from);
  if (idx < 0) return [];
  const remaining: Envelope[] = [];
  for (let i = idx + 1; i < all.length; i++) {
    const g = all[i];
    if (g) remaining.push(...g);
  }
  return remaining;
}
