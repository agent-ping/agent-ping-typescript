import { newId } from "./ids.js";
import { warnOnce } from "./warnings.js";
import type { SdkState } from "./state.js";

export interface RunStartOptions {
  customerId?: string;
  feature?: string;
  metadata?: Record<string, unknown>;
  parentRunId?: string;
}

export interface RunFinishOptions {
  status?: "success" | "failed" | "timeout";
  output?: Record<string, unknown>;
  scores?: Record<string, number>;
  metadata?: Record<string, unknown>;
}

export class Run {
  public readonly id: string;
  public readonly agent: string;
  public readonly startedAt: string;
  private finished = false;

  constructor(
    private readonly state: SdkState,
    agent: string,
    options: RunStartOptions = {},
  ) {
    this.id = newId("run", state.region);
    this.agent = agent;
    this.startedAt = new Date().toISOString();

    const body: Record<string, unknown> = {
      id: this.id,
      agent,
      started_at: this.startedAt,
    };
    if (options.customerId) body["customer_id"] = options.customerId;
    if (options.feature) body["feature"] = options.feature;
    if (options.metadata) body["metadata"] = options.metadata;
    const parent =
      options.parentRunId ??
      (typeof process !== "undefined"
        ? process.env["AGENTPING_PARENT_RUN"]
        : undefined);
    if (parent) body["parent_run_id"] = parent;

    state.queue.push({ kind: "run_start", body });
    state.worker.notifyEnqueued();
  }

  event(type: string, data: Record<string, unknown> = {}): void {
    try {
      if (this.finished) {
        warnOnce(
          "client_error",
          "event() called after finish(); event still queued but server may reject.",
        );
      }
      const evtId = newId("evt", this.state.region);
      const event: Record<string, unknown> = {
        id: evtId,
        type,
        ts: new Date().toISOString(),
        data,
      };
      this.state.queue.push({
        kind: "run_events",
        runId: this.id,
        event,
      });
      this.state.worker.notifyEnqueued();
    } catch {
      // never throw from observability code
    }
  }

  async finish(options: RunFinishOptions = {}): Promise<void> {
    try {
      if (this.finished) return;
      this.finished = true;
      const body: Record<string, unknown> = {
        status: options.status ?? "success",
        finished_at: new Date().toISOString(),
      };
      if (options.output) body["output"] = options.output;
      if (options.scores) body["scores"] = options.scores;
      if (options.metadata) body["metadata"] = options.metadata;

      this.state.queue.push({
        kind: "run_finish",
        runId: this.id,
        body,
      });
      this.state.worker.notifyEnqueued();
    } catch {
      // never throw from finish
    }
  }
}
