import { newId } from "./ids.js";
import { warnOnce } from "./warnings.js";
import type { SdkState } from "./state.js";

export interface RunStartOptions {
  customerId?: string;
  feature?: string;
  metadata?: Record<string, unknown>;
  parentRunId?: string;
  /**
   * What the run is supposed to achieve. Feeds AgentPing's evaluations;
   * without it the review has little to judge against.
   */
  goal?: string;
  /**
   * Close the run automatically once no event has arrived for this long
   * (60-86400). For callers that cannot be relied on to send finish: a browser
   * that gets closed, a process that gets killed. Also makes the run resumable --
   * a later event reopens it. See conversation().
   */
  idleTimeoutSeconds?: number;
  /**
   * What that silence meant. Defaults server-side to "timeout": we stopped
   * waiting, we do not know how it went. A chat should send "success".
   */
  idleTimeoutStatus?: "success" | "failed" | "timeout" | "cancelled";
  /**
   * Reuse a known run id and start time instead of minting new ones, so a
   * stateless backend can re-open the same conversation on a later request.
   */
  resume?: { id: string; startedAt: string };
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
    this.id = options.resume?.id ?? newId("run", state.region);
    this.agent = agent;
    this.startedAt = options.resume?.startedAt ?? new Date().toISOString();

    const body: Record<string, unknown> = {
      id: this.id,
      agent,
      started_at: this.startedAt,
    };
    if (options.customerId) body["customer_id"] = options.customerId;
    if (options.feature) body["feature"] = options.feature;
    if (options.goal) body["goal"] = options.goal;
    if (options.metadata) body["metadata"] = options.metadata;
    if (options.idleTimeoutSeconds !== undefined) {
      body["idle_timeout_seconds"] = options.idleTimeoutSeconds;
      if (options.idleTimeoutStatus) body["idle_timeout_status"] = options.idleTimeoutStatus;
    }
    const parent =
      options.parentRunId ??
      (typeof process !== "undefined"
        ? process.env["AGENTPING_PARENT_RUN"]
        : undefined);
    if (parent) body["parent_run_id"] = parent;

    state.queue.push({ kind: "run_start", body });
    state.worker.notifyEnqueued();
  }

  /**
   * Append an event to the run.
   *
   * @param turn Which exchange of a conversation this belongs to, counting from
   *   1. One turn is often several events -- call the model, run a tool, call it
   *   again -- and this is what groups them. Omit for work that is not a
   *   conversation.
   */
  event(type: string, data: Record<string, unknown> = {}, turn?: number): void {
    try {
      if (this.finished) {
        warnOnce(
          "client_error",
          "event() called after finish(); event still queued but server may reject.",
        );
      }
      const evtId = newId("evt", this.state.region);
      const event: Record<string, unknown> = {
        external_id: evtId,
        type,
        ts: new Date().toISOString(),
        data,
      };
      if (turn !== undefined) event["turn"] = turn;
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
