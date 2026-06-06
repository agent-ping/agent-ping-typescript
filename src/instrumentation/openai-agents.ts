/**
 * OpenAI Agents SDK (TypeScript) integration.
 *
 * Exposes `AgentPingHooks`, a class compatible with the `RunHooks`
 * interface from `@openai/agents`. Attach via:
 *
 *     Runner.run(agent, input, { hooks: new AgentPingHooks(run) })
 *
 * The constructor argument is optional. When omitted the hooks resolve
 * the active run via AsyncLocalStorage, so under a `runScope(run, ...)`
 * block you can simply pass `new AgentPingHooks()`.
 *
 * Emits:
 *  - `llm_call` on every model response (one per agent turn)
 *  - `tool_call` on every tool invocation
 *  - `handoff` on every agent-to-agent handoff
 */

import { getActiveRun } from "../context.js";

interface AgentsUsage {
  inputTokens?: number;
  outputTokens?: number;
  totalTokens?: number;
}

interface AgentsModelResponse {
  usage?: AgentsUsage;
}

interface AgentsAgent {
  name?: string;
  model?: unknown;
}

interface AgentsTool {
  name?: string;
}

interface RunLike {
  event: (type: string, data: Record<string, unknown>) => void;
}

export class AgentPingHooks {
  private llmStarts = new WeakMap<object, number>();
  private explicitRun?: RunLike;

  constructor(run?: RunLike) {
    this.explicitRun = run;
  }

  private get run(): RunLike | undefined {
    return this.explicitRun ?? getActiveRun();
  }

  async onLLMStart(context: object, _agent: AgentsAgent): Promise<void> {
    this.llmStarts.set(context, Date.now());
  }

  async onLLMEnd(
    context: object,
    agent: AgentsAgent,
    response: AgentsModelResponse,
  ): Promise<void> {
    const start = this.llmStarts.get(context);
    const latencyMs = start ? Date.now() - start : 0;
    this.llmStarts.delete(context);

    try {
      const run = this.run;
      if (!run) return;
      const usage = response.usage ?? {};
      const model =
        typeof agent.model === "string"
          ? agent.model
          : (agent.model as { model?: string } | undefined)?.model ?? "unknown";

      run.event("llm_call", {
        provider: "openai",
        model,
        input_tokens: usage.inputTokens ?? 0,
        output_tokens: usage.outputTokens ?? 0,
        latency_ms: latencyMs,
      });
    } catch {
      // swallow
    }
  }

  async onToolStart(
    _context: object,
    _agent: AgentsAgent,
    tool: AgentsTool,
  ): Promise<void> {
    try {
      const run = this.run;
      if (!run) return;
      run.event("tool_call", { tool: tool.name ?? "unknown" });
    } catch {
      // swallow
    }
  }

  async onToolEnd(
    _context: object,
    _agent: AgentsAgent,
    _tool: AgentsTool,
    _result: unknown,
  ): Promise<void> {
    // No-op: tool start is sufficient for traces. Override if you need
    // per-tool duration tracking.
  }

  async onHandoff(
    _context: object,
    from: AgentsAgent,
    to: AgentsAgent,
  ): Promise<void> {
    try {
      const run = this.run;
      if (!run) return;
      run.event("handoff", {
        from: from.name ?? "unknown",
        to: to.name ?? "unknown",
      });
    } catch {
      // swallow
    }
  }

  async onAgentStart(_context: object, _agent: AgentsAgent): Promise<void> {
    // No-op: the AgentPing run is the trace root; per-agent boundaries
    // are surfaced by `handoff` events. Override if you need explicit
    // agent_start events.
  }

  async onAgentEnd(
    _context: object,
    _agent: AgentsAgent,
    _output: unknown,
  ): Promise<void> {
    // No-op: see onAgentStart.
  }
}
