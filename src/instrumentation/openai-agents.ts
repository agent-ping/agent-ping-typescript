/**
 * OpenAI Agents SDK (TypeScript) integration.
 *
 * `@openai/agents` reports the lifecycle of a run through event emitters:
 * a `Runner` emits `agent_start`, `agent_end`, `agent_handoff`,
 * `agent_tool_start` and `agent_tool_end`. Attach AgentPing to one with:
 *
 *     const runner = new Runner();
 *     new AgentPingHooks(run).attach(runner);
 *     await runner.run(agent, input);
 *
 * The constructor argument is optional. When omitted the hooks resolve
 * the active run via AsyncLocalStorage, so under a `runScope(run, ...)`
 * block you can simply write `new AgentPingHooks().attach(runner)`.
 *
 * Emits:
 *  - `tool_call` when a tool finishes, with its arguments, result and
 *    latency (arguments and results can be turned off or capped with the
 *    `captureToolPayloads` and `toolPayloadMaxChars` options)
 *  - `step` with `kind: "handoff"` on every agent-to-agent handoff
 *  - `step` with `kind: "agent"` when an agent finishes its turn
 *
 * The Agents SDK does not expose model calls through hooks. To price
 * them, give the SDK an instrumented OpenAI client:
 *
 *     setDefaultOpenAIClient(instrumentOpenAI(new OpenAI(), { run }));
 *
 * `onLLMStart` / `onLLMEnd` remain for code that already calls them by
 * hand with a `{ usage }` response. With `init({ captureLlmContent: true })`
 * that `llm_call` also carries the `output_text` of the response's
 * assistant messages as `output`, the agent's string `instructions` as
 * `system_instructions`, and, when `onLLMStart` is given the system prompt
 * and input items, those as `system_instructions` and `input`. The SDK
 * runs the stream itself, so there is no `ttft_ms` here.
 */

import { getActiveRun } from "../context.js";
import {
  DEFAULT_TOOL_PAYLOAD_MAX_CHARS,
  putContent,
  putIfPositive,
  putIfString,
  stringifyPayload,
  type RunLike,
  type ToolPayloadOptions,
} from "./shared.js";

interface AgentsUsage {
  inputTokens?: number;
  outputTokens?: number;
  totalTokens?: number;
  inputTokensDetails?: Array<{ cached_tokens?: number }> | { cached_tokens?: number };
  outputTokensDetails?: Array<{ reasoning_tokens?: number }> | { reasoning_tokens?: number };
}

/** A `ModelResponse` output item; only assistant messages are read here. */
interface AgentsOutputItem {
  type?: string;
  role?: string;
  content?: unknown;
}

interface AgentsModelResponse {
  usage?: AgentsUsage;
  output?: AgentsOutputItem[];
}

interface AgentsAgent {
  name?: string;
  model?: unknown;
  /** A string, or a function the SDK resolves at run time (not read). */
  instructions?: unknown;
}

/** What `onLLMStart` saw, folded into the `llm_call` by `onLLMEnd`. */
interface AgentsLlmStart {
  start: number;
  systemPrompt?: unknown;
  input?: unknown;
}

interface AgentsTool {
  name?: string;
}

interface AgentsToolDetails {
  toolCall?: { callId?: string; name?: string; arguments?: string };
}

interface HookEmitter {
  on: (event: string, listener: (...args: any[]) => void) => unknown;
}

export class AgentPingHooks {
  private llmStarts = new WeakMap<object, AgentsLlmStart>();
  private agentStarts = new WeakMap<object, Map<string, number>>();
  private toolStarts = new WeakMap<object, Map<string, number>>();
  private explicitRun?: RunLike;
  private options: ToolPayloadOptions;

  constructor(run?: RunLike, options: ToolPayloadOptions = {}) {
    this.explicitRun = run;
    this.options = options;
  }

  private get run(): RunLike | undefined {
    return this.explicitRun ?? getActiveRun();
  }

  /** Subscribe to a `Runner` (or an `Agent`) from `@openai/agents`. */
  attach<T extends HookEmitter>(emitter: T): T {
    emitter.on("agent_start", (context: object, agent: AgentsAgent) => {
      void this.onAgentStart(context, agent);
    });
    emitter.on("agent_end", (context: object, agent: AgentsAgent, output: unknown) => {
      void this.onAgentEnd(context, agent, output);
    });
    emitter.on("agent_handoff", (context: object, from: AgentsAgent, to: AgentsAgent) => {
      void this.onHandoff(context, from, to);
    });
    emitter.on(
      "agent_tool_start",
      (context: object, agent: AgentsAgent, tool: AgentsTool, details?: AgentsToolDetails) => {
        void this.onToolStart(context, agent, tool, details);
      },
    );
    emitter.on(
      "agent_tool_end",
      (
        context: object,
        agent: AgentsAgent,
        tool: AgentsTool,
        result: unknown,
        details?: AgentsToolDetails,
      ) => {
        void this.onToolEnd(context, agent, tool, result, details);
      },
    );
    return emitter;
  }

  /**
   * `systemPrompt` and `inputItems` are optional: pass the resolved system
   * prompt and the model input (a string or message items) so content
   * capture can report them on the `llm_call`.
   */
  async onLLMStart(
    context: object,
    _agent: AgentsAgent,
    systemPrompt?: unknown,
    inputItems?: unknown,
  ): Promise<void> {
    this.llmStarts.set(context, { start: Date.now(), systemPrompt, input: inputItems });
  }

  async onLLMEnd(
    context: object,
    agent: AgentsAgent,
    response: AgentsModelResponse,
  ): Promise<void> {
    const startInfo = this.llmStarts.get(context);
    const start = startInfo?.start;
    const latencyMs = start ? Date.now() - start : 0;
    this.llmStarts.delete(context);

    try {
      const run = this.run;
      if (!run) return;
      const usage = response.usage ?? {};
      const model = modelName(agent);
      const data: Record<string, unknown> = {
        provider: providerFor(model),
        model,
        input_tokens: usage.inputTokens ?? 0,
        output_tokens: usage.outputTokens ?? 0,
        latency_ms: latencyMs,
      };
      putIfPositive(data, "cached_input_tokens", firstDetail(usage.inputTokensDetails, "cached_tokens"));
      putIfPositive(data, "reasoning_tokens", firstDetail(usage.outputTokensDetails, "reasoning_tokens"));
      putContent(data, {
        input: inputMessages(startInfo?.input),
        system: startInfo?.systemPrompt ?? (typeof agent.instructions === "string" ? agent.instructions : undefined),
        output: outputText(response.output),
      });
      run.event("llm_call", data);
    } catch {
      // swallow
    }
  }

  async onToolStart(
    context: object,
    _agent: AgentsAgent,
    tool: AgentsTool,
    details?: AgentsToolDetails,
  ): Promise<void> {
    const key = toolKey(tool, details);
    let starts = this.toolStarts.get(context);
    if (!starts) {
      starts = new Map();
      this.toolStarts.set(context, starts);
    }
    starts.set(key, Date.now());
  }

  async onToolEnd(
    context: object,
    _agent: AgentsAgent,
    tool: AgentsTool,
    result: unknown,
    details?: AgentsToolDetails,
  ): Promise<void> {
    const key = toolKey(tool, details);
    const starts = this.toolStarts.get(context);
    const start = starts?.get(key);
    starts?.delete(key);

    try {
      const run = this.run;
      if (!run) return;
      const capture = this.options.captureToolPayloads ?? true;
      const maxChars = this.options.toolPayloadMaxChars ?? DEFAULT_TOOL_PAYLOAD_MAX_CHARS;
      const data: Record<string, unknown> = {
        tool: tool.name ?? details?.toolCall?.name ?? "unknown",
        status: "success",
        latency_ms: start ? Date.now() - start : 0,
      };
      putIfString(data, "tool_invocation_id", details?.toolCall?.callId);
      if (capture) {
        const input = stringifyPayload(details?.toolCall?.arguments, maxChars);
        if (input !== undefined) data["input"] = input;
        const output = stringifyPayload(result, maxChars);
        if (output !== undefined) data["output"] = output;
      }
      run.event("tool_call", data);
    } catch {
      // swallow
    }
  }

  async onHandoff(
    _context: object,
    from: AgentsAgent,
    to: AgentsAgent,
  ): Promise<void> {
    try {
      const run = this.run;
      if (!run) return;
      run.event("step", {
        kind: "handoff",
        from: from.name ?? "unknown",
        to: to.name ?? "unknown",
      });
    } catch {
      // swallow
    }
  }

  async onAgentStart(context: object, agent: AgentsAgent): Promise<void> {
    let starts = this.agentStarts.get(context);
    if (!starts) {
      starts = new Map();
      this.agentStarts.set(context, starts);
    }
    starts.set(agent.name ?? "unknown", Date.now());
  }

  async onAgentEnd(
    context: object,
    agent: AgentsAgent,
    _output: unknown,
  ): Promise<void> {
    const name = agent.name ?? "unknown";
    const starts = this.agentStarts.get(context);
    const start = starts?.get(name);
    starts?.delete(name);

    try {
      const run = this.run;
      if (!run) return;
      const data: Record<string, unknown> = { kind: "agent", agent: name };
      if (start) data["latency_ms"] = Date.now() - start;
      run.event("step", data);
    } catch {
      // swallow
    }
  }
}

/**
 * The model input as messages: a plain string is the user turn; from an
 * item list only the role-bearing message items are kept (function calls
 * and their results are reported as `tool_call` events already).
 */
function inputMessages(input: unknown): unknown {
  if (typeof input === "string") return input;
  if (!Array.isArray(input)) return undefined;
  const messages = input.filter(
    (item) =>
      typeof item === "string" ||
      (item && typeof item === "object" && typeof (item as { role?: unknown }).role === "string"),
  );
  return messages.length > 0 ? messages : undefined;
}

/** The `output_text` parts of the response's assistant messages, joined. */
function outputText(output: AgentsOutputItem[] | undefined): string | undefined {
  if (!Array.isArray(output)) return undefined;
  const messages: string[] = [];
  for (const item of output) {
    if (item?.type !== "message" || !Array.isArray(item.content)) continue;
    const text = (item.content as Array<{ type?: string; text?: unknown }>)
      .filter((part) => part?.type === "output_text" && typeof part.text === "string")
      .map((part) => part.text as string)
      .join("");
    if (text) messages.push(text);
  }
  return messages.length > 0 ? messages.join("\n") : undefined;
}

function toolKey(tool: AgentsTool, details?: AgentsToolDetails): string {
  return details?.toolCall?.callId ?? tool.name ?? "unknown";
}

function modelName(agent: AgentsAgent): string {
  if (typeof agent.model === "string") return agent.model;
  const m = agent.model as { model?: string; modelId?: string } | undefined;
  return m?.model ?? m?.modelId ?? "unknown";
}

/** The Agents SDK is OpenAI-first; other providers show up through a model prefix. */
function providerFor(model: string): string {
  const slash = model.indexOf("/");
  if (slash > 0) return model.slice(0, slash);
  return "openai";
}

function firstDetail(
  details: Array<Record<string, number | undefined>> | Record<string, number | undefined> | undefined,
  key: string,
): number | undefined {
  if (!details) return undefined;
  const entry = Array.isArray(details) ? details[0] : details;
  return entry?.[key];
}
