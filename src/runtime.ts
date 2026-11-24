import type Anthropic from "@anthropic-ai/sdk";
import type { ToolSpec } from "./tools.js";
import { TransientToolError, stableStringify, toApiTool } from "./tools.js";
import { Tracer } from "./trace.js";

/** Anything that can answer a Messages API request: the SDK client, or a test double. */
export interface ModelClient {
  create(params: Anthropic.MessageCreateParamsNonStreaming): Promise<Anthropic.Message>;
}

export function anthropicClient(client: Anthropic): ModelClient {
  return { create: (p) => client.messages.create(p) };
}

/** USD per million tokens. Cache read/write pricing is not modelled here. */
export const PRICING: Record<string, { input: number; output: number }> = {
  "claude-opus-5-5": { input: 4, output: 20 },
};

export interface Limits {
  maxSteps: number; // model calls
  maxToolCalls: number;
  maxCostUsd: number;
  toolTimeoutMs: number;
  toolRetries: number; // extra attempts for idempotent tools on transient errors
  maxRepeats: number; // identical (tool, args) calls tolerated before intervening
  maxTokens: number; // per model call
}

export const DEFAULT_LIMITS: Limits = {
  maxSteps: 12,
  maxToolCalls: 24,
  maxCostUsd: 0.5,
  toolTimeoutMs: 10_000,
  toolRetries: 2,
  maxRepeats: 2,
  maxTokens: 16_000,
};

export type StopReason =
  | "completed"
  | "max_steps"
  | "max_tool_calls"
  | "budget"
  | "loop_detected"
  | "refusal"
  | "max_tokens";

export type Approver = (call: { name: string; input: unknown }) => Promise<boolean> | boolean;

export interface RunOptions {
  model: ModelClient;
  modelName?: string;
  system?: string;
  tools: ToolSpec[];
  input: string | Anthropic.MessageParam[];
  limits?: Partial<Limits>;
  /** Called before every "write" tool; returning false sends a declined result. */
  approve?: Approver;
  tracer?: Tracer;
  sleep?: (ms: number) => Promise<void>;
}

export interface RunResult {
  stop: StopReason;
  text: string;
  messages: Anthropic.MessageParam[];
  steps: number;
  toolCalls: number;
  usage: { input: number; output: number };
  costUsd: number;
  /** Tool calls blocked by validation, approval, or loop guard — useful in evals. */
  interventions: { kind: string; tool: string; detail: string }[];
  trace: Tracer;
}

const textOf = (content: Anthropic.ContentBlock[]) =>
  content
    .filter((b): b is Anthropic.TextBlock => b.type === "text")
    .map((b) => b.text)
    .join("");

export async function runAgent(opts: RunOptions): Promise<RunResult> {
  const lim = { ...DEFAULT_LIMITS, ...opts.limits };
  const modelName = opts.modelName ?? "claude-opus-5-5";
  const price = PRICING[modelName];
  const tracer = opts.tracer ?? new Tracer();
  const sleep = opts.sleep ?? ((ms: number) => new Promise<void>((r) => setTimeout(r, ms)));
  const byName = new Map(opts.tools.map((t) => [t.name, t]));
  const apiTools = opts.tools.map(toApiTool);
  const messages: Anthropic.MessageParam[] =
    typeof opts.input === "string" ? [{ role: "user", content: opts.input }] : [...opts.input];
  const repeats = new Map<string, number>();
  const res: RunResult = {
    stop: "completed",
    text: "",
    messages,
    steps: 0,
    toolCalls: 0,
    usage: { input: 0, output: 0 },
    costUsd: 0,
    interventions: [],
    trace: tracer,
  };
  const finish = (stop: StopReason) => {
    res.stop = stop;
    return res;
  };

  return tracer.span("agent", { model: modelName }, async (root) => {
    for (;;) {
      if (res.steps >= lim.maxSteps) return finish("max_steps");
      res.steps++;
      const resp = await tracer.span("model", { step: res.steps }, async (s) => {
        const r = await opts.model.create({
          model: modelName,
          max_tokens: lim.maxTokens,
          ...(opts.system ? { system: opts.system } : {}),
          ...(apiTools.length ? { tools: apiTools } : {}),
          messages,
        });
        Object.assign(s.attrs, {
          stop_reason: r.stop_reason,
          input_tokens: r.usage.input_tokens,
          output_tokens: r.usage.output_tokens,
        });
        return r;
      });
      res.usage.input += resp.usage.input_tokens;
      res.usage.output += resp.usage.output_tokens;
      if (price) {
        res.costUsd = (res.usage.input * price.input + res.usage.output * price.output) / 1e6;
      }
      root.attrs.cost_usd = res.costUsd;
      // Append the assistant turn unchanged (thinking blocks must be passed back as-is).
      messages.push({ role: "assistant", content: resp.content });
      res.text = textOf(resp.content) || res.text;

      switch (resp.stop_reason) {
        case "end_turn":
        case "stop_sequence":
          return finish("completed");
        case "refusal":
          return finish("refusal");
        case "max_tokens":
          return finish("max_tokens");
        case "pause_turn":
          continue;
        default:
          break;
      }
      if (res.costUsd > lim.maxCostUsd) return finish("budget");

      const uses = resp.content.filter((b): b is Anthropic.ToolUseBlock => b.type === "tool_use");
      if (uses.length === 0) return finish("completed");
      const results: Anthropic.ToolResultBlockParam[] = [];
      let loop = false;
      for (const use of uses) {
        res.toolCalls++;
        if (res.toolCalls > lim.maxToolCalls) return finish("max_tool_calls");
        const r = await executeTool(use);
        results.push(r.block);
        loop ||= r.loop;
      }
      messages.push({ role: "user", content: results });
      if (loop) return finish("loop_detected");
    }
  });

  async function executeTool(
    use: Anthropic.ToolUseBlock,
  ): Promise<{ block: Anthropic.ToolResultBlockParam; loop: boolean }> {
    const err = (kind: string, detail: string, loop = false) => {
      res.interventions.push({ kind, tool: use.name, detail });
      return {
        block: { type: "tool_result" as const, tool_use_id: use.id, content: detail, is_error: true },
        loop,
      };
    };
    return tracer.span("tool", { tool: use.name }, async (s) => {
      const tool = byName.get(use.name);
      if (!tool) return err("unknown_tool", `Unknown tool "${use.name}".`);

      const key = `${use.name}:${stableStringify(use.input)}`;
      const n = (repeats.get(key) ?? 0) + 1;
      repeats.set(key, n);
      if (n > lim.maxRepeats + 1) return err("loop", "Repeated identical call; stopping.", true);
      if (n > lim.maxRepeats) {
        return err(
          "loop",
          `You already called ${use.name} with these exact arguments ${n - 1} times. ` +
            "The result will not change; use a different approach or answer the user.",
        );
      }

      const parsed = tool.schema.safeParse(use.input);
      if (!parsed.success) {
        const issues = parsed.error.issues
          .map((i) => `${i.path.join(".") || "(root)"}: ${i.message}`)
          .join("; ");
        return err("invalid_input", `Invalid input for ${use.name}: ${issues}`);
      }

      if (tool.effect === "write" && opts.approve) {
        const ok = await tracer.span("approval", { tool: use.name }, async (a) => {
          const v = await opts.approve!({ name: use.name, input: parsed.data });
          a.attrs.approved = v;
          return v;
        });
        if (!ok) return err("declined", "A human reviewer declined this action. Do not retry it.");
      }

      const attempts = 1 + (tool.idempotent ? lim.toolRetries : 0);
      let last = "";
      for (let attempt = 1; attempt <= attempts; attempt++) {
        s.attrs.attempts = attempt;
        const ac = new AbortController();
        let timer: ReturnType<typeof setTimeout> | undefined;
        try {
          const out = await Promise.race([
            Promise.resolve(tool.run(parsed.data, { signal: ac.signal, attempt })),
            new Promise<never>((_, rej) => {
              timer = setTimeout(() => {
                ac.abort();
                rej(new TransientToolError(`timed out after ${lim.toolTimeoutMs} ms`));
              }, lim.toolTimeoutMs);
            }),
          ]);
          const content = typeof out === "string" ? out : JSON.stringify(out);
          return { block: { type: "tool_result" as const, tool_use_id: use.id, content }, loop: false };
        } catch (e) {
          last = e instanceof Error ? e.message : String(e);
          const transient = e instanceof TransientToolError;
          if (!transient || attempt === attempts) break;
          await sleep(Math.random() * 100 * 2 ** attempt); // full jitter
        } finally {
          clearTimeout(timer);
        }
      }
      s.status = "error";
      return err("tool_error", `${use.name} failed: ${last}`);
    });
  }
}
