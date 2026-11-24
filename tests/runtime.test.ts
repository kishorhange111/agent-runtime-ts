import type Anthropic from "@anthropic-ai/sdk";
import { describe, expect, it } from "vitest";
import { z } from "zod";
import {
  type ModelClient,
  TransientToolError,
  defineTool,
  runAgent,
  stableStringify,
  toApiTool,
} from "../src/index.js";

let ids = 0;
const toolUse = (name: string, input: unknown): Anthropic.ToolUseBlock =>
  ({ type: "tool_use", id: `tu_${++ids}`, name, input }) as Anthropic.ToolUseBlock;
const text = (t: string): Anthropic.TextBlock => ({ type: "text", text: t }) as Anthropic.TextBlock;

/** Replays a fixed script of assistant turns; records every request it saw. */
function scripted(turns: { content: Anthropic.ContentBlock[]; stop?: string }[]) {
  const seen: Anthropic.MessageCreateParamsNonStreaming[] = [];
  let i = 0;
  const model: ModelClient = {
    async create(p) {
      seen.push(structuredClone(p));
      const t = turns[Math.min(i++, turns.length - 1)]!;
      const hasTool = t.content.some((b) => b.type === "tool_use");
      return {
        id: `msg_${i}`,
        type: "message",
        role: "assistant",
        model: p.model,
        content: t.content,
        stop_reason: (t.stop ?? (hasTool ? "tool_use" : "end_turn")) as Anthropic.StopReason,
        stop_sequence: null,
        usage: { input_tokens: 1000, output_tokens: 200 },
      } as unknown as Anthropic.Message;
    },
  };
  return { model, seen };
}

const getOrder = defineTool({
  name: "get_order",
  description: "Look up an order",
  schema: z.object({ order_id: z.string().regex(/^O\d+$/) }),
  effect: "read",
  run: ({ order_id }) => ({ order_id, status: "delivered", total: 42 }),
});

const refund = defineTool({
  name: "refund",
  description: "Refund an order",
  schema: z.object({ order_id: z.string(), amount: z.number().positive() }),
  effect: "write",
  run: ({ order_id, amount }) => `refunded ${amount} on ${order_id}`,
});

describe("runAgent", () => {
  it("runs a tool loop to completion and returns tool results to the model", async () => {
    const { model, seen } = scripted([
      { content: [toolUse("get_order", { order_id: "O1" })] },
      { content: [text("Your order O1 was delivered.")] },
    ]);
    const r = await runAgent({ model, tools: [getOrder], input: "where is O1?" });
    expect(r.stop).toBe("completed");
    expect(r.text).toContain("delivered");
    expect(r.steps).toBe(2);
    const last = seen[1]!.messages.at(-1)!;
    expect(JSON.stringify(last.content)).toContain('"status\\":\\"delivered');
    expect(r.costUsd).toBeCloseTo((2000 * 4 + 400 * 20) / 1e6);
  });

  it("rejects schema-invalid input without running the tool", async () => {
    let ran = false;
    const t = defineTool({ ...getOrder, run: () => ((ran = true), "x") });
    const { model } = scripted([
      { content: [toolUse("get_order", { order_id: "bad" })] },
      { content: [text("sorry")] },
    ]);
    const r = await runAgent({ model, tools: [t], input: "q" });
    expect(ran).toBe(false);
    expect(r.interventions[0]?.kind).toBe("invalid_input");
  });

  it("asks for approval before write tools and reports a decline", async () => {
    const calls: unknown[] = [];
    const { model, seen } = scripted([
      { content: [toolUse("refund", { order_id: "O1", amount: 42 })] },
      { content: [text("I could not refund that.")] },
    ]);
    const r = await runAgent({
      model,
      tools: [getOrder, refund],
      input: "refund O1",
      approve: (c) => (calls.push(c), false),
    });
    expect(calls).toEqual([{ name: "refund", input: { order_id: "O1", amount: 42 } }]);
    expect(r.interventions.map((i) => i.kind)).toEqual(["declined"]);
    expect(JSON.stringify(seen[1]!.messages.at(-1))).toContain("is_error");
  });

  it("warns on repeated identical calls, then stops the loop", async () => {
    const { model } = scripted([{ content: [toolUse("get_order", { order_id: "O1" })] }]);
    const r = await runAgent({ model, tools: [getOrder], input: "q", limits: { maxRepeats: 2 } });
    expect(r.stop).toBe("loop_detected");
    expect(r.interventions.map((i) => i.kind)).toEqual(["loop", "loop"]);
    expect(r.toolCalls).toBe(4);
  });

  it("retries idempotent tools on transient errors but not write tools", async () => {
    let n = 0;
    const flaky = defineTool({
      ...getOrder,
      run: () => {
        if (++n < 3) throw new TransientToolError("503");
        return "ok";
      },
    });
    let w = 0;
    const flakyWrite = defineTool({
      ...refund,
      run: () => {
        w++;
        throw new TransientToolError("503");
      },
    });
    const { model } = scripted([
      { content: [toolUse("get_order", { order_id: "O1" }), toolUse("refund", { order_id: "O1", amount: 1 })] },
      { content: [text("done")] },
    ]);
    const r = await runAgent({ model, tools: [flaky, flakyWrite], input: "q", sleep: async () => {} });
    expect(n).toBe(3);
    expect(w).toBe(1);
    expect(r.interventions.map((i) => i.kind)).toEqual(["tool_error"]);
  });

  it("times out hanging tools", async () => {
    const hang = defineTool({ ...getOrder, idempotent: false, run: () => new Promise(() => {}) });
    const { model } = scripted([
      { content: [toolUse("get_order", { order_id: "O1" })] },
      { content: [text("done")] },
    ]);
    const r = await runAgent({ model, tools: [hang], input: "q", limits: { toolTimeoutMs: 20 } });
    expect(r.interventions[0]?.detail).toMatch(/timed out/);
  });

  it("enforces step and cost budgets", async () => {
    const { model } = scripted([
      { content: [toolUse("get_order", { order_id: "O1" })] },
      { content: [toolUse("get_order", { order_id: "O2" })] },
      { content: [toolUse("get_order", { order_id: "O3" })] },
    ]);
    const steps = await runAgent({ model, tools: [getOrder], input: "q", limits: { maxSteps: 2 } });
    expect(steps.stop).toBe("max_steps");
    const { model: m2 } = scripted([{ content: [toolUse("get_order", { order_id: "O1" })] }]);
    const cost = await runAgent({ model: m2, tools: [getOrder], input: "q", limits: { maxCostUsd: 0.001 } });
    expect(cost.stop).toBe("budget");
  });

  it("passes pause_turn back and stops on refusal", async () => {
    const { model, seen } = scripted([
      { content: [text("thinking...")], stop: "pause_turn" },
      { content: [text("no")], stop: "refusal" },
    ]);
    const r = await runAgent({ model, tools: [], input: "q" });
    expect(seen).toHaveLength(2);
    expect(r.stop).toBe("refusal");
  });

  it("records a nested trace", async () => {
    const { model } = scripted([
      { content: [toolUse("get_order", { order_id: "O1" })] },
      { content: [text("ok")] },
    ]);
    const r = await runAgent({ model, tools: [getOrder], input: "q" });
    const names = r.trace.spans.map((s) => s.name);
    expect(names).toEqual(["agent", "model", "tool", "model"]);
    expect(r.trace.spans[2]!.parent).toBe(r.trace.spans[0]!.id);
    expect(r.trace.summary()["tool:get_order"]?.count).toBe(1);
  });
});

describe("tools", () => {
  it("generates a JSON schema from zod", () => {
    const t = toApiTool(getOrder);
    expect(t.input_schema.type).toBe("object");
    expect(t.input_schema.required).toEqual(["order_id"]);
  });
  it("stableStringify ignores key order", () => {
    expect(stableStringify({ b: 1, a: [{ d: 2, c: 3 }] })).toBe(stableStringify({ a: [{ c: 3, d: 2 }], b: 1 }));
  });
});
