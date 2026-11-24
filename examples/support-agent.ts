/**
 * Customer-support agent on the runtime: read tools run freely, the refund tool
 * needs human approval, and the whole run is budgeted and traced.
 *
 *   npm run build && node dist/examples/support-agent.js "Refund my order O1001"
 */
import Anthropic from "@anthropic-ai/sdk";
import { createInterface } from "node:readline/promises";
import { z } from "zod";
import { anthropicClient, defineTool, runAgent, Tracer } from "../src/index.js";

const ORDERS: Record<string, { customer: string; total: number; status: string }> = {
  O1001: { customer: "C1", total: 59.0, status: "delivered" },
  O1002: { customer: "C2", total: 120.5, status: "in_transit" },
};

const getOrder = defineTool({
  name: "get_order",
  description: "Fetch an order by id (e.g. O1001). Returns customer, total and status.",
  schema: z.object({ order_id: z.string().regex(/^O\d{4}$/, "format O#### expected") }),
  effect: "read",
  run: ({ order_id }) => ORDERS[order_id] ?? { error: "not_found" },
});

const issueRefund = defineTool({
  name: "issue_refund",
  description: "Refund a delivered order, up to its total. Requires human approval.",
  schema: z.object({ order_id: z.string(), amount: z.number().positive(), reason: z.string() }),
  effect: "write",
  run: ({ order_id, amount }) => {
    const o = ORDERS[order_id];
    if (!o) return { error: "not_found" };
    if (o.status !== "delivered") return { error: "only delivered orders can be refunded" };
    if (amount > o.total) return { error: `amount exceeds order total ${o.total}` };
    return { refund_id: `R-${order_id}`, amount };
  },
});

const rl = createInterface({ input: process.stdin, output: process.stdout });
const tracer = new Tracer();
const result = await runAgent({
  model: anthropicClient(new Anthropic()),
  system: "You are a support agent. Verify orders before acting. Never refund more than the total.",
  tools: [getOrder, issueRefund],
  input: process.argv[2] ?? "Please refund order O1001, it arrived damaged.",
  limits: { maxSteps: 8, maxCostUsd: 0.25 },
  approve: async ({ name, input }) =>
    (await rl.question(`Approve ${name} ${JSON.stringify(input)}? [y/N] `)).trim() === "y",
  tracer,
});
rl.close();

console.log(`\n${result.text}\n`);
console.log({ stop: result.stop, steps: result.steps, costUsd: result.costUsd.toFixed(4) });
console.table(tracer.summary());
tracer.export("trace.jsonl");
