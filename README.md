# agent-runtime-ts

A production-minded **agent runtime in TypeScript** on the Claude API. It is the layer between "call the model in a loop" and something you can deploy for a customer.

| Concern | What the runtime does |
|---|---|
| **Typed tools** | One zod schema per tool generates the JSON schema sent to the model **and** validates the model's input before `run`. Invalid input goes back as an `is_error` tool result, and the tool never runs. |
| **Side effects** | Tools declare `effect: "read" \| "write"`. Every write goes through an `approve` hook (human-in-the-loop). A decline is returned to the model as "do not retry". |
| **Retries** | Full-jitter retries on `TransientToolError`, **only for idempotent tools**. Reads default to idempotent; a refund is never retried blindly. |
| **Timeouts** | A per-call deadline with `AbortSignal` propagation, so hanging tools can't stall the agent. |
| **Loop guard** | Identical `(tool, args)` calls are counted using canonical JSON. After `maxRepeats` the model gets a corrective error; one more repeat ends the run with `loop_detected`. |
| **Budgets** | Caps on model steps, tool calls and USD cost, using Opus 5.5 pricing: $4 / $20 per MTok input/output, with cache pricing not modelled. |
| **Stop reasons** | `end_turn`, `tool_use`, `pause_turn` (resent so the server continues), `refusal` and `max_tokens` are all handled explicitly. |
| **Tracing** | Nested spans (agent → model → tool → approval) with token counts, attempts and errors. Exported as JSONL in an OpenTelemetry-compatible shape, with a per-tool latency/error summary. |
| **Evals** | Every guard firing is recorded in `result.interventions`, which you can assert on in regression tests. |

Assistant turns are appended **unchanged**, so thinking blocks round-trip as the API requires.

## Usage

```ts
import Anthropic from "@anthropic-ai/sdk";
import { z } from "zod";
import { anthropicClient, defineTool, runAgent } from "agent-runtime-ts";

const getOrder = defineTool({
  name: "get_order",
  description: "Fetch an order by id",
  schema: z.object({ order_id: z.string().regex(/^O\d{4}$/) }),
  effect: "read",
  run: ({ order_id }) => db.orders.get(order_id),
});

const result = await runAgent({
  model: anthropicClient(new Anthropic()),
  tools: [getOrder, issueRefund],
  input: "Refund my order O1001",
  limits: { maxSteps: 8, maxCostUsd: 0.25 },
  approve: async (call) => askHuman(call),
});
// result.stop, result.text, result.costUsd, result.interventions, result.trace.summary()
```

See [`examples/support-agent.ts`](examples/support-agent.ts) for a runnable support agent with terminal approval. It needs `ANTHROPIC_API_KEY`.

## Testing without an API key

`ModelClient` is a one-method interface, so the test suite drives the runtime with a **scripted model** that replays assistant turns. The suite (`tests/runtime.test.ts`) covers:
- the tool loop
- schema rejection
- approval declines
- the loop guard
- retries (idempotent only)
- timeouts
- step and cost budgets
- `pause_turn` and refusal
- the trace tree

```
npm ci
npm run typecheck
npm test
```
