import type Anthropic from "@anthropic-ai/sdk";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { describe, expect, it } from "vitest";
import { z } from "zod";
import { type ModelClient, mcpTools, runAgent, toApiTool } from "../src/index.js";

async function crmServer() {
  const server = new McpServer({ name: "crm", version: "1.0.0" });
  const tickets: string[] = [];
  server.registerTool(
    "get_customer",
    {
      description: "Look up a customer by id",
      inputSchema: { customer_id: z.string().regex(/^C\d+$/) },
      annotations: { readOnlyHint: true },
    },
    async ({ customer_id }) => ({
      content: [{ type: "text", text: JSON.stringify({ customer_id, tier: "gold" }) }],
    }),
  );
  server.registerTool(
    "open_ticket",
    { description: "Open a support ticket", inputSchema: { subject: z.string().min(3) } },
    async ({ subject }) => {
      tickets.push(subject);
      return { content: [{ type: "text", text: `ticket #${tickets.length}` }] };
    },
  );
  server.registerTool("broken", { description: "Always fails" }, async () => ({
    isError: true,
    content: [{ type: "text", text: "backend down" }],
  }));
  const [a, b] = InMemoryTransport.createLinkedPair();
  await server.connect(a);
  const client = new Client({ name: "runtime", version: "0.1.0" });
  await client.connect(b);
  return { client, tickets };
}

function scripted(turns: Anthropic.ContentBlock[][]): ModelClient {
  let i = 0;
  return {
    async create(p) {
      const content = turns[Math.min(i++, turns.length - 1)]!;
      return {
        id: `m${i}`,
        type: "message",
        role: "assistant",
        model: p.model,
        content,
        stop_reason: content.some((c) => c.type === "tool_use") ? "tool_use" : "end_turn",
        stop_sequence: null,
        usage: { input_tokens: 10, output_tokens: 10 },
      } as unknown as Anthropic.Message;
    },
  };
}
const use = (id: string, name: string, input: unknown) =>
  ({ type: "tool_use", id, name, input }) as Anthropic.ToolUseBlock;
const text = (t: string) => ({ type: "text", text: t }) as Anthropic.TextBlock;

describe("mcpTools", () => {
  it("exposes server tools with their JSON schema and conservative effects", async () => {
    const { client } = await crmServer();
    const untrusted = await mcpTools(client, { prefix: "crm" });
    expect(untrusted.map((t) => t.name).sort()).toEqual(["crm__broken", "crm__get_customer", "crm__open_ticket"]);
    expect(untrusted.every((t) => t.effect === "write")).toBe(true); // hints not trusted by default
    const trusted = await mcpTools(client, { trustAnnotations: true, allow: ["get_customer", "open_ticket"] });
    expect(trusted.find((t) => t.name === "get_customer")?.effect).toBe("read");
    expect(trusted.find((t) => t.name === "open_ticket")?.effect).toBe("write");
    const api = toApiTool(trusted[0]!);
    expect(api.input_schema.required).toEqual(["customer_id"]);
  });

  it("runs MCP tools through the runtime: validation, approval and error mapping", async () => {
    const { client, tickets } = await crmServer();
    const tools = await mcpTools(client, { trustAnnotations: true });
    const approvals: string[] = [];
    const r = await runAgent({
      model: scripted([
        [use("1", "get_customer", { customer_id: "bad" })], // fails ajv (pattern)
        [use("2", "get_customer", { customer_id: "C7" }), use("3", "open_ticket", { subject: "Refund" })],
        [use("4", "broken", {})],
        [text("Opened a ticket for gold customer C7.")],
      ]),
      tools,
      input: "help C7",
      approve: (c) => (approvals.push(c.name), true),
    });
    expect(r.stop).toBe("completed");
    expect(tickets).toEqual(["Refund"]);
    expect(approvals).toEqual(["open_ticket", "broken"]); // reads skip approval
    expect(r.interventions.map((i) => i.kind)).toEqual(["invalid_input", "tool_error"]);
    expect(r.interventions[1]?.detail).toContain("backend down");
  });
});
