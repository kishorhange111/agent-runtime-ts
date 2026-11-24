import type { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { Ajv } from "ajv";
import { z } from "zod";
import type { ToolSpec } from "./tools.js";

export interface McpToolOptions {
  /** Prefix tool names (``crm__get_customer``) so several servers can't collide. */
  prefix?: string;
  /** Only expose these server tool names (least privilege). */
  allow?: string[];
  /**
   * Trust the server's annotations. MCP annotations are *hints* from the server;
   * by default a tool is treated as a read only if it says ``readOnlyHint: true``
   * AND ``trustAnnotations`` is set. Everything else goes through approval.
   */
  trustAnnotations?: boolean;
}

const ajv = new Ajv({ allErrors: true, strict: false });

/**
 * Wrap every tool of a connected MCP server as a runtime ``ToolSpec``: the server's
 * JSON schema is sent to the model as-is and enforced locally with ajv before the
 * call; side-effect class comes from annotations (conservatively by default).
 */
export async function mcpTools(client: Client, opts: McpToolOptions = {}): Promise<ToolSpec[]> {
  const { tools } = await client.listTools();
  return tools
    .filter((t) => !opts.allow || opts.allow.includes(t.name))
    .map((t) => {
      const jsonSchema = { ...(t.inputSchema as Record<string, unknown>) };
      delete jsonSchema.$schema;
      const validate = ajv.compile(jsonSchema);
      const schema = z.unknown().superRefine((v, ctx) => {
        if (!validate(v)) {
          for (const e of validate.errors ?? []) {
            ctx.addIssue({
              code: "custom",
              message: `${e.instancePath || "(root)"} ${e.message ?? "invalid"}`,
            });
          }
        }
      });
      const ann = t.annotations ?? {};
      const readOnly = Boolean(opts.trustAnnotations && ann.readOnlyHint === true);
      const spec: ToolSpec = {
        name: opts.prefix ? `${opts.prefix}__${t.name}` : t.name,
        description: t.description ?? t.title ?? t.name,
        schema,
        jsonSchema,
        effect: readOnly ? "read" : "write",
        idempotent: readOnly || Boolean(opts.trustAnnotations && ann.idempotentHint === true),
        run: async (input, ctx) => {
          const res = await client.callTool(
            { name: t.name, arguments: input as Record<string, unknown> },
            undefined,
            { signal: ctx.signal },
          );
          const content = (res.content ?? []) as { type: string; text?: string }[];
          const text = content
            .filter((c) => c.type === "text")
            .map((c) => c.text ?? "")
            .join("\n");
          if (res.isError) throw new Error(text || "MCP tool reported an error");
          return text || JSON.stringify(res.structuredContent ?? content);
        },
      };
      return spec;
    });
}
