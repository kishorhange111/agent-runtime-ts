import { z } from "zod";
import type Anthropic from "@anthropic-ai/sdk";

/**
 * A typed tool. The zod schema is the single source of truth: it generates the
 * JSON schema sent to the model AND validates the model's input before `run`.
 */
export interface ToolSpec<S extends z.ZodType = z.ZodType> {
  name: string;
  description: string;
  schema: S;
  /** Raw JSON schema sent to the model instead of one generated from ``schema``
   * (used for tools whose schema comes from elsewhere, e.g. an MCP server). */
  jsonSchema?: Record<string, unknown>;
  /** "write" tools mutate external state: they go through the approval hook. */
  effect: "read" | "write";
  /** Safe to retry on transient failure (reads usually are; payments are not). */
  idempotent?: boolean;
  run: (input: z.infer<S>, ctx: ToolContext) => Promise<unknown> | unknown;
}

export interface ToolContext {
  signal: AbortSignal;
  attempt: number;
}

export function defineTool<S extends z.ZodType>(spec: ToolSpec<S>): ToolSpec<S> {
  return { idempotent: spec.effect === "read", ...spec };
}

export function toApiTool(t: ToolSpec): Anthropic.Tool {
  const schema = { ...(t.jsonSchema ?? (z.toJSONSchema(t.schema) as Record<string, unknown>)) };
  delete schema.$schema;
  return {
    name: t.name,
    description: t.description,
    input_schema: schema as Anthropic.Tool.InputSchema,
  };
}

/** Error a tool throws to mark a failure as transient (retryable if idempotent). */
export class TransientToolError extends Error {
  override name = "TransientToolError";
}

/** Canonical JSON (sorted keys) for detecting repeated identical calls. */
export function stableStringify(v: unknown): string {
  if (Array.isArray(v)) return `[${v.map(stableStringify).join(",")}]`;
  if (v && typeof v === "object") {
    const o = v as Record<string, unknown>;
    return `{${Object.keys(o)
      .sort()
      .map((k) => `${JSON.stringify(k)}:${stableStringify(o[k])}`)
      .join(",")}}`;
  }
  return JSON.stringify(v);
}
