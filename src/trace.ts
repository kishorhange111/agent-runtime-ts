import { writeFileSync } from "node:fs";

export interface Span {
  id: number;
  parent: number | null;
  name: string;
  startMs: number;
  endMs: number;
  attrs: Record<string, unknown>;
  status: "ok" | "error";
}

/**
 * Minimal structured tracer: nested spans with attributes, exported as JSONL.
 * Shaped like OpenTelemetry spans so it maps 1:1 onto an OTel exporter.
 */
export class Tracer {
  readonly spans: Span[] = [];
  private stack: number[] = [];
  private next = 1;

  constructor(private readonly now: () => number = () => performance.now()) {}

  async span<T>(
    name: string,
    attrs: Record<string, unknown>,
    fn: (s: Span) => Promise<T>,
  ): Promise<T> {
    const s: Span = {
      id: this.next++,
      parent: this.stack.at(-1) ?? null,
      name,
      startMs: this.now(),
      endMs: Number.NaN,
      attrs: { ...attrs },
      status: "ok",
    };
    this.spans.push(s);
    this.stack.push(s.id);
    try {
      return await fn(s);
    } catch (e) {
      s.status = "error";
      s.attrs.error = e instanceof Error ? e.message : String(e);
      throw e;
    } finally {
      s.endMs = this.now();
      this.stack.pop();
    }
  }

  toJSONL(): string {
    return this.spans.map((s) => JSON.stringify(s)).join("\n");
  }

  export(path: string): void {
    writeFileSync(path, `${this.toJSONL()}\n`);
  }

  /** Per-span-name count, error count and total/max latency. */
  summary(): Record<string, { count: number; errors: number; totalMs: number; maxMs: number }> {
    const out: Record<string, { count: number; errors: number; totalMs: number; maxMs: number }> =
      {};
    for (const s of this.spans) {
      const key = s.name === "tool" ? `tool:${String(s.attrs.tool)}` : s.name;
      const d = (out[key] ??= { count: 0, errors: 0, totalMs: 0, maxMs: 0 });
      const ms = s.endMs - s.startMs;
      d.count++;
      d.errors += s.status === "error" ? 1 : 0;
      d.totalMs += ms;
      d.maxMs = Math.max(d.maxMs, ms);
    }
    return out;
  }
}
