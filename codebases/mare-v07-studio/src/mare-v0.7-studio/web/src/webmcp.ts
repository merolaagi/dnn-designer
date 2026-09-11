import { runs } from "./api";
interface ModelContext {
  registerTool(
    tool: {
      name: string;
      description: string;
      inputSchema: object;
      annotations: object;
      execute(input: unknown): unknown;
    },
    options: { signal: AbortSignal },
  ): void | Promise<void>;
}
export function registerResearchTools() {
  const context = (document as Document & { modelContext?: ModelContext })
    .modelContext;
  if (!context) return () => {};
  const lifecycle = new AbortController();
  try {
    void Promise.resolve(
      context.registerTool(
        {
          name: "list_research_runs",
          description:
            "Read the latest 50 research run statuses in the current authenticated workspace.",
          inputSchema: {
            type: "object",
            properties: {},
            additionalProperties: false,
          },
          annotations: { readOnlyHint: true, untrustedContentHint: true },
          async execute(input) {
            if (
              !input ||
              typeof input !== "object" ||
              Array.isArray(input) ||
              Object.keys(input).length
            )
              throw new Error("Expected an empty object");
            return (await runs()).map(
              ({ id, title, status, current_round, target_rounds }) => ({
                id,
                title,
                status,
                current_round,
                target_rounds,
              }),
            );
          },
        },
        { signal: lifecycle.signal },
      ),
    ).catch(() => {});
  } catch {
    /* Optional browser capability. The normal UI remains available. */
  }
  return () => lifecycle.abort();
}
