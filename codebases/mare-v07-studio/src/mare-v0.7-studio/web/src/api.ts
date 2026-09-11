import type { components } from "./schema";
export type Run = components["schemas"]["RunSummary"];
export type Detail = components["schemas"]["RunDetail"];
export type State = Detail["snapshot"];
export type Create = components["schemas"]["CreateRun"];
export type Event = components["schemas"]["EventOut"];
let token = "";
export function setToken(value: string) {
  token = value;
}
export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public requestId?: string,
  ) {
    super(message);
  }
}
export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(`/api${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...init.headers,
    },
  });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new ApiError(
      body?.error?.message || `Request failed (${response.status})`,
      response.status,
      body?.error?.request_id || response.headers.get("x-request-id"),
    );
  }
  return response.json();
}
export const runs = () => api<Run[]>("/runs");
export const detail = (id: string) =>
  api<Detail>(`/runs/${encodeURIComponent(id)}`);
export const create = (body: Create) =>
  api<Run>("/runs", { method: "POST", body: JSON.stringify(body) });
export const action = (id: string, command: "cancel" | "resume") =>
  api<Run>(`/runs/${encodeURIComponent(id)}/${command}`, { method: "POST" });
export function parseFrame(frame: string): {
  id?: number;
  event?: string;
  data?: Event;
} {
  const lines = frame.split("\n");
  const id = lines.find((l) => l.startsWith("id: "))?.slice(4);
  const event = lines.find((l) => l.startsWith("event: "))?.slice(7);
  const data = lines
    .filter((l) => l.startsWith("data: "))
    .map((l) => l.slice(6))
    .join("\n");
  return {
    id: id ? Number(id) : undefined,
    event,
    data: data ? JSON.parse(data) : undefined,
  };
}
export async function stream(
  id: string,
  after: number,
  signal: AbortSignal,
  onEvent: (e: Event) => void,
): Promise<boolean> {
  const response = await fetch(
    `/api/runs/${encodeURIComponent(id)}/stream?after=${after}`,
    { signal, headers: token ? { Authorization: `Bearer ${token}` } : {} },
  );
  if (!response.ok || !response.body)
    throw new ApiError("Live connection unavailable", response.status);
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (!signal.aborted) {
      const chunk = await reader.read();
      if (chunk.done) break;
      buffer += decoder.decode(chunk.value, { stream: true });
      let end: number;
      while ((end = buffer.indexOf("\n\n")) !== -1) {
        const frame = parseFrame(buffer.slice(0, end));
        buffer = buffer.slice(end + 2);
        if (frame.event === "settled") return true;
        if (frame.event === "progress" && frame.data) onEvent(frame.data);
      }
    }
    return false;
  } finally {
    await reader.cancel();
    reader.releaseLock();
  }
}
