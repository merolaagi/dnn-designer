import { describe, it, expect, vi, afterEach } from "vitest";
import { api, ApiError, parseFrame, setToken } from "./api";

afterEach(() => {
  vi.unstubAllGlobals();
  setToken("");
});
describe("authenticated API and event contracts", () => {
  it("parses SSE cursors and ignores heartbeat frames", () => {
    expect(parseFrame(": heartbeat")).toEqual({
      id: undefined,
      event: undefined,
      data: undefined,
    });
    expect(
      parseFrame('id: 42\nevent: progress\ndata: {"id":42}'),
    ).toMatchObject({ id: 42, event: "progress", data: { id: 42 } });
  });
  it("sends a bearer header without putting credentials in URLs", async () => {
    const fetcher = vi.fn().mockResolvedValue(new Response("[]"));
    vi.stubGlobal("fetch", fetcher);
    setToken("workspace-secret");
    await api("/runs");
    expect(fetcher.mock.calls[0][0]).toBe("/api/runs");
    expect(fetcher.mock.calls[0][1].headers.Authorization).toBe(
      "Bearer workspace-secret",
    );
  });
  it("retains server error status and request ID", async () => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValue(
          new Response(
            JSON.stringify({ error: { message: "Denied", request_id: "r1" } }),
            { status: 401 },
          ),
        ),
    );
    await expect(api("/runs")).rejects.toMatchObject({
      message: "Denied",
      status: 401,
      requestId: "r1",
    });
    expect(new ApiError("x", 500)).toBeInstanceOf(Error);
  });
});
