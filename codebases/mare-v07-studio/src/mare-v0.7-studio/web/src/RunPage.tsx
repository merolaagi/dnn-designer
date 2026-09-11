import { useCallback, useEffect, useRef, useState } from "react";
import { Link, NavLink, Route, Routes, useParams } from "react-router-dom";
import {
  GitBranch,
  Network,
  Pause,
  Play,
  Radio,
  RefreshCw,
  Download,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  action,
  api,
  detail,
  stream,
  type Detail,
  type Event,
  type State,
} from "./api";
import { useResource } from "./hooks";
import { Badge, Empty, ErrorBox, Loading, Metric } from "./ui";

const tabs = [
  ["", "Overview"],
  ["branches", "Branches"],
  ["claims", "Claims & evidence"],
  ["failures", "Failures"],
  ["questions", "Questions"],
  ["graph", "Proof DAG"],
  ["policy", "Neural policy"],
  ["budget", "Budgets & costs"],
  ["events", "Logs & events"],
];
function useEvents(id: string, epoch: number) {
  const [events, setEvents] = useState<Event[]>([]);
  const [connection, setConnection] = useState("Connecting");
  const cursor = useRef(0);
  useEffect(() => {
    setEvents([]);
    cursor.current = 0;
  }, [id]);
  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    async function connect() {
      if (controller.signal.aborted) return;
      setConnection("Connecting");
      try {
        const settled = await stream(
          id,
          cursor.current,
          controller.signal,
          (e) => {
            if (e.id <= cursor.current) return;
            cursor.current = e.id;
            setEvents((old) => [...old, e].slice(-1000));
            setConnection("Live");
          },
        );
        if (!controller.signal.aborted) {
          setConnection(settled ? "Up to date" : "Reconnecting");
          if (!settled) timer = setTimeout(connect, 1500);
        }
      } catch {
        if (!controller.signal.aborted) {
          setConnection("Polling fallback");
          timer = setTimeout(connect, 5000);
        }
      }
    }
    void connect();
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [id, epoch]);
  return { events, connection };
}
export default function RunPage() {
  const { id = "" } = useParams();
  const loader = useCallback(() => detail(id), [id]);
  const { data: run, error, loading, reload } = useResource(loader);
  const [epoch, setEpoch] = useState(0);
  const { events, connection } = useEvents(id, epoch);
  const [actionError, setActionError] = useState<Error>();
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    const timer = setInterval(reload, 4000);
    return () => clearInterval(timer);
  }, [reload]);
  const latest = events.at(-1)?.id;
  useEffect(() => {
    const timer = setTimeout(reload, 200);
    return () => clearTimeout(timer);
  }, [latest, reload]);
  async function command(value: "cancel" | "resume") {
    setBusy(true);
    setActionError(undefined);
    try {
      await action(id, value);
      reload();
      setEpoch((n) => n + 1);
    } catch (e) {
      setActionError(e as Error);
    } finally {
      setBusy(false);
    }
  }
  function download() {
    if (!run) return;
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(run, null, 2)], { type: "application/json" }),
    );
    const a = document.createElement("a");
    a.href = url;
    a.download = `${run.id}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }
  if (error && !run) return <ErrorBox error={error} retry={reload} />;
  if (!run) return loading ? <Loading /> : <Empty>Run not found.</Empty>;
  const state = run.snapshot;
  return (
    <>
      <div className="eyebrow">
        <Link to="/">WORKSPACE</Link> / INVESTIGATION{" "}
        <span className="mono">{id}</span>
      </div>
      <div className="page-heading">
        <div>
          <h1>{run.title}</h1>
          <div className="run-meta">
            <Badge value={run.status} />
            <Badge value={run.provider} />
            <span className="muted">
              <Radio size={13} />
              {connection}
            </span>
            <span className="muted">
              Round {run.current_round} of {run.target_rounds}
            </span>
          </div>
        </div>
        <div className="actions">
          <button onClick={download} title="Export checkpoint">
            <Download size={16} />
            Export
          </button>
          {["queued", "running"].includes(run.status) && (
            <button
              className="danger"
              disabled={busy}
              onClick={() => void command("cancel")}
            >
              <Pause size={15} />
              Cancel run
            </button>
          )}
          {["failed", "cancelled"].includes(run.status) && (
            <button
              className="primary"
              disabled={busy}
              onClick={() => void command("resume")}
            >
              <Play size={15} />
              Resume
            </button>
          )}
          <button aria-label="Refresh run" onClick={reload}>
            <RefreshCw size={16} />
          </button>
        </div>
      </div>
      {actionError && <ErrorBox error={actionError} />}{" "}
      {error && <ErrorBox error={error} retry={reload} />}{" "}
      {run.error && (
        <div className="notice error">
          Last execution error: {run.error}. Check worker configuration;
          resuming replays the interrupted round.
        </div>
      )}
      {run.status === "cancelling" && (
        <div className="notice" role="status">
          Cancellation requested. Waiting for the worker to stop the active
          round; the last completed checkpoint is retained.
        </div>
      )}
      <nav className="tabs" aria-label="Run sections">
        {tabs.map(([path, label]) => (
          <NavLink
            key={path}
            end={path === ""}
            to={`/runs/${id}${path ? "/" + path : ""}`}
          >
            {label}
          </NavLink>
        ))}
      </nav>
      <Routes>
        <Route index element={<Overview run={run} events={events} />} />
        <Route path="branches" element={<Branches state={state} />} />
        <Route path="claims" element={<Claims state={state} />} />
        <Route path="failures" element={<Failures state={state} />} />
        <Route path="questions" element={<Questions state={state} />} />
        <Route path="graph" element={<Graph state={state} />} />
        <Route path="policy" element={<Policy state={state} />} />
        <Route
          path="budget"
          element={<Budget state={state} provider={run.provider} />}
        />
        <Route path="events" element={<Events id={id} events={events} />} />
        <Route path="*" element={<Empty>Unknown run section.</Empty>} />
      </Routes>
    </>
  );
}
function Overview({ run, events }: { run: Detail; events: Event[] }) {
  const s = run.snapshot;
  const history = events
    .filter((e) => e.kind === "checkpoint")
    .map((e) => ({
      round: e.payload.round,
      claims: e.payload.claims,
      verified: e.payload.verified,
    }));
  return (
    <>
      <div className="metrics">
        <Metric
          label="Research branches"
          value={s.branches.length}
          caption={`${s.branches.filter((b) => b.status === "terminated").length} terminated`}
        />
        <Metric
          label="Claims verified"
          value={
            s.claims.filter(
              (c) => c.status === "verified" || c.status === "formalized",
            ).length
          }
          caption={`${s.claims.length} total claims`}
        />
        <Metric
          label="Evidence records"
          value={s.evidence.length}
          caption={`${s.failures.length} reusable failures`}
        />
        <Metric
          label="Open questions"
          value={s.questions.filter((q) => q.status === "open").length}
        />
      </div>
      <div className="two-col">
        <section className="panel compact">
          <div className="section-heading plain">
            <h2>Research target</h2>
            <GitBranch size={19} />
          </div>
          <p className="statement">{s.problem.canonical_statement}</p>
          <h3>Assumptions</h3>
          <ul className="text-list">
            {s.problem.assumptions.map((a, i) => (
              <li key={i}>{a}</li>
            ))}
          </ul>
          <h3>Success conditions</h3>
          <ul className="text-list">
            {s.problem.success_conditions.map((a, i) => (
              <li key={i}>{a}</li>
            ))}
          </ul>
        </section>
        <section className="panel compact">
          <h2>Research progress</h2>
          <p className="muted">Claims at completed round checkpoints</p>
          {history.length ? (
            <div className="chart">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={history}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="round" />
                  <YAxis allowDecimals={false} />
                  <Tooltip />
                  <Line
                    name="All claims"
                    type="monotone"
                    dataKey="claims"
                    stroke="#8fa1ae"
                    strokeWidth={2}
                  />
                  <Line
                    name="Verified"
                    type="monotone"
                    dataKey="verified"
                    stroke="#277b60"
                    strokeWidth={2}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <Empty>Progress appears when the first round finishes.</Empty>
          )}
          <div className="legend">
            <span>
              <i style={{ background: "#8fa1ae" }} />
              All claims
            </span>
            <span>
              <i style={{ background: "#277b60" }} />
              Verified
            </span>
          </div>
        </section>
      </div>
      <section className="panel compact">
        <h2>Research boundaries</h2>
        <p>
          Execution completion does not mean the problem is solved. Symbolic
          checks, assumption audits, and formal proof records remain distinct.
          This web service does not submit generated code or Lean source for
          execution.
        </p>
        {s.synthesis_notes.length > 0 && (
          <details>
            <summary>Synthesis notes ({s.synthesis_notes.length})</summary>
            {s.synthesis_notes.map((n, i) => (
              <p key={i}>{n}</p>
            ))}
          </details>
        )}
      </section>
    </>
  );
}
function Branches({ state }: { state: State }) {
  return !state.branches.length ? (
    <Empty>The director has not committed any branches yet.</Empty>
  ) : (
    <div className="branch-grid">
      {state.branches.map((b) => (
        <section key={b.id} className="panel compact">
          <div className="section-heading plain">
            <GitBranch size={19} />
            <Badge value={b.status} />
          </div>
          <h2>{b.title}</h2>
          <p>{b.research_question}</p>
          <p className="muted">{b.strategy}</p>
          <div className="score-row">
            <span>Research score</span>
            <strong>{b.score?.toFixed(3)}</strong>
          </div>
          <div className="progress">
            <span style={{ width: `${Math.min(1, b.score ?? 0) * 100}%` }} />
          </div>
          <dl>
            <div>
              <dt>Heuristic</dt>
              <dd>{b.heuristic_score?.toFixed(3)}</dd>
            </div>
            <div>
              <dt>Neural utility</dt>
              <dd>{b.policy_score?.toFixed(3) ?? "Cold start"}</dd>
            </div>
            <div>
              <dt>Neural blend</dt>
              <dd>{((b.policy_blend ?? 0) * 100).toFixed(0)}%</dd>
            </div>
            <div>
              <dt>Claims</dt>
              <dd>{state.claims.filter((c) => c.branch_id === b.id).length}</dd>
            </div>
          </dl>
          <small className="mono">{b.id}</small>
        </section>
      ))}
    </div>
  );
}
function ClaimCard({
  claim,
  state,
}: {
  claim: State["claims"][number];
  state: State;
}) {
  const evidence = state.evidence.filter((e) => e.claim_id === claim.id);
  const audits = state.assumption_audits.filter((a) => a.claim_id === claim.id);
  const objections = state.objections.filter((o) => o.claim_id === claim.id);
  return (
    <article className="panel compact claim-card">
      <div className="run-meta">
        <Badge value={claim.status ?? "proposed"} />
        <span className="muted">
          {claim.claim_type} · {claim.scope?.replaceAll("_", " ")}
        </span>
        <small className="mono">{claim.id}</small>
      </div>
      <h3>{claim.statement}</h3>
      <p>{claim.derivation}</p>
      <div className="muted">
        {state.branches.find((b) => b.id === claim.branch_id)?.title ??
          claim.branch_id}{" "}
        · Round {claim.round_no} · {claim.created_by}
      </div>
      <details>
        <summary>
          Evidence & review ({evidence.length} evidence, {objections.length}{" "}
          objections)
        </summary>
        {evidence.map((e) => (
          <div className="evidence" key={e.id}>
            <div className="run-meta">
              <Badge value={e.result} />
              <strong>{e.verifier}</strong>
              <span className="muted">
                {e.reproducible ? "Reproducible" : "Not reproducible"}
              </span>
            </div>
            <p>{e.details}</p>
          </div>
        ))}
        {objections.map((o) => (
          <div className="evidence" key={o.id}>
            <Badge value={o.severity} />
            <p>{o.description}</p>
            <small>
              {o.resolved ? "Resolved" : "Unresolved"} · {o.created_by}
            </small>
          </div>
        ))}
        {audits.map((a) => (
          <div className="evidence" key={a.id}>
            <Badge value={a.status} />
            <strong> Assumption audit</strong>
            <p>{a.notes}</p>
            {a.violations?.map((v, i) => (
              <p key={i}>{v}</p>
            ))}
          </div>
        ))}
        {claim.assumptions?.length > 0 && (
          <>
            <h3>Assumptions</h3>
            <ul>
              {claim.assumptions.map((a, i) => (
                <li key={i}>{a}</li>
              ))}
            </ul>
          </>
        )}
        {claim.dependencies?.length > 0 && (
          <p>Requires: {claim.dependencies.join(", ")}</p>
        )}
        {!evidence.length && !objections.length && !audits.length && (
          <p>No review records yet.</p>
        )}
      </details>
    </article>
  );
}
function Claims({ state }: { state: State }) {
  const [filter, setFilter] = useState("all");
  const [query, setQuery] = useState("");
  const filtered = state.claims.filter(
    (c) =>
      (filter === "all" || c.status === filter) &&
      c.statement.toLowerCase().includes(query.toLowerCase()),
  );
  return (
    <>
      <div className="filter-bar">
        <label className="sr-only" htmlFor="claim-search">
          Search claims
        </label>
        <input
          id="claim-search"
          placeholder="Search claim statements…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <label className="sr-only" htmlFor="claim-status">
          Claim status
        </label>
        <select
          id="claim-status"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
        >
          <option value="all">All claim statuses</option>
          {[...new Set(state.claims.map((c) => c.status))].map((status) => (
            <option key={status} value={status}>
              {status?.replaceAll("_", " ")}
            </option>
          ))}
        </select>
        <span className="muted">{filtered.length} claims</span>
      </div>
      {filtered.length ? (
        filtered.map((c) => <ClaimCard key={c.id} claim={c} state={state} />)
      ) : (
        <Empty>No matching claims.</Empty>
      )}
    </>
  );
}
function Failures({ state }: { state: State }) {
  return state.failures.length ? (
    <>
      {state.failures.map((f) => (
        <article key={f.id} className="panel compact">
          <div className="run-meta">
            <Badge value={f.severity ?? "major"} />
            <small className="mono">{f.id}</small>
          </div>
          <h2>{f.mechanism}</h2>
          <p>{f.failure_reason}</p>
          {f.reusable_constraint && (
            <div className="notice">
              <strong>Reusable constraint</strong>
              <p>{f.reusable_constraint}</p>
            </div>
          )}
          <small>
            Failed claim: {f.failed_claim_id ?? "Branch-level failure"} ·{" "}
            {state.branches.find((b) => b.id === f.branch_id)?.title ??
              f.branch_id}
          </small>
        </article>
      ))}
    </>
  ) : (
    <Empty>No failures recorded in the current checkpoint.</Empty>
  );
}
function Questions({ state }: { state: State }) {
  return state.questions.length ? (
    <>
      {state.questions.map((q) => (
        <article className="panel compact" key={q.id}>
          <div className="run-meta">
            <Badge value={q.status ?? "open"} />
            <small className="mono">{q.id}</small>
          </div>
          <h2>{q.question}</h2>
          <p>{q.rationale}</p>
          <div className="question-scores">
            <span>
              Information gain{" "}
              <strong>{q.expected_information_gain?.toFixed(2)}</strong>
            </span>
            <span>
              Impact <strong>{q.expected_impact?.toFixed(2)}</strong>
            </span>
            <span>
              Difficulty <strong>{q.difficulty?.toFixed(2)}</strong>
            </span>
            <span>
              Neural utility{" "}
              <strong>{q.policy_score?.toFixed(3) ?? "Unscored"}</strong>
            </span>
          </div>
          {!!q.answered_by_claim_ids?.length && (
            <small>Answered by {q.answered_by_claim_ids.join(", ")}</small>
          )}
        </article>
      ))}
    </>
  ) : (
    <Empty>No research questions committed yet.</Empty>
  );
}
export function layoutGraph(state: State) {
  const ids = new Set(state.claims.map((c) => c.id));
  const edges = state.dependency_edges.filter(
    (e) =>
      e.relation === "requires" &&
      ids.has(e.parent_claim_id) &&
      ids.has(e.child_claim_id),
  );
  const indegree = new Map(
    state.claims.map((c) => [
      c.id,
      edges.filter((e) => e.child_claim_id === c.id).length,
    ]),
  );
  const depth = new Map(state.claims.map((c) => [c.id, 0]));
  const queue = [...indegree].filter(([, n]) => !n).map(([id]) => id);
  let seen = 0;
  while (queue.length) {
    const id = queue.shift()!;
    seen++;
    for (const e of edges.filter((e) => e.parent_claim_id === id)) {
      depth.set(
        e.child_claim_id,
        Math.max(depth.get(e.child_claim_id)!, depth.get(id)! + 1),
      );
      indegree.set(e.child_claim_id, indegree.get(e.child_claim_id)! - 1);
      if (!indegree.get(e.child_claim_id)) queue.push(e.child_claim_id);
    }
  }
  const counts = new Map<number, number>();
  const positions = new Map(
    state.claims.map((c) => {
      const d = depth.get(c.id)!;
      const row = counts.get(d) ?? 0;
      counts.set(d, row + 1);
      return [c.id, { x: 30 + d * 320, y: 30 + row * 125 }];
    }),
  );
  return {
    edges,
    positions,
    cycle: seen !== state.claims.length,
    width: Math.max(900, 100 + (Math.max(0, ...depth.values()) + 1) * 320),
    height: Math.max(360, 70 + Math.max(0, ...counts.values()) * 125),
  };
}
function Graph({ state }: { state: State }) {
  const [selected, setSelected] = useState<string>();
  const [zoom, setZoom] = useState(1);
  const graph = layoutGraph(state);
  const claim = state.claims.find((c) => c.id === selected);
  if (!state.claims.length)
    return <Empty>The proof graph appears after claims are committed.</Empty>;
  return (
    <>
      <section className="panel">
        <div className="section-heading">
          <div>
            <h2>
              <Network size={18} /> Proof dependencies
            </h2>
            <small>
              Arrows connect prerequisites to dependent claims. Select a node to
              inspect evidence.
            </small>
          </div>
          <div className="actions">
            <button
              aria-label="Zoom out"
              onClick={() => setZoom((z) => Math.max(0.5, z - 0.2))}
            >
              −
            </button>
            <button onClick={() => setZoom(1)}>
              {Math.round(zoom * 100)}%
            </button>
            <button
              aria-label="Zoom in"
              onClick={() => setZoom((z) => Math.min(2, z + 0.2))}
            >
              +
            </button>
          </div>
        </div>
        {graph.cycle && (
          <div className="notice error">
            Cycle detected. This graph cannot be treated as a closed proof DAG.
          </div>
        )}
        <div className="graph-scroll">
          <svg
            role="img"
            aria-label="Interactive proof dependency graph"
            width={graph.width * zoom}
            height={graph.height * zoom}
            viewBox={`0 0 ${graph.width} ${graph.height}`}
          >
            <defs>
              <marker
                id="arrow"
                markerWidth="10"
                markerHeight="10"
                refX="8"
                refY="3"
                orient="auto"
              >
                <path d="M0,0 L0,6 L8,3 z" fill="#8a9f94" />
              </marker>
            </defs>
            {graph.edges.map((e) => {
              const a = graph.positions.get(e.parent_claim_id)!;
              const b = graph.positions.get(e.child_claim_id)!;
              return (
                <path
                  key={e.id}
                  d={`M${a.x + 250},${a.y + 42} C${a.x + 290},${a.y + 42} ${b.x - 40},${b.y + 42} ${b.x},${b.y + 42}`}
                  stroke="#8a9f94"
                  strokeWidth={1.5}
                  fill="none"
                  markerEnd="url(#arrow)"
                />
              );
            })}
            {state.claims.map((c) => {
              const p = graph.positions.get(c.id)!;
              return (
                <g
                  key={c.id}
                  role="button"
                  tabIndex={0}
                  aria-label={`${c.status}: ${c.statement}`}
                  onClick={() => setSelected(c.id)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      setSelected(c.id);
                    }
                  }}
                  className="graph-node"
                  transform={`translate(${p.x},${p.y})`}
                >
                  <rect
                    width={250}
                    height={86}
                    rx={6}
                    fill={c.id === selected ? "#e7f3ec" : "white"}
                    stroke={
                      c.status === "rejected"
                        ? "#c98071"
                        : c.status === "verified"
                          ? "#5a9b7f"
                          : "#b8c6be"
                    }
                    strokeWidth={c.id === selected ? 3 : 1.5}
                  />
                  <text x={14} y={21} fontSize={11} fill="#65786d">
                    {c.id} · {c.status}
                  </text>
                  <text x={14} y={43} fontSize={12} fill="#293f34">
                    {c.statement.slice(0, 34)}
                  </text>
                  <text x={14} y={63} fontSize={12} fill="#293f34">
                    {c.statement.slice(34, 67)}
                    {c.statement.length > 67 ? "…" : ""}
                  </text>
                </g>
              );
            })}
          </svg>
        </div>
        <div className="legend padded">
          <Badge value="verified" />
          <Badge value="rejected" />
          <Badge value="proposed" />
          <span className="muted">Symbolically verified ≠ formally proved</span>
        </div>
      </section>
      {claim ? (
        <ClaimCard claim={claim} state={state} />
      ) : (
        <div className="notice">
          Select a claim to see its full statement, derivation, and review
          records.
        </div>
      )}
    </>
  );
}
function GraphPlannerStatus({ state }: { state: State }) {
  const data = state.planner_data ?? {};
  const status = (data.status ?? {}) as {
    mode?: string;
    active?: boolean;
    reason?: string;
    checkpoint?: string;
    architecture?: { width: number; depth: number };
  };
  const predictions = (data.predictions ?? {}) as Record<string, number[]>;
  const traces = Array.isArray(data.traces) ? data.traces.length : 0;
  return (
    <section className="panel compact">
      <h2>Experimental graph planner</h2>
      <Badge
        value={
          status.active
            ? status.mode === "blend"
              ? "active"
              : "shadow"
            : traces
              ? "recording"
              : "disabled"
        }
      />
      <p>
        {status.reason ??
          "Not enabled for this run. Enable planner capture in the worker configuration to record research decisions."}
      </p>
      <p>
        {status.architecture?.depth ?? 3} relational message-passing blocks · utility, normalized explorer
        token reservation, and question resolution heads. Lexical problem
        features; no pretrained language encoder.
      </p>
      <dl>
        <div>
          <dt>Recorded rounds</dt>
          <dd>{traces} / 100 retained</dd>
        </div>
        <div>
          <dt>Mode</dt>
          <dd>{status.mode ?? "Disabled"}</dd>
        </div>
        <div>
          <dt>Checkpoint</dt>
          <dd>{status.checkpoint?.slice(0, 16) ?? "None"}</dd>
        </div>
      </dl>
      <p>
        Shadow predictions do not change allocation. Synthetic checkpoints
        cannot enable blending. These scores do not verify claims.
      </p>
      {Object.keys(predictions).length > 0 && (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Candidate</th>
                <th>Utility</th>
                <th>Reserved token cost</th>
                <th>Resolution</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(predictions).map(([id, values]) => (
                <tr key={id}>
                  <td>
                    {state.branches.find((b) => b.id === id)?.title ??
                      state.questions.find((q) => q.id === id)?.question ??
                      id}
                  </td>
                  {values.map((v, i) => (
                    <td key={i}>{v.toFixed(3)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

function Policy({ state }: { state: State }) {
  const p = state.policy;
  return (
    <>
      <GraphPlannerStatus state={state} />
      <div className="notice">
        <strong>
          {p.enabled ? "Hybrid policy enabled" : "Heuristic policy selected"}
        </strong>
        <p>
          Neural outputs estimate research utility, not mathematical truth or
          calibrated probabilities. Learned weights belong to this run and
          survive checkpoint recovery.
        </p>
      </div>
      <div className="two-col">
        {[p.branch_network, p.question_network].map((n, i) => (
          <section className="panel compact" key={i}>
            <h2>{i ? "Question" : "Branch"} network</h2>
            {n ? (
              <>
                <Badge value={n.active ? "active" : "cold_start"} />
                <dl>
                  <div>
                    <dt>Architecture</dt>
                    <dd>
                      {n.input_size} → {n.hidden_size} ReLU → 1 sigmoid
                    </dd>
                  </div>
                  <div>
                    <dt>Training examples</dt>
                    <dd>{n.trained_examples}</dd>
                  </div>
                  <div>
                    <dt>Training loss</dt>
                    <dd>{n.loss?.toFixed(6) ?? "Not trained"}</dd>
                  </div>
                  <div>
                    <dt>Epochs</dt>
                    <dd>{n.epochs}</dd>
                  </div>
                  <div>
                    <dt>Activation threshold</dt>
                    <dd>
                      {i ? p.min_question_examples : p.min_branch_examples}{" "}
                      examples
                    </dd>
                  </div>
                </dl>
              </>
            ) : (
              <Empty>No network trained yet.</Empty>
            )}
          </section>
        ))}
      </div>
      <section className="panel compact">
        <h2>Policy influence by branch</h2>
        <p className="muted">
          Maximum neural blend {(p.max_neural_blend * 100).toFixed(0)}%.
          Terminated branches retain their lifecycle state.
        </p>
        {state.branches.length ? (
          <div className="chart tall">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={state.branches.map((b) => ({
                  name: b.title,
                  heuristic: b.heuristic_score,
                  neural: b.policy_score,
                  final: b.score,
                }))}
                layout="vertical"
                margin={{ left: 15, right: 20 }}
              >
                <CartesianGrid horizontal={false} strokeDasharray="3 3" />
                <XAxis type="number" domain={[0, 1]} />
                <YAxis
                  dataKey="name"
                  type="category"
                  width={130}
                  tick={{ fontSize: 12 }}
                />
                <Tooltip />
                <Bar dataKey="heuristic" fill="#a4b9c7" />
                <Bar dataKey="neural" fill="#c5b48d" />
                <Bar dataKey="final" fill="#367d62" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <Empty>Branch scores appear after research.</Empty>
        )}
        <div className="legend">
          <span>
            <i style={{ background: "#a4b9c7" }} />
            Heuristic
          </span>
          <span>
            <i style={{ background: "#c5b48d" }} />
            Neural
          </span>
          <span>
            <i style={{ background: "#367d62" }} />
            Final
          </span>
        </div>
      </section>
    </>
  );
}
function Budget({ state, provider }: { state: State; provider: string }) {
  const b = state.budget;
  const rows = [
    ["Model calls", b.model_calls_used, b.max_model_calls],
    [
      "Reserved output tokens",
      b.reserved_output_tokens_used,
      b.max_reserved_output_tokens,
    ],
    ["Tool calls", b.tool_calls_used, b.max_tool_calls],
    ["Rounds", state.current_round, b.max_rounds],
  ] as const;
  return (
    <>
      <div className="metrics">
        <Metric
          label="Provider charges"
          value={provider === "mock" ? "$0" : "Unavailable"}
          caption={
            provider === "mock"
              ? "Deterministic local provider"
              : "Consult your provider billing dashboard"
          }
        />
        <Metric
          label="Output reservations"
          value={b.reserved_output_tokens_used.toLocaleString()}
          caption="Not actual token consumption"
        />
        <Metric label="Model invocations" value={state.invocations.length} />
        <Metric
          label="Tool invocations"
          value={state.tool_invocations.length}
        />
      </div>
      <section className="panel compact">
        <h2>Engine budget utilization</h2>
        {rows.map(([name, used, max]) => (
          <div className="budget-row" key={name}>
            <div className="score-row">
              <strong>{name}</strong>
              <span>
                {used.toLocaleString()} / {max.toLocaleString()}
              </span>
            </div>
            <div className="progress">
              <span
                style={{ width: `${Math.min(100, (used / max) * 100)}%` }}
              />
            </div>
          </div>
        ))}
      </section>
      <div className="notice">
        <strong>Accounting boundary</strong>
        <p>
          These counters cover committed engine work. Interrupted rounds can
          issue paid requests before a checkpoint and may be replayed. They are
          not a billing ledger or a hard dollar cap. Configure provider-side
          spend limits before live use.
        </p>
      </div>
    </>
  );
}
function Events({ id, events }: { id: string; events: Event[] }) {
  const [query, setQuery] = useState("");
  const [older, setOlder] = useState<Event[]>([]);
  const [cursor, setCursor] = useState(0);
  const [error, setError] = useState<Error>();
  const [busy, setBusy] = useState(false);
  async function load() {
    setBusy(true);
    try {
      const batch = await api<Event[]>(
        `/runs/${id}/events?after=${cursor}&limit=200`,
      );
      setOlder((old) => [...old, ...batch]);
      if (batch.length) setCursor(batch.at(-1)!.id);
      setError(undefined);
    } catch (e) {
      setError(e as Error);
    } finally {
      setBusy(false);
    }
  }
  const all = [
    ...new Map([...older, ...events].map((e) => [e.id, e])).values(),
  ].sort((a, b) => b.id - a.id);
  const filtered = all.filter(
    (e) =>
      e.kind.includes(query) ||
      JSON.stringify(e.payload).toLowerCase().includes(query.toLowerCase()),
  );
  return (
    <>
      <div className="filter-bar">
        <label className="sr-only" htmlFor="event-search">
          Search events
        </label>
        <input
          id="event-search"
          placeholder="Filter events…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <button disabled={busy} onClick={() => void load()}>
          Load history page
        </button>
      </div>
      <p className="muted">
        Live buffer: latest 1,000 events. Load history pages from the beginning
        for older records. Engine events commit with the round.
      </p>
      {error && <ErrorBox error={error} />}
      <section className="panel">
        {filtered.length ? (
          filtered.map((e) => (
            <details className="event-row" key={e.id}>
              <summary>
                <span className="mono">#{e.id}</span>
                <strong>{e.kind.replaceAll("_", " ")}</strong>
                <time>{new Date(e.created_at).toLocaleTimeString()}</time>
              </summary>
              <pre>{JSON.stringify(e.payload, null, 2)}</pre>
            </details>
          ))
        ) : (
          <Empty>No matching events.</Empty>
        )}
      </section>
    </>
  );
}
