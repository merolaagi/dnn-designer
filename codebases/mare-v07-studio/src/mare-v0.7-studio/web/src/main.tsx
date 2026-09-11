import Studio from "./Studio";
import {
  StrictMode,
  useCallback,
  useState,
  useEffect,
  lazy,
  Suspense,
} from "react";
import { createRoot } from "react-dom/client";
import {
  BrowserRouter,
  NavLink,
  Link,
  Route,
  Routes,
  useNavigate,
} from "react-router-dom";
import {
  Activity,
  ArrowUpRight,
  Beaker,
  ChevronRight,
  CircleHelp,
  LayoutDashboard,
  FlaskConical,
  Plus,
  Settings as SettingsIcon,
  ShieldCheck,
} from "lucide-react";
import { api, create, runs, setToken, type Create } from "./api";
import { useResource } from "./hooks";
import "./style.css";
const RunPage = lazy(() => import("./RunPage"));
import { registerResearchTools } from "./webmcp";
import { Badge, Empty, ErrorBox, Loading, Metric } from "./ui";

export function Dashboard() {
  const { data, error, loading, reload } = useResource(runs);
  return (
    <>
      <div className="page-heading">
        <div>
          <div className="eyebrow">WORKSPACE / OVERVIEW</div>
          <h1>Research dashboard</h1>
          <p>Follow an investigation from question to evidence.</p>
        </div>
        <Link className="button primary" to="/new">
          <Plus size={17} />
          Create run
        </Link>
      </div>
      <div className="metrics">
        <Metric
          label="Recent investigations"
          value={data?.length ?? "—"}
          caption="Latest 50 runs"
        />
        <Metric
          label="In progress"
          value={
            data?.filter((r) =>
              ["running", "queued", "cancelling"].includes(r.status),
            ).length ?? "—"
          }
        />
        <Metric
          label="Completed runs"
          value={data?.filter((r) => r.status === "completed").length ?? "—"}
          caption="Execution complete; proof status is separate"
        />
        <Metric
          label="Engine"
          value="v0.4"
          caption="Neural + symbolic research"
        />
      </div>
      <section className="panel">
        <div className="section-heading">
          <h2>Investigations</h2>
          <button onClick={reload}>Refresh</button>
        </div>
        {error ? (
          <ErrorBox error={error} retry={reload} />
        ) : loading && !data ? (
          <Loading />
        ) : data?.length ? (
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Research problem</th>
                  <th>Status</th>
                  <th>Progress</th>
                  <th>Provider</th>
                  <th>Created</th>
                </tr>
              </thead>
              <tbody>
                {data.map((r) => (
                  <tr key={r.id}>
                    <td>
                      <Link
                        className="run-link"
                        to={`/${r.workflow === "studio" ? "studio" : "runs"}/${r.id}`}
                      >
                        <strong>{r.title}</strong>
                        <ArrowUpRight size={16} />
                      </Link>
                      <small className="mono">{r.id}</small>
                    </td>
                    <td>
                      <Badge value={r.status} />
                    </td>
                    <td>
                      <div className="progress">
                        <span
                          style={{
                            width: `${(r.current_round / r.target_rounds) * 100}%`,
                          }}
                        />
                      </div>
                      <small>
                        {r.current_round} / {r.target_rounds} rounds
                      </small>
                    </td>
                    <td>
                      <Badge value={r.provider} />
                    </td>
                    <td>{new Date(r.created_at).toLocaleDateString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <Empty>
            No investigations yet. Create a deterministic demo run to explore
            MARE.
          </Empty>
        )}
      </section>
      <div className="two-col">
        <section className="panel compact">
          <ShieldCheck />
          <h2>Evidence before conclusions</h2>
          <p>
            Neural scores guide research allocation. Claims keep their
            verification, scope, and formalization status.
          </p>
        </section>
        <section className="panel compact">
          <Beaker />
          <h2>A reproducible starting point</h2>
          <p>
            The mock provider investigates Burgers gradient blow-up with
            symbolic checks and adversarial review. No API key required.
          </p>
          <Link to="/new">
            Start the benchmark <ChevronRight size={14} />
          </Link>
        </section>
      </div>
    </>
  );
}
function NewRun() {
  const navigate = useNavigate();
  const [error, setError] = useState<Error>();
  const [busy, setBusy] = useState(false);
  const [provider, setProvider] = useState("mock");
  const configLoader = useCallback(
    () => api<{ providers: string[] }>("/settings"),
    [],
  );
  const config = useResource(configLoader);
  async function submit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    setError(undefined);
    const f = new FormData(event.currentTarget);
    const body: Create = {
      title: String(f.get("title")),
      provider: provider as "mock" | "openai",
      rounds: Number(f.get("rounds")),
      policy: String(f.get("policy")) as "hybrid" | "heuristic",
      max_model_calls: Number(f.get("calls")),
      max_reserved_output_tokens: Number(f.get("tokens")),
      max_tool_calls: Number(f.get("tools")),
    };
    if (provider === "openai") {
      body.statement = String(f.get("statement"));
      body.assumptions = String(f.get("assumptions"))
        .split("\n")
        .filter(Boolean);
      body.success_conditions = String(f.get("success"))
        .split("\n")
        .filter(Boolean);
    }
    try {
      const r = await create(body);
      navigate(`/runs/${r.id}`);
    } catch (e) {
      setError(e as Error);
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <div className="eyebrow">WORKSPACE / NEW INVESTIGATION</div>
      <h1>Create research run</h1>
      <p>Define a research target and give the engine a bounded budget.</p>
      <form onSubmit={submit} className="panel form-panel">
        <h2>Research target</h2>
        <label>
          Run title
          <input
            name="title"
            required
            maxLength={200}
            defaultValue="Burgers gradient blow-up"
          />
        </label>
        <div className="form-grid">
          <label>
            Provider
            <select
              value={provider}
              onChange={(e) => setProvider(e.target.value)}
            >
              <option value="mock">Deterministic mock · no key needed</option>
              <option
                value="openai"
                disabled={!config.data?.providers.includes("openai")}
              >
                OpenAI · server configuration required
              </option>
            </select>
          </label>
          <label>
            Research policy
            <select name="policy">
              <option value="hybrid">Hybrid neural + heuristic</option>
              <option value="heuristic">Heuristic only</option>
            </select>
          </label>
        </div>
        {provider === "mock" ? (
          <div className="notice">
            <strong>Inviscid Burgers benchmark</strong>
            <p>
              Investigate finite-time gradient blow-up for uₜ + u uₓ = 0. The
              problem, assumptions, and checks are fixed by the preserved mock
              provider.
            </p>
          </div>
        ) : (
          <>
            <label>
              Canonical statement
              <textarea
                name="statement"
                required
                minLength={10}
                maxLength={12000}
              />
            </label>
            <label>
              Assumptions · one per line
              <textarea name="assumptions" />
            </label>
            <label>
              Success conditions · one per line
              <textarea name="success" />
            </label>
            <div className="notice">
              The inherited symbolic checks are Burgers-specific. Unrecognized
              checks remain inconclusive; a model proposal is not a proof.
            </div>
          </>
        )}
        <h2>Execution budget</h2>
        <div className="form-grid">
          <label>
            Rounds
            <input
              name="rounds"
              type="number"
              min={1}
              max={30}
              defaultValue={2}
              required
            />
          </label>
          <label>
            Maximum model calls
            <input
              name="calls"
              type="number"
              min={1}
              max={2000}
              defaultValue={250}
              required
            />
          </label>
          <label>
            Reserved output tokens
            <input
              name="tokens"
              type="number"
              min={1}
              max={8000000}
              defaultValue={1000000}
              required
            />
          </label>
          <label>
            Maximum tool calls
            <input
              name="tools"
              type="number"
              min={1}
              max={10000}
              defaultValue={2000}
              required
            />
          </label>
        </div>
        <p className="muted">
          Token reservations are engine limits, not billed usage or a dollar
          spending cap.
        </p>
        {error && <ErrorBox error={error} />}
        <div className="form-actions">
          <Link to="/">Cancel</Link>
          <button className="primary" disabled={busy}>
            {busy ? "Queuing…" : "Queue research run"}
            <ChevronRight size={16} />
          </button>
        </div>
      </form>
    </>
  );
}
function Settings() {
  const loader = useCallback(
    () => api<Record<string, unknown>>("/settings"),
    [],
  );
  const { data, error, reload } = useResource(loader);
  const [saved, setSaved] = useState(false);
  return (
    <>
      <div className="eyebrow">WORKSPACE / CONFIGURATION</div>
      <h1>Settings</h1>
      <section className="panel form-panel">
        <h2>Workspace access</h2>
        <p>
          Enter the server’s workspace token when authentication is enabled. It
          is held only in this page’s memory and clears on reload.
        </p>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            setToken(String(new FormData(e.currentTarget).get("token")));
            e.currentTarget.reset();
            setSaved(true);
            reload();
          }}
        >
          <label>
            Bearer token
            <input name="token" type="password" autoComplete="off" />
          </label>
          <button className="primary">Apply token</button>
          {saved && <span role="status"> Token applied for this session.</span>}
        </form>
      </section>
      <section className="panel compact">
        <h2>Server configuration</h2>
        {error ? (
          <ErrorBox error={error} retry={reload} />
        ) : data ? (
          <dl>
            {Object.entries(data).map(([key, value]) => (
              <div key={key}>
                <dt>{key.replaceAll("_", " ")}</dt>
                <dd>
                  {Array.isArray(value)
                    ? value.join(", ")
                    : String(value ?? "Not configured")}
                </dd>
              </div>
            ))}
          </dl>
        ) : (
          <Loading />
        )}
        <p className="muted">
          Provider credentials, database access, origins, and worker limits are
          configured on the server. No secrets are exposed here.
        </p>
      </section>
    </>
  );
}
function App() {
  useEffect(registerResearchTools, []);
  return (
    <BrowserRouter>
      <div className="app-shell">
        <aside className="sidebar">
          <Link to="/" className="brand">
            <span className="brand-symbol">M</span>
            <span>
              MARE<small>RESEARCH WORKBENCH</small>
            </span>
          </Link>
          <div className="workspace-label">
            <span className="dot" />
            Research workspace
          </div>
          <nav>
            <NavLink to="/" end>
              <LayoutDashboard size={18} />
              Dashboard
            </NavLink>
            <NavLink to="/studio">
              <FlaskConical size={18} /> Paper studio
            </NavLink>
            <NavLink to="/new">
              <Plus size={18} />
              Create run
            </NavLink>
            <NavLink to="/settings">
              <SettingsIcon size={18} />
              Settings
            </NavLink>
          </nav>
          <div className="sidebar-note">
            <CircleHelp size={18} />
            <p>
              Explore freely.
              <br />
              Verify rigorously.
            </p>
            <small>v0.7-studio</small>
          </div>
        </aside>
        <div className="main-shell">
          <header className="topbar">
            <span>
              <Activity size={16} />
              Research operations
            </span>
            <span className="muted">Single workspace</span>
          </header>
          <main>
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/new" element={<NewRun />} />
              <Route path="/studio" element={<Studio />} />
              <Route path="/studio/:id" element={<Studio />} />
              <Route path="/settings" element={<Settings />} />
              <Route
                path="/runs/:id/*"
                element={
                  <Suspense fallback={<Loading />}>
                    <RunPage />
                  </Suspense>
                }
              />
              <Route
                path="*"
                element={
                  <Empty>
                    Page not found. <Link to="/">Return to dashboard</Link>
                  </Empty>
                }
              />
            </Routes>
          </main>
        </div>
      </div>
    </BrowserRouter>
  );
}
createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
