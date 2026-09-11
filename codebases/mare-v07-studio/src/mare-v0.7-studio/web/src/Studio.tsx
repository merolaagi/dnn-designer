import { useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api, detail, type Detail } from "./api";
import { Badge, ErrorBox, Loading } from "./ui";
import "./studio.css";

type Model = {
  name: string;
  variables: string[];
  derivatives: string[];
  initial: number[];
  parameters: { name: string; value: number }[];
  horizon: number;
  assumptions: string[];
  provenance: string;
};
type Finding = {
  kind: string;
  quote: string;
  interpretation: string;
  source_start: number;
  source_end: number;
  status: string;
};
type Experiment = {
  points: { t: number; y: number[] }[];
  first_step: {
    h: number;
    before: number[];
    k1: number[];
    k2: number[];
    k3: number[];
    k4: number[];
    after: number[];
  } | null;
  grid_difference: number;
  comparison_complete: boolean;
  failure: { t: number; reason: string; meaning: string } | null;
  method: string;
  conclusion: string;
};
type Packet = {
  stage: string;
  source?: { text: string; sha256: string };
  input: { title: string; text: string };
  model?: Model;
  model_origin?: string;
  analysis?: {
    summary: string;
    findings: Finding[];
    limitations: string[];
    next_methods: string[];
    model_rationale: string;
  };
  experiment?: Experiment;
  trace?: { title: string; detail: string; status: string }[];
  error?: string;
  parent_run_id?: string;
};
const examples = [
  {
    label: "Mathematics · gradient blow-up",
    text: "Worked example authored for this studio, not an imported research paper.\nConsider the inviscid Burgers equation u_t + u*u_x = 0.\nAssume a smooth classical solution while following a characteristic dx/dt = u.\nSet m = u_x along that characteristic. Differentiating the PDE yields dm/dt = -m**2.\nFor m(0) = -1, the reduced equation gives m(t) = -1/(1-t) while t < 1.\nThis reduced calculation does not solve the Navier-Stokes global regularity problem.\nOpen question: which assumptions allow a related mechanism in a different PDE?",
    model: {
      name: "Burgers characteristic gradient",
      variables: ["m"],
      derivatives: ["-m**2"],
      initial: [-1],
      parameters: [],
      horizon: 0.9,
      assumptions: [
        "A smooth inviscid Burgers solution exists up to the times considered.",
        "m follows a characteristic; this is not the full PDE simulation.",
      ],
      provenance:
        "Explicit worked example reduction; not a model learned from an external paper.",
    },
  },
  {
    label: "Biology · population growth",
    text: "Worked example, not a research paper.\nAssume a well-mixed population with constant growth rate r and carrying capacity K.\nThe logistic growth model is dx/dt = r*x*(1-x/K).\nThe initial population is x(0) = 1.\nOpen question: what observations would contradict a constant carrying capacity?",
    model: {
      name: "Logistic population",
      variables: ["x"],
      derivatives: ["r*x*(1-x/K)"],
      initial: [1],
      parameters: [
        { name: "r", value: 1 },
        { name: "K", value: 10 },
      ],
      horizon: 8,
      assumptions: [
        "Constant carrying capacity.",
        "No spatial structure or stochastic noise.",
      ],
      provenance:
        "Illustrative model with learner-selected parameters; not fitted biological evidence.",
    },
  },
  {
    label: "Chemistry · first-order decay",
    text: "Worked example, not a research paper.\nAssume a well-mixed closed system and a constant rate k for a first-order conversion A to product.\nThe concentration obeys dx/dt = -k*x.\nThe initial concentration is x(0) = 1.\nOpen question: is a first-order rate law compatible with measured concentration trajectories?",
    model: {
      name: "First-order decay",
      variables: ["x"],
      derivatives: ["-k*x"],
      initial: [1],
      parameters: [{ name: "k", value: 0.5 }],
      horizon: 8,
      assumptions: [
        "First-order rate law.",
        "Constant temperature and rate coefficient.",
      ],
      provenance:
        "Illustrative kinetic equation, not an experimentally validated mechanism.",
    },
  },
] satisfies { label: string; text: string; model: Model }[];
const clone = <T,>(value: T): T => JSON.parse(JSON.stringify(value));

function Theatre({
  model,
  experiment,
  onClose,
}: {
  model: Model;
  experiment: Experiment;
  onClose: () => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null),
    canvas = useRef<HTMLCanvasElement>(null),
    recorder = useRef<MediaRecorder | null>(null),
    media = useRef<MediaStream | null>(null);
  const mounted = useRef(true);
  const [frame, setFrame] = useState(0),
    [playing, setPlaying] = useState(false),
    [recording, setRecording] = useState(false),
    [message, setMessage] = useState("");
  useEffect(() => {
    mounted.current = true;
    dialog.current?.showModal();
    return () => {
      mounted.current = false;
      if (recorder.current?.state === "recording") recorder.current.stop();
      media.current?.getTracks().forEach((t) => t.stop());
    };
  }, []);
  useEffect(() => {
    if (!playing) return;
    const timer = setInterval(
      () =>
        setFrame((f) => {
          if (f >= experiment.points.length - 1) {
            setPlaying(false);
            return f;
          }
          return f + 1;
        }),
      50,
    );
    return () => clearInterval(timer);
  }, [playing, experiment.points.length]);
  useEffect(() => {
    const ctx = canvas.current?.getContext("2d");
    if (!ctx) return;
    const W = 900,
      H = 450;
    ctx.fillStyle = "#f5f7f4";
    ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = "#213a34";
    ctx.font = "bold 20px sans-serif";
    ctx.fillText(model.name.slice(0, 65), 30, 32);
    ctx.font = "14px monospace";
    ctx.fillText(
      model.variables
        .map((v, i) => `d${v}/dt = ${model.derivatives[i]}`)
        .join(" ; ")
        .slice(0, 100),
      30,
      58,
    );
    const points = experiment.points,
      vals = points.flatMap((p) => p.y);
    let min = Math.min(...vals),
      max = Math.max(...vals);
    if (max === min) {
      max += 1;
      min -= 1;
    }
    const pad = (max - min) * 0.1;
    min -= pad;
    max += pad;
    const X = (t: number) => 65 + (780 * t) / model.horizon,
      Y = (y: number) => 355 - (260 * (y - min)) / (max - min);
    ctx.strokeStyle = "#b6c5be";
    ctx.beginPath();
    ctx.moveTo(65, 85);
    ctx.lineTo(65, 355);
    ctx.lineTo(845, 355);
    ctx.stroke();
    ctx.fillStyle = "#425e54";
    ctx.font = "13px sans-serif";
    ctx.fillText(`t = ${points[frame]?.t.toFixed(4)}`, 65, 385);
    ctx.fillText(`${max.toPrecision(4)}`, 5, 95);
    ctx.fillText(`${min.toPrecision(4)}`, 5, 355);
    const colors = ["#2d7564", "#9a623c", "#4b67a0", "#7a558a"];
    model.variables.forEach((v, j) => {
      ctx.strokeStyle = colors[j];
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      points.slice(0, frame + 1).forEach((p, i) => {
        if (i === 0) ctx.moveTo(X(p.t), Y(p.y[j]));
        else ctx.lineTo(X(p.t), Y(p.y[j]));
      });
      ctx.stroke();
      const p = points[frame];
      ctx.fillStyle = colors[j];
      ctx.beginPath();
      ctx.arc(X(p.t), Y(p.y[j]), 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillText(`${v} = ${p.y[j].toPrecision(5)}`, 280 + j * 140, 385);
    });
    ctx.fillStyle = "#5e6d66";
    ctx.fillText(
      "Recorded RK4 trajectory · numerical illustration, not a proof",
      65,
      420,
    );
  }, [frame, model, experiment]);
  useEffect(() => {
    if (
      frame === experiment.points.length - 1 &&
      recorder.current?.state === "recording"
    )
      recorder.current.stop();
  }, [frame, experiment.points.length]);
  function record() {
    try {
      const element = canvas.current;
      if (
        !element ||
        !("captureStream" in element) ||
        typeof MediaRecorder === "undefined"
      ) {
        setMessage(
          "Video recording is unavailable in this browser. The timeline remains interactive.",
        );
        return;
      }
      const mime = [
        "video/webm;codecs=vp9",
        "video/webm;codecs=vp8",
        "video/webm",
      ].find((t) => MediaRecorder.isTypeSupported(t));
      if (!mime) {
        setMessage(
          "This browser cannot encode WebM. Use an up-to-date Chromium browser.",
        );
        return;
      }
      setFrame(0);
      const stream = element.captureStream(20);
      media.current = stream;
      const r = new MediaRecorder(stream, { mimeType: mime });
      recorder.current = r;
      const chunks: Blob[] = [];
      r.ondataavailable = (e) => {
        if (e.data.size) chunks.push(e.data);
      };
      r.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        if (!mounted.current) return;
        setRecording(false);
        setMessage("Animation exported as WebM from the recorded trajectory.");
        const url = URL.createObjectURL(new Blob(chunks, { type: mime }));
        const a = document.createElement("a");
        a.href = url;
        a.download = "mare-experiment.webm";
        a.click();
        setTimeout(() => URL.revokeObjectURL(url), 30000);
      };
      r.start();
      setRecording(true);
      setPlaying(true);
      setMessage(
        "Recording the calculated trajectory. Export finishes at the last frame.",
      );
    } catch (e) {
      setMessage(String(e));
    }
  }
  return (
    <dialog ref={dialog} className="theatre" onCancel={onClose}>
      <div className="studio-row">
        <h2>Research theatre</h2>
        <button onClick={onClose}>Close</button>
      </div>
      <canvas
        ref={canvas}
        width={900}
        height={450}
        aria-label="Animated computed trajectories"
      />
      <div className="studio-row">
        <button disabled={recording} onClick={() => setPlaying(!playing)}>
          {playing ? "Pause" : "Play"}
        </button>
        <button
          disabled={recording}
          onClick={() => {
            setFrame(0);
            setPlaying(false);
          }}
        >
          Restart
        </button>
        <button
          disabled={recording || experiment.points.length < 2}
          onClick={record}
        >
          Export animation as WebM
        </button>
      </div>
      <label>
        Recorded step {frame} / {experiment.points.length - 1}
        <input
          type="range"
          min={0}
          max={experiment.points.length - 1}
          value={frame}
          disabled={recording}
          onChange={(e) => {
            setPlaying(false);
            setFrame(Number(e.target.value));
          }}
        />
      </label>
      <p role="status">{message}</p>
      <p>{experiment.conclusion}</p>
      <details>
        <summary>How the first RK4 update was computed</summary>
        <p>
          yₙ₊₁ = yₙ + h(k₁ + 2k₂ + 2k₃ + k₄)/6. These are recorded slopes from
          this experiment.
        </p>
        <pre>{JSON.stringify(experiment.first_step, null, 2)}</pre>
      </details>
    </dialog>
  );
}

export default function Studio() {
  const [objective, setObjective] = useState("");
  const { id } = useParams(),
    navigate = useNavigate();
  const [run, setRun] = useState<Detail | null>(null),
    [error, setError] = useState<unknown>(null),
    [busy, setBusy] = useState(false);
  const [title, setTitle] = useState("Paper-inspired experiment"),
    [text, setText] = useState(""),
    [pdf, setPdf] = useState<string | undefined>(),
    [fileName, setFileName] = useState(""),
    [provider, setProvider] = useState("mock"),
    [model, setModel] = useState<Model | null>(null),
    [parent, setParent] = useState<string | undefined>(),
    [theatre, setTheatre] = useState(false),
    [selected, setSelected] = useState<Finding | null>(null);
  useEffect(() => {
    if (!id) {
      setRun(null);
      return;
    }
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    async function load() {
      try {
        const r = await detail(id!);
        if (!cancelled) {
          setRun(r);
          setError(null);
          if (
            !["completed", "failed", "cancelled", "budget_exhausted"].includes(
              r.status,
            )
          )
            timer = setTimeout(load, 1200);
        }
      } catch (e) {
        if (!cancelled) setError(e);
      }
    }
    load();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [id]);
  const quoteRef = useRef<HTMLElement>(null);
  useEffect(() => {
    if (selected)
      quoteRef.current?.scrollIntoView({ behavior: "smooth", block: "center" });
  }, [selected]);
  const packet = run?.snapshot.studio_data as Packet | undefined;
  async function submit() {
    setBusy(true);
    setError(null);
    try {
      const r = await api<{ id: string }>("/studio", {
        method: "POST",
        body: JSON.stringify({
          title,
          text,
          pdf_base64: pdf,
          provider,
          model,
          parent_run_id: parent,
        }),
      });
      setRun(null);
      navigate(`/studio/${r.id}`);
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  }
  async function upload(file?: File) {
    if (!file) return;
    setError(null);
    if (file.size > 1000000) {
      setError(new Error("Choose a PDF or text excerpt under 1 MB."));
      return;
    }
    setModel(null);
    setFileName(file.name);
    if (file.name.toLowerCase().endsWith(".pdf")) {
      const bytes = new Uint8Array(await file.arrayBuffer());
      let binary = "";
      bytes.forEach((b) => (binary += String.fromCharCode(b)));
      setPdf(btoa(binary));
      setText("");
    } else {
      setText(await file.text());
      setPdf(undefined);
    }
  }
  function fork() {
    if (!packet) return;
    setTitle(`${run!.title} · alternative`);
    setText(packet.source?.text ?? "");
    setPdf(undefined);
    setModel(packet.model ? clone(packet.model) : null);
    setParent(run!.id);
    navigate("/studio");
  }
  function exportRecord() {
    if (!run) return;
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(run, null, 2)], { type: "application/json" }),
    );
    const link = document.createElement("a");
    link.href = url;
    link.download = `${run.id}-research-record.json`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 30000);
  }
  async function launchResearch() {
    setBusy(true);
    setError(null);
    try {
      const r = await api<{ id: string }>(`/studio/${id}/research`, {
        method: "POST",
        body: JSON.stringify({ objective }),
      });
      navigate(`/runs/${r.id}`);
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  }
  function updateModel(update: Partial<Model>) {
    if (model)
      setModel({
        ...model,
        ...update,
        provenance:
          "Learner-modified proposal; re-check the source assumptions and units.",
      });
  }
  return (
    <div className="studio">
      <div className="studio-row">
        <div>
          <p className="eyebrow">
            SOURCE → HYPOTHESIS → EXPERIMENT → EXPLANATION
          </p>
          <h1>Paper research studio</h1>
        </div>
        <Link
          to="/studio"
          onClick={() => {
            setRun(null);
            setParent(undefined);
          }}
        >
          New investigation
        </Link>
      </div>
      <p>
        Study a source, propose a method, and watch its calculations. Every
        tested result stays separate from the paper’s claims and from
        mathematical proof.
      </p>
      {error != null && (
        <ErrorBox
          error={error instanceof Error ? error : new Error(String(error))}
          retry={() => (id ? window.location.reload() : setError(null))}
        />
      )}
      {!id ? (
        <>
          <section className="panel compact">
            <h2>Start with a worked example</h2>
            <div className="studio-row">
              {examples.map((e) => (
                <button
                  key={e.label}
                  onClick={() => {
                    setTitle(e.model.name);
                    setText(e.text);
                    setPdf(undefined);
                    setFileName("");
                    setModel(clone(e.model));
                  }}
                >
                  {e.label}
                </button>
              ))}
            </div>
            <p>
              These are authored learning examples, not fabricated research
              papers.
            </p>
          </section>
          <section className="panel compact">
            <h2>1. Bring the source</h2>
            <label>
              Investigation title
              <input value={title} onChange={(e) => setTitle(e.target.value)} />
            </label>
            <label>
              Import text or a text-based PDF
              <input
                type="file"
                accept=".pdf,.txt,.md,.tex"
                onChange={(e) => void upload(e.target.files?.[0])}
              />
            </label>
            {fileName && <p>{fileName}</p>}
            <label>
              Paper text or mathematical content
              <textarea
                rows={9}
                value={text}
                disabled={!!pdf}
                maxLength={60000}
                onChange={(e) => setText(e.target.value)}
              />
            </label>
            <label>
              Interpretation method
              <select
                value={provider}
                onChange={(e) => setProvider(e.target.value)}
              >
                <option value="mock">
                  Offline passage detection — no language understanding
                </option>
                <option value="openai">
                  OpenAI structured paper interpretation — server key required
                </option>
              </select>
            </label>
            {provider === "openai" && (
              <p>
                The imported text will be sent to the server-configured OpenAI
                model for interpretation. Its proposals require scientific
                review.
              </p>
            )}
          </section>
          <section className="panel compact">
            <h2>2. Choose an experimental model</h2>
            <p>
              Supports up to four coupled first-order ODEs. Other mathematics
              remains a source analysis until an appropriate model or solver is
              supplied.
            </p>
            {!model ? (
              <>
                <p>
                  No model selected. OpenAI may propose a supported model;
                  offline mode will retain the extracted passages without
                  inventing one.
                </p>
                <button
                  onClick={() =>
                    setModel({
                      name: "My proposed method",
                      variables: ["x"],
                      derivatives: ["0"],
                      initial: [1],
                      parameters: [],
                      horizon: 4,
                      assumptions: [],
                      provenance:
                        "Learner-proposed model; not derived automatically from the source.",
                    })
                  }
                >
                  Create an editable model without changing the source
                </button>
              </>
            ) : (
              <>
                <label>
                  Model name
                  <input
                    value={model.name}
                    onChange={(e) => updateModel({ name: e.target.value })}
                  />
                </label>
                {model.variables.map((v, i) => (
                  <div className="model-equation" key={i}>
                    <label>
                      Variable
                      <input
                        value={v}
                        onChange={(e) =>
                          updateModel({
                            variables: model.variables.map((x, j) =>
                              j === i ? e.target.value : x,
                            ),
                          })
                        }
                      />
                    </label>
                    <label>
                      d{v}/dt
                      <input
                        value={model.derivatives[i]}
                        onChange={(e) =>
                          updateModel({
                            derivatives: model.derivatives.map((x, j) =>
                              j === i ? e.target.value : x,
                            ),
                          })
                        }
                      />
                    </label>
                    <label>
                      {v}(0)
                      <input
                        type="number"
                        value={model.initial[i]}
                        onChange={(e) =>
                          updateModel({
                            initial: model.initial.map((x, j) =>
                              j === i ? Number(e.target.value) : x,
                            ),
                          })
                        }
                      />
                    </label>
                  </div>
                ))}
                <button
                  disabled={model.variables.length >= 4}
                  onClick={() =>
                    updateModel({
                      variables: [
                        ...model.variables,
                        `v${model.variables.length}`,
                      ],
                      derivatives: [...model.derivatives, "0"],
                      initial: [...model.initial, 1],
                    })
                  }
                >
                  Add state variable
                </button>
                {model.parameters.map((p, i) => (
                  <div className="studio-row" key={i}>
                    <label>
                      Parameter
                      <input
                        value={p.name}
                        onChange={(e) =>
                          updateModel({
                            parameters: model.parameters.map((x, j) =>
                              j === i ? { ...x, name: e.target.value } : x,
                            ),
                          })
                        }
                      />
                    </label>
                    <label>
                      Value
                      <input
                        type="number"
                        value={p.value}
                        onChange={(e) =>
                          updateModel({
                            parameters: model.parameters.map((x, j) =>
                              j === i
                                ? { ...x, value: Number(e.target.value) }
                                : x,
                            ),
                          })
                        }
                      />
                    </label>
                  </div>
                ))}
                <button
                  disabled={model.parameters.length >= 8}
                  onClick={() =>
                    updateModel({
                      parameters: [
                        ...model.parameters,
                        { name: `p${model.parameters.length}`, value: 1 },
                      ],
                    })
                  }
                >
                  Add parameter
                </button>
                <label>
                  End time
                  <input
                    type="number"
                    min={0.001}
                    max={100}
                    step={0.1}
                    value={model.horizon}
                    onChange={(e) =>
                      updateModel({ horizon: Number(e.target.value) })
                    }
                  />
                </label>
                <p>
                  Allowed: + − * / **, sin, cos, exp, log, sqrt, abs. No Python
                  code or external tools are executed.
                </p>
                <label>
                  Assumptions and units, one per line
                  <textarea
                    rows={4}
                    value={model.assumptions.join("\n")}
                    onChange={(e) =>
                      updateModel({
                        assumptions: e.target.value.split("\n").filter(Boolean),
                      })
                    }
                  />
                </label>
                <button onClick={() => setModel(null)}>
                  Remove model; analyze source only
                </button>
              </>
            )}
            <button
              className="primary"
              disabled={busy || (!text && !pdf && !model)}
              onClick={() => void submit()}
            >
              {busy ? "Queuing…" : "Analyze and test proposal"}
            </button>
            {parent && (
              <p>
                Alternative to investigation {parent}. The earlier experiment is
                preserved.
              </p>
            )}
          </section>
        </>
      ) : !run ? (
        <Loading />
      ) : (
        <>
          <div className="studio-row">
            <Badge value={run.status} />
            <strong>{packet?.stage ?? "Not a studio investigation"}</strong>
            <Link to={`/runs/${run.id}/events`}>Full event log</Link>
            <button onClick={fork}>Fork and try your own method</button>
            <button onClick={exportRecord}>Export research record</button>
          </div>
          {run.error && <p className="notice">{run.error}</p>}
          {packet?.error && <p className="notice">{packet.error}</p>}
          <section className="panel compact">
            <h2>How this attempt unfolded</h2>
            <ol>
              {packet?.trace?.map((step, i) => (
                <li key={i}>
                  <strong>{step.title}</strong> <Badge value={step.status} />
                  <p>{step.detail}</p>
                </li>
              ))}
            </ol>
            {!packet?.trace?.length && (
              <p>
                The worker is preparing the source. Updates arrive as each stage
                is checkpointed.
              </p>
            )}
          </section>
          {packet?.analysis && (
            <section className="panel compact">
              <h2>Rules, equations, and open questions</h2>
              <p>{packet.analysis.summary}</p>
              {packet.analysis.findings.map((f, i) => (
                <article className="finding" key={i}>
                  <div className="studio-row">
                    <strong>{f.kind}</strong>
                    <Badge value={f.status} />
                    {f.source_start >= 0 && (
                      <button onClick={() => setSelected(f)}>
                        Locate in source
                      </button>
                    )}
                  </div>
                  <blockquote>{f.quote}</blockquote>
                  <p>{f.interpretation}</p>
                </article>
              ))}
              {packet.analysis.findings.length === 0 && (
                <p>
                  No candidate rules found. This is an extraction limitation,
                  not evidence that the source contains none.
                </p>
              )}
              <h3>Limitations and next methods</h3>
              <ul>
                {[
                  ...packet.analysis.limitations,
                  ...packet.analysis.next_methods,
                ].map((s, i) => (
                  <li key={i}>{s}</li>
                ))}
              </ul>
            </section>
          )}
          {packet?.model && (
            <section className="panel compact">
              <h2>The proposed model</h2>
              <p>
                {packet.model_origin} · {packet.model.provenance}
              </p>
              {packet.model.variables.map((v, i) => (
                <p className="equation" key={v}>
                  d{v}/dt = {packet.model!.derivatives[i]} ; {v}(0) ={" "}
                  {packet.model!.initial[i]}
                </p>
              ))}
              <ul>
                {packet.model.assumptions.map((a, i) => (
                  <li key={i}>{a}</li>
                ))}
              </ul>
              {packet.experiment && (
                <>
                  <p>
                    Maximum difference between sampled 200-step and 400-step
                    trajectories:{" "}
                    <strong>
                      {packet.experiment.grid_difference.toExponential(3)}
                    </strong>
                    .{" "}
                    {packet.experiment.comparison_complete
                      ? "Both trajectories completed."
                      : "Comparison covers only their shared computed interval."}
                  </p>
                  {packet.experiment.failure && (
                    <p className="notice">
                      Attempt failed near t=
                      {packet.experiment.failure.t.toPrecision(4)}:{" "}
                      {packet.experiment.failure.reason}.{" "}
                      {packet.experiment.failure.meaning}
                    </p>
                  )}
                  <button className="primary" onClick={() => setTheatre(true)}>
                    Open animated research theatre
                  </button>
                  <p>{packet.experiment.conclusion}</p>
                </>
              )}
            </section>
          )}
          {packet?.analysis && (
            <section className="panel compact">
              <h2>Carry this idea into MARE research</h2>
              <p>
                Investigate a harder question using the existing branch, critic,
                evidence, and proof-DAG workflow. The paper stays an unverified
                source. Requires the server-configured OpenAI provider; capped
                at 20 model calls and 80,000 reserved output tokens.
              </p>
              <label>
                Open problem or new method to investigate
                <textarea
                  rows={3}
                  value={objective}
                  onChange={(e) => setObjective(e.target.value)}
                  maxLength={4000}
                />
              </label>
              <button
                disabled={busy || objective.trim().length < 10}
                onClick={() => void launchResearch()}
              >
                Start MARE research from this source
              </button>
            </section>
          )}
          {packet?.source && (
            <section className="panel compact">
              <h2>Preserved source</h2>
              <small>SHA-256: {packet.source.sha256}</small>
              <pre className="source-text">
                {selected ? (
                  <>
                    {Array.from(packet.source.text)
                      .slice(0, selected.source_start)
                      .join("")}
                    <mark ref={quoteRef}>
                      {Array.from(packet.source.text)
                        .slice(selected.source_start, selected.source_end)
                        .join("")}
                    </mark>
                    {Array.from(packet.source.text)
                      .slice(selected.source_end)
                      .join("")}
                  </>
                ) : (
                  packet.source.text
                )}
              </pre>
            </section>
          )}
          {theatre && packet?.model && packet.experiment && (
            <Theatre
              model={packet.model}
              experiment={packet.experiment}
              onClose={() => setTheatre(false)}
            />
          )}
        </>
      )}
    </div>
  );
}
