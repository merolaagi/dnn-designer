# Deep Network Designer

A visual editor for neural networks that generates real PyTorch, trains it,
checks its own arithmetic against the framework, and refuses to guess when it
does not know.

Version 1.31.2 · 10,449 lines of Python, 6,868 of frontend, 112 tests · runs
locally at `127.0.0.1:8770`

---

## What it is

You draw a network as a diagram. It works out every shape, tells you the exact
parameter count, writes the PyTorch file, and trains it — all locally, with no
account, no upload and no service behind it.

That much exists elsewhere. Three things here are less common:

**It checks itself against PyTorch.** The canvas computes shapes and parameter
counts by arithmetic. *Test layer* builds the selected layer on its own, pushes
a real tensor through it, and reports whether the two agree. Every model it
imports is compared against what PyTorch says the original holds, and any gap is
reported with the exact number.

**It says when it does not know.** A layer whose mathematics is not written up
says so rather than producing something plausible. A model whose `forward`
cannot be traced is refused with the specific reason. A mixture of experts is
refused a diagram entirely, because a static graph would show every expert
running when only two do — the drawing would be wrong about the model even
though it was right about the code.

**It closes the loop from opinion to measurement.** The assistant reviews a
network and finds real problems. The repair agent then trains each of its
suggestions against the unchanged network and tells you which actually helped.
On a test network, dropout won, an activation between two dense layers came
second, and batch normalization made things slightly worse.

---

## What it can do

### Drawing

- **63 layer types** across 19 categories, plus 21 plug-in blocks you can edit
  as Python and reload without restarting.
- Shapes propagate as you draw. An error is attached to the layer that caused it
  and phrased as a sentence, not a stack trace.
- Node shape follows flowchart convention rather than decoration: circles are
  terminators, diamonds are merges, parallelograms are the data symbol for
  Flatten and Reshape, side-barred rectangles are predefined processes for
  plug-in blocks, pills are activations.
- **Workbooks**: several sheets, tabbed like a spreadsheet. A `Subgraph` node on
  one sheet stands for another, and the referenced sheet is generated as its own
  module class — a class per sheet, as you would write it by hand. Circular
  references are refused with the circle named.
- Panels dock left, bottom or right, and everything resizes. The arrangement is
  saved on the server, so it belongs to the project rather than the browser.

### Understanding

- **The Maths panel** gives each layer's equation with *your* numbers
  substituted, not a textbook's. Not "1,792 parameters" but
  `(C_in/g)·k²·C_out + C_out = 3×9×64 + 64 = 1,792`, and
  `⌊(32 + 2 − 2 − 1)/2⌋ + 1 = 16` for the output size.
- Each operation family has an animated diagram — a convolution window walking
  its input, a distribution recentring, the actual curve of the activation you
  chose. 27 layers have full write-ups; the rest say they do not.
- **Where the freedom is**: what you can vary and what varying it does. For a
  dense layer it computes the rank below which factorising becomes cheaper *for
  your widths*. For attention it explains why the √d_k divisor exists.

### Importing

- **From GitHub**: type `karpathy/minGPT`, press Fetch. Downloaded, unpacked,
  scanned; nothing runs until you pick a class.
- **From an installed library**: point it at `transformers/models` and it finds
  691 classes. Filter them, hover to read the source, press **Try** to find out
  whether one imports before committing to it.
- **From pasted code**, a folder, a torchvision name, an `.onnx` file or a `.pt`.
- Imports check their own parameter count against the model's and report any
  difference.

### Training and experimenting

- 7 dataset kinds, 6 training recipes, live loss curves, checkpoints, and a run
  history that records the full design so any past run can be reopened on the
  canvas.
- **Studies** run experiments for you: hyperparameter sweeps, architecture
  search, and *try the review's fixes*. Variants that would not build are
  discarded before anything trains. Every trial is an ordinary run.
- **101 guided projects**, each a plan explaining why each layer, what you would
  use instead, and where it goes wrong. Build step by step beside the canvas, or
  bring the whole thing in at once, or tick individual steps.

### Two worked systems

- **MicroLean** — a proof kernel over equational logic and a transformer that
  learns which tactic to apply. On theorems never seen in training it proves
  119/120 of 1–3 step goals and 114/120 of up to 5 steps, every proof verified
  by the kernel rather than by the model.
- **MicroGo** — Go on a small board with the rules done properly (capture,
  suicide, ko, area scoring), AlphaZero-style search, and a policy-value
  network. Self-play at a runnable budget does **not** produce a strong player,
  and it says so: at 40 simulations the search's visit counts have entropy 2.96
  against a uniform 3.22, so the policy is being trained toward noise.

---

## What makes it unusual

Most tools in this space are one of three things: a diagram editor that draws
but does not run, a training framework with no picture, or a hosted service that
wants your model. This is a local application that does all three and is
unusually careful about the boundary between what it knows and what it is
guessing.

Concretely, the habits that are rare:

- **Claims are checked against the framework**, not asserted. When an imported
  GPT-2 block reported 7,087,872 parameters, that was confirmed against PyTorch;
  when a reconstruction was tested for *behaviour* rather than shape, it matched
  the original to 0.00e+00.
- **Refusals name the obstacle.** "Needs arguments" and "cannot be traced" are
  different problems and only one is yours to fix, so they are different
  messages.
- **Negative results are reported.** The Go player does not work and the
  changelog says why, with the measurement. A theorem-proving score of 120/120
  was thrown out on discovering 32% of the test set overlapped training, and
  replaced with a lower number from disjoint sets.
- **The mathematics is instantiated.** Reading a result tells you what is; seeing
  the substitution tells you what to reach for. That is the difference between
  reference material and a collaborator.

Its limits are equally specific. Attention modules from Hugging Face cannot be
imported, because their `forward` takes `**kwargs` and `torch.fx` cannot trace
through it. Mixtures of experts cannot be drawn honestly. Older GPT-2 attention
slices its mask by a traced length and is refused. All three are stated, with
the reason, at the point you hit them.

---

## A guide for a beginner

### Getting it running

```bash
cd deep-network-designer
python doctor.py                       # what your machine has, and what it unlocks
uvicorn main:app --reload --port 8770
```

Open `127.0.0.1:8770`. `doctor.py` tells you which interpreter it checked and
what each missing package would give you — worth reading once, because a package
installed for a different Python is invisible from inside a virtual environment
and that trips almost everyone.

### Your first half hour

**1. Open something that works.** *Open → MiniGPT*. Press **Fit** in the
toolbar. You are looking at a real language model: an embedding, transformer
blocks, a head.

**2. Read a layer.** Click any layer, then open the **Maths** tab. You get the
equation, what each symbol means for *that* layer, the arithmetic worked
through, and what you could change. Click a different layer; it follows you.

**3. See the code.** The **Code** tab holds the whole generated file. Selecting a
layer jumps to its line and says which class it landed in. This is a file you
could run outside the app — press Download and you have it.

**4. Build something yourself.** *Projects* in the sidebar, pick **Handwritten
digits — small CNN**, press **Build step by step**. You are taken to the canvas
with the plan beside it. Press *Add this layer* and watch the shapes resolve as
each one lands. Read the "instead of this" notes — they are where the reasoning
is.

**5. Check a layer against PyTorch.** Select a Conv2d, open the **Layer** tab,
press **Test layer**. It builds that layer alone and runs a tensor through it:

```
3×32×32 → 64×16×16
1,792 parameters · 1.2 ms · float32
shape agrees with the canvas
```

**6. Train it.** The **Train** tab, dataset *synthetic* to start, a few epochs.
Watch the loss curve. Then try a real dataset. If `doctor.py` showed **mps**, set
the device to that rather than auto.

**7. Ask for a review.** The **Assistant** tab, type `review`. It reads your
network and reports real problems — two dense layers with nothing between them,
a Flatten handing the next layer eight million weights. Then try an edit:
`add dropout after the last layer`. It happens on the canvas and Cmd+Z undoes it.

**8. Find out whether the advice was right.** *Studies* in the sidebar, pick
**Try the review's fixes**, run it. It trains each suggestion against your
unchanged network and ranks them. This is the part worth internalising: the
assistant has opinions, the study has measurements.

### When you get stuck

- **Red on a layer** — the message says what is wrong with that layer's shapes.
  The **Checks** entry in the sidebar collects them all.
- **An import refused** — read the reason. "Needs arguments" means type the
  config in the box on that row. "Cannot be traced" means that model cannot come
  in, and no amount of fiddling will change it.
- **Something looks wrong** — `python tests/test_designer.py` runs 112 checks in
  under a minute and will usually tell you if it is the app rather than you.

### Going further

- **Plug-in blocks**: *Blocks & recipes* in the sidebar. Each is a Python file
  defining a layer. Edit one, save, and it reloads into the palette without a
  restart.
- **Import a real model**: *Import → owner/name*, try `karpathy/minGPT`. Feed-
  forwards and norms import cleanly across most libraries; attention often does
  not, and it will tell you which.
- **The two worked systems**: `python prove.py` trains the theorem prover and
  reports proofs the kernel accepted. `python play.py` trains the Go player and
  reports games won. One of them works well and the other does not, which is a
  more useful pair of examples than two successes would be.

---

## Where things live

```
main.py            the server and its 78 routes
layers.py          layer definitions: shapes, code, parameter counts
graph.py           shape propagation and error reporting
codegen.py         PyTorch and Keras output
workbook.py        multi-sheet designs and cross-sheet references
importer.py        torch.fx, ONNX, folders, GitHub
mathbook.py        the mathematics of each layer, instantiated
assistant.py       the command grammar and the architecture review
agents.py          sweeps, architecture search, repair studies
train.py           the training loop, datasets, run history
microlean.py       the proof kernel and tactic space
microgo.py         the Go rules and PUCT search
auth.py            accounts and per-user workspaces
blocks/            plug-in layers, hot-reloaded
recipes/           training loops for GANs, diffusion, contrastive learning
projects/          the 101 guided builds
examples/          the designs every workspace starts with
tests/             112 checks
```

---

## A note on the tests

There are 112, and a disproportionate number exist because something broke in a
way the existing tests could not see. A few worth knowing about, because they
describe the failure modes this codebase actually has:

- *no two functions on the page share a name* — a duplicate definition silently
  replaced an earlier one, and every check passed because nothing was undefined.
- *every node on a workbook sheet actually draws* — a reference to a variable
  above its own declaration threw mid-render and left sixteen of eighteen layers
  invisible. The analysis, the code and the counts were all correct.
- *a split's branches keep their own piece* — `q, k, v = t.split(...)` gave all
  three branches the same tensor. Shapes right, parameter count right, network
  computing something else entirely.
- *workspace state cannot ride along in a release* — the file recording which
  examples a workspace had been given was shipping inside the release archive
  and telling every machine that unpacked it that examples it had never seen
  were already delivered.

The pattern in all four: the thing being checked was fine, and the thing not
being checked was wrong.
