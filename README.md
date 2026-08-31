# Deep Network Designer

Version 1.0.0 — see CHANGELOG.md. MIT licensed.

Drag layers onto a canvas, wire them together, watch activation shapes resolve
live, export PyTorch or Keras, and train the result on a local Python backend.

## Tests

```
python tests/test_designer.py
```

Seventeen checks. The ones that matter most confirm that generated code runs,
that shapes predicted on the canvas match what PyTorch produces, and that the
code shown in the inspector is the same text as the exported file. A designer
whose predictions disagree with the framework is worse than no designer.

## Run it

Create a virtual environment and install the dependencies:

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Start the server:

```
uvicorn main:app --reload --port 8770
```

Open http://127.0.0.1:8770

Torch and torchvision are only needed for the Training tab. Everything else —
the canvas, shape checking, code export — runs without them.

## Using it

Drag a layer from the left rail onto the canvas. If a layer is selected when you
drop the new one, they connect automatically, which covers most of the work when
you're building a stack top to bottom. To wire things by hand, drag from the dot
at the bottom of one layer to the dot at the top of another.

Every wire is annotated with the tensor it carries, and its thickness scales with
the log of the activation volume — a thick wire early on is usually where your
memory is going. Layers that can't accept what they're handed turn red, and the
Problems tab spells out why.

The Code panel regenerates on every edit. What you see there is exactly what the
Training tab compiles and runs, so there's no gap between the exported file and
the trained model.

Select a layer and the inspector shows the code it generates — the constructor
line and the forward line, taken from the generated file rather than rebuilt, so
the two cannot disagree. Layers that ship a class definition can expand it there.
Blocks get an Edit button that opens their file in the Blocks tab; core layers
get Show in Code, which jumps to the generated file and highlights the line.

Large graphs need the navigation controls at the top right of the canvas: a
minimap with the current viewport on it, Cmd/Ctrl+K to find a layer by name and
centre on it, a toggle between curved and right-angled wires, and snap-to-grid.
Shift-drag on empty canvas sweeps a marquee selection.

Anything past about seventy layers wraps into columns rather than running down
in one strip, which is the difference between a graph you can read and a canvas
46,000 pixels tall.

Shift-click extends the selection and `Cmd/Ctrl+A` takes everything. A group
drags together, and copy-paste preserves connections between the copied layers,
rewiring them to the new nodes. `Cmd/Ctrl+Z` undoes and `Cmd/Ctrl+Shift+Z` redoes.

Keyboard: `Delete` removes the selection, `Cmd/Ctrl+D` duplicates it, `Escape`
deselects, scroll zooms, and dragging empty canvas pans.

## Training on your own data

The Training tab takes three kinds of data.

**Synthetic** generates random tensors matched to every Input layer. It won't
learn anything, but it's the fastest way to confirm a graph runs and to see how
long an epoch takes.

**Built-in image sets** (MNIST, Fashion-MNIST, CIFAR-10) are resized to match
your Input layer, converting between grayscale and RGB as needed.

**A folder of images** wants one subfolder per class. Scan reports what it found
and flags a mismatch against your last Linear layer. Augmentations apply to the
training split only.

**Uploaded tables** are the interesting case. Upload a CSV, pick the target
column, and the remaining columns are handed to your Input layers in order — six
columns to a `[6]` Input, twelve to a `[3, 4]` Input, reshaped to fit. You can
override the split per Input by typing column names. Text columns become integer
codes, missing values become the column mean, and features are standardized using
statistics from the training split only. A text target is factorized into
classes, and if the class count doesn't match your last layer's output units, the
run says so in the log instead of quietly training a model that can't be right.

## Worked example: a small GPT

A design ships with the app. Press **Open**, choose `MiniGPT`, then go to the
Training tab, set the dataset to *Text file*, pick `demo_corpus.txt`, and start.

It is eight nodes: token Input, Embedding, PositionalEncoding, Dropout,
`GPTStack`, a Linear head back to vocabulary size, an Output set to
`language_modeling`, and a `TextGenerator` attached past the Output. That comes
to 805,632 parameters — about a ten-thousandth of GPT-2 small — and it trains on
CPU in a few minutes.

The corpus is 471 KB of synthetic support dialogue about this app, generated
from templates. It is deliberately repetitive: a character model needs visible
structure to latch onto, and this makes progress obvious within one epoch.

After epoch 1 the model produces shaped noise. By epoch 3 it has the turn
structure and most words. By epoch 10, at a validation perplexity of 1.13, it
answers in the form it was taught:

```
user: how do I add a GPTStack to my network?
model: Every GPTStack you ad shows up in the generated file straight away.
```

It is a character model with under a million parameters, so it recombines
learned phrasing rather than understanding anything. That is the honest ceiling
at this scale, and watching where it breaks is most of what makes the demo
worth running.

### Talking to it

The **Chat** tab is where you prompt it. Pick a checkpoint, type, press Enter.
Dialogue mode wraps what you type as `user: ... \nmodel:` and stops the reply at
the end of its turn; raw mode just continues your text, which is the better way
to see what the model has actually learned.

Temperature controls how surprising the sampling is — low is repetitive, high
wanders. Top-k caps how many candidates each character is drawn from. Both
matter more at this scale than they would on a large model.

The checkpoint is rebuilt from the design stored inside it and held in memory
after the first message, so replies take a fraction of a second.

To generate from a trained checkpoint outside the app, run the exported file — `build_runtime()`
returns the sampler, already pointed at the vocabulary.

## Importing an existing model

The Import button brings a model in three ways: a torchvision architecture by
name, an `.onnx` export, or a `.pt` holding the module itself.

PyTorch import traces with `torch.fx`, which keeps layers whole — a traced
`Conv2d` arrives with its real arguments, so what lands on the canvas is what you
would have built by hand. Eleven of the sixteen torchvision architectures tested
rebuild with parameter counts identical to the original, including every ResNet,
both VGGs, both MobileNets, DenseNet121, SqueezeNet, AlexNet and RegNet.

Tracing has a hard limit. A model whose `forward` branches on tensor values has
no single graph to read, and the import says so. Architectures with unusual
plumbing — ConvNeXt, ShuffleNet, Inception — import structurally but land with
shape problems flagged in the Problems tab; the graph is there to work with, it
just is not finished.

An operation the registry has no form for becomes a stub node that keeps the
original call in its values and appears in the import summary. Emitting a
multi-line module repr inline would produce a file that does not parse, and
substituting something plausible would produce a file that parses and is wrong,
so a flagged stub is the honest third option.

Tracing carries no shapes, which is why the dialog asks for an input shape. It
seeds the analysis; everything downstream is inferred.

ONNX import is lossier by nature. The exporter folds BatchNorm into the
preceding convolution, so a designer model exported to ONNX and re-imported
comes back structurally identical but without its normalization layers.

## Transfer learning

Drop a `Backbone` on the canvas, follow it with `GlobalAvgPool` and a `Linear`
sized to your classes, and you have the standard transfer-learning setup. The
block loads a pretrained torchvision network with its classifier removed, so
what comes out is a feature map rather than ImageNet logits.

Freezing is a count of trailing stages left trainable. Zero trains only your
head; raise it once the head has settled to fine-tune deeper. A non-RGB Input is
handled by averaging the pretrained stem's kernels across the colour axis, which
starts far better calibrated than a fresh stem.

Weights download on first use, so the first run needs a network connection.

## Saved weights

Each run writes its best epoch and its final epoch to `checkpoints/`. A
checkpoint holds the weights, the metrics, and the design that produced them, so
you can reopen that design on the canvas from the Training tab.

Start from checkpoint copies every tensor whose name and shape match. Anything
that does not match is reported and left freshly initialized — which is exactly
what you want when you keep a trunk and resize the head for a different number
of classes.

## Several inputs, several outputs

A graph with more than one Input trains normally. `forward()` takes its arguments
in topological order and the loader feeds them in that same order, so with a CSV
each Input draws from its own columns, and with an image set every Input receives
the same batch — which is what siamese graphs want.

More than one Output also works: the first carries full weight and the rest are
scaled by the extra-head weight in the form, defaulting to 0.3. Every head sees
the same target, so this covers auxiliary classifiers rather than genuinely
independent labels.

## Shape convention

The graph is channels-first with the batch dimension left out: `[C, H, W]` for
images, `[L, C]` for sequences, `[F]` for vectors, `[L]` for token ids. The Keras
generator converts to channels-last at the Input layer and lets Keras infer the
rest. `Reshape` and `Permute` are the two nodes where that difference is visible,
and the generated Keras file flags them with a comment.

## Blocks: the plug-in system

Everything in the palette below the core layers comes from a file in `blocks/`.
Drop a file in, press Reload in the Blocks tab, and it appears — no restart, no
edits to the core. You can write blocks in your editor or in the browser: the
Blocks tab lists every file, opens its source, and saves with Cmd/Ctrl+S. A
block that fails to import shows its traceback in the tab and is the only thing
that breaks; every other block stays loaded.

Each block carries a `prelude`, which is real class source copied verbatim into
the generated file. That is the whole idea — a block is code you can read and
change, not a node with hidden behaviour.

### Two kinds of block

**`kind="layer"`** is differentiable. It lives in the graph, declares how it
reshapes an activation, and its code lands inside `forward()`.

**`kind="runtime"`** is everything that is not a tensor transform — tree search,
self-play, solvers that need a loop around the model. It attaches to the graph so
the design shows it exists, but it generates a separate `build_runtime(model)`
function rather than joining `forward()`, and it has no output shape because it
produces no activation. Runtime blocks are the only nodes allowed to read from an
Output; anything else trying to is flagged.

This distinction is why MCTS is not a layer. AlphaGo is a network *plus* a search
*plus* a self-play loop, and only the first is a tensor graph.

### What ships in the box

| Block | Kind | Notes |
| --- | --- | --- |
| `ResidualBlock` | layer | Projects the skip path only when the shape changes |
| `SqueezeExcite` | layer | Channel attention |
| `InceptionBlock` | layer | Four branches concatenated |
| `PolicyHead` | layer | Move logits for a board game |
| `ValueHead` | layer | Position score in [-1, 1] |
| `MCTSSearch` | runtime | PUCT search plus a self-play loop |
| `GraphConv` | layer | Node features and adjacency in, GCN propagation |
| `RidgeSolve` | layer | Differentiable linear solve |
| `ODEBlock` | layer | Continuous depth via RK4 |
| `FixedPoint` | layer | Iterates one transform to convergence |

`MCTSSearch` generates a working PUCT implementation and a `self_play_game`
helper. What it cannot generate is your game: the file defines an `Environment`
with five methods (`legal_actions`, `step`, `is_terminal`, `reward`, `encode`)
and leaves them for you. Fill those in and the search runs.

Blocks that only have PyTorch code are honest about it. The Keras file starts
with a header naming every node it could not translate and marks each one inline,
rather than quietly emitting something that is not your model.

### Writing one

```python
from blocks_sdk import Block, Param, ShapeError, install, need_rank

PRELUDE = """
class GatedMix(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Linear(dim, dim)

    def forward(self, x):
        return x * torch.sigmoid(self.gate(x))
"""

install(Block(
    name="GatedMix",
    category="Custom blocks",
    doc="Sigmoid gate over the last dimension.",
    params=[Param("scale", "float", 1.0, help="Shown in the inspector")],
    infer=lambda p, shapes: list(shapes[0]),
    learnables=lambda p, ins, out: ins[0][-1] ** 2 + ins[0][-1],
    prelude=PRELUDE,
    torch_init=lambda p, ins: f"GatedMix({ins[0][-1]})",
))
```

`infer` raises `ShapeError` with a plain sentence when the input will not do, and
that sentence is what shows up on the red node. `learnables` is optional but
worth writing — it keeps the parameter count in the header honest.

The New button in the Blocks tab writes this skeleton for you.

## Adding a core layer type

For something that belongs alongside `Conv2d` rather than in `blocks/`, one
`LayerSpec` in `backend/layers.py` defines everything — the parameters shown
in the inspector, the shape rule, and the code emitted for each framework. The
palette and the inspector build themselves from that registry, so nothing in the
frontend needs touching:

```python
register(LayerSpec(
    name="MyLayer",
    category="Dense",
    params=[P("units", "int", 64, min=1)],
    infer=lambda p, ins: ins[0][:-1] + [int(p["units"])],
    torch_init=lambda p, ins: f"MyLayer({ins[0][-1]}, {int(p['units'])})",
    torch_call=lambda p, ins, mod, s: f"{mod}({ins[0]})",
    keras_call=lambda p, ins, s: f"MyLayer({int(p['units'])})({ins[0]})",
))
```

For one-off layers there's also the `Custom` node in the palette, which takes a
shape rule and a code snippet per framework without a server restart. Anything in
braces is evaluated with `shape`, `shapes` and `p` in scope, so
`MyLayer(dim={p['dim']})` picks up whatever you put in its values field.

## Layout

The server finds `index.html` whether `main.py` sits in a `backend/` folder
beside `frontend/`, or flat in the project root next to it. If it cannot find the
page it says so on the homepage and lists where it looked, instead of throwing a
500.

```
main.py          FastAPI routes
layers.py        core layer registry: parameters, shape rules, code templates
graph.py         topological order, shape propagation, validation
codegen.py       IR to PyTorch and Keras source
train.py         background training jobs, metrics queue
blocks_sdk.py    the Block/Param surface plug-ins write against
blockloader.py   scans blocks/, hot-reloads, isolates failures
version.py       release number, read by the header and /health
importer.py      torch.fx and ONNX import onto the canvas
blocks/          plug-in blocks, one file each
checkpoints/     saved weights, each carrying the design that made it
saved/           designs saved from the Open/Save buttons
uploads/         CSV tables uploaded from the Training tab
frontend/
  index.html     the whole designer
```

## Known limits

Multiple Outputs all train against the same target column, which suits auxiliary
heads but not independent multi-task labels. Table columns are read as flat
numeric features, so there's no windowing for time series yet — a `[L, C]` Input
takes `L × C` columns and reshapes them rather than sliding over rows. Learnable
counts on the canvas are analytic estimates; they match PyTorch exactly for every
layer tested, but a custom node reports zero until you run the file. Custom shape
rules and code templates are evaluated with a restricted `eval`, which is fine
for a tool you run on your own machine and not something to expose publicly.
