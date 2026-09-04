# Deep Network Designer

A visual editor for neural network architectures. Drag layers onto a canvas,
wire them together, watch activation shapes resolve as you go, export PyTorch or
Keras, and train the result — all locally, in the browser, against a Python
backend on your own machine.

It is an open answer to MATLAB's Deep Network Designer, with a plug-in system and
code generation for two frameworks instead of one.

<!-- A screenshot of the canvas belongs here. Drop one in docs/ and link it. -->

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8770
```

Then open http://127.0.0.1:8770 and press **Open → MiniGPT** for the worked
example.

## Start here

**Projects** holds 101 guided builds. Pick one to read its plan, then bring it in
three ways: *Build step by step* puts the plan in a panel beside the canvas and
places one layer per press, explaining why that layer and what you would use
instead; *Add all layers* brings the whole thing in at once; or tick individual
steps and *Add selected* to lift just the part you want. Any of them can append
to what is already on the canvas instead of replacing it.

Not in the list? Describe what you want in the box and the catalogue is searched
for the closest starting points. That search is a keyword matcher over project
tags, not a language model, and it says so when nothing fits.

## What it does

**Shapes resolve live.** Every wire is annotated with the tensor it carries, and
its thickness scales with the log of the activation volume, so a thick wire early
on shows where the memory is going. A layer that cannot accept what it is handed
turns red and says why — `SelfAttention: 256 channels is not divisible by 7
heads`, rather than a stack trace at runtime.

**The code is the point.** Select any layer and the inspector shows the exact
constructor and forward line it produces. Those lines are recorded as codegen
emits them, not reconstructed for display, so the panel and the exported file
cannot drift apart. There is a test asserting it.

**It trains.** MNIST, Fashion-MNIST, CIFAR-10, a folder of your own images, a CSV
table, or a text corpus. Checkpoints save automatically with the design embedded,
so you can reopen the network that produced a set of weights.

**It is extensible in three directions.** Layers past the core set are files in
`blocks/`. Training loops are files in `recipes/`. Guided projects are files in
`projects/`. All three hot reload, and a broken one is the only thing that
breaks.

## What it is not

This designs and trains models. It is not an LLM application framework — no
retrieval, prompt management, tool calling, or API orchestration. If you want to
build an app on top of GPT-4 or Claude, use LangChain and leave this alone. The
two compose fine: train something here, serve the checkpoint, call it from there.

## What's in the box

- **47 layers.** Convolution, pooling, normalization, recurrent, embedding,
  attention, transformer, shape surgery, and merges with PyTorch broadcasting.
- **13 plug-in blocks.** ResidualBlock, SqueezeExcite, InceptionBlock, Backbone
  (nine pretrained torchvision networks), GraphConv, RidgeSolve, ODEBlock,
  FixedPoint, GPTStack, PolicyHead, ValueHead, plus two runtime blocks:
  MCTSSearch and TextGenerator.
- **PyTorch and Keras generation** from one graph, with the Keras file honestly
  flagging anything it cannot translate.
- **Model import** from torchvision by name, ONNX, or a saved `.pt`.
- **A chat panel** for talking to a language model you trained.

## The worked example

A design ships with the app. Open **MiniGPT**, go to Training, set the dataset to
*Text file*, pick `demo_corpus.txt`, and start.

Eight nodes — token Input, Embedding, PositionalEncoding, Dropout, GPTStack, a
Linear head, an Output set to `language_modeling`, and a TextGenerator attached
past the Output. 805,632 parameters, roughly a ten-thousandth of GPT-2 small. It
trains on CPU in about three minutes:

```
epoch  1  loss 1.860  perplexity 6.42  next-char acc 42.8%
epoch  3  loss 0.310  perplexity 1.36  next-char acc 90.0%
epoch 10  loss 0.120  perplexity 1.13  next-char acc 95.5%
```

After epoch 1 it produces shaped noise. By epoch 10, in the Chat tab:

```
> the loss is not going down
  Lower the learning rate and train for longer.

> what is the difference between Embedding and Dropout?
  A GPTStack and a GPTStack are used at different points in the network.
```

That second answer is the honest picture: at this scale it learns the shape of an
answer long before it learns which layer goes with which fact. Watching where it
breaks is most of what makes the demo worth running.

The corpus is 471 KB of synthetic support dialogue about this app, generated from
templates — deliberately repetitive, so a character model has structure to grab.

In the **Chat** tab, dialogue mode wraps what you type as `user: ... \nmodel:` and
stops the reply at the end of its turn; raw mode just continues your text, which
is more revealing of what the model actually learned. Temperature and top-k
matter far more at this scale than they would on a large model. To generate
outside the app, run the exported file — `build_runtime()` returns the sampler,
already pointed at the vocabulary.

## Releasing

```
./release.sh "what changed in one line"
```

Runs the tests, commits, tags from `version.py`, and pushes. It will not push if
the tests fail, because a tagged commit that does not pass is worse than no tag.

## Tests

```
python tests/test_designer.py
```

Eighty-eight checks, with the torch-dependent ones skipping themselves when it is
absent. They cover what would make the tool untrustworthy rather than merely
broken: that generated code runs, that predicted shapes match what PyTorch
produces, that the inspector text is byte-identical to the export, that the
causal mask in GPTStack actually masks, and that resnet18 reimports to the exact
parameter count.

Writing them found a real bug — `padding="same"` was accepted alongside a stride
above 1, which Keras allows and PyTorch refuses, so the canvas was approving a
network the generated code could not build.

## Using the canvas

Drag a layer from the palette. If a layer is selected when you drop the new one,
they connect automatically, which covers most of the work when building a stack
top to bottom. To wire by hand, drag from the dot at the bottom of one layer to
the dot at the top of another.

Shape tells you what a node is before you read it, following the flowchart
conventions the roles actually match: circles are terminators (Input, Output),
diamonds are decisions (merges, where paths join), parallelograms are the data
symbol (Flatten, Reshape, Permute — the same values read differently),
rectangles with side bars are predefined processes (any plug-in block), pills
are simple steps (activations), dashed hexagons are preparation (runtime
components outside `forward()`), and plain rectangles are ordinary layers.

To move the whole graph, use the hand tool, hold space, or drag with the middle
mouse button. It is a mode rather than the default because layers here can be
dragged individually.

Click the `+` on a node's outgoing port to add the next layer, or drag from it to
wire by hand. To put a layer *between* two that are already connected, hover the
wire and click the `+` at its midpoint. Either opens a picker beside the button
with tiles for the common layers, a search box, and the full set behind *All
layers*. It only offers layers that can go where you are putting them.

The toolbar switches the flow between top-to-bottom and left-to-right, and cycles
the grid between full, half, quarter and off.

Both panels can be docked left, bottom or right from the controls in their
headers, and every seam between them can be dragged to resize. The arrangement
is saved on the server, so it comes back with the project rather than with the
browser. It rewires both sides and pushes what follows out
of the way, which is the usual way a stack actually grows.

Each node shows the constructor it contributes to the generated file, so the
canvas and the Code panel say the same thing without switching between them.

**Test layer**, in the Layer tab, builds the selected layer alone and runs a
tensor through it, reporting the real output shape against the predicted one. The
canvas does shape arithmetic; this checks that arithmetic against PyTorch.
**Freeze this layer** parks its weights: the generated code stops their
gradients and the header count drops to what is actually being trained.

The **Maths** tab explains the selected layer: its equation, what each symbol is
in terms of this node, the shape and parameter arithmetic worked through with
the numbers substituted, and what can be varied along with what varying it does.
Each operation family has an animated diagram — a convolution window walking its
input, a distribution recentring, the actual curve of the activation you chose.
A layer whose mathematics is not written up says so rather than inventing
something.

The **Assistant** tab edits the graph and checks it over. It is rule-based, not
a language model: `add dropout after the activation`, `remove the flatten`, `set
units to 64 on linear_1`, `freeze conv2d_2`. Edits arrive as a graph, so they
appear on the canvas and undo normally. `review` reports what is actually wrong
— two Linear layers with nothing between them, a Flatten handing the next layer
eight million weights, a convolution with no normalization before its
activation. It says when it does not understand rather than guessing. Setting
`ASSISTANT_API_KEY` forwards unrecognised questions to a real model along with a
summary of the graph.

The right panel has seven tabs. **Layer** edits the selection. **Network**
summarises the design and its Inputs and Outputs. **Code** shows the generated
file. **Train** runs it. **Needs** answers what this
design requires before it will run: which plug-in blocks it pulls in and from
which file, whether it will download pretrained weights, which Python packages
it needs and why, which dataset kinds its Inputs can accept, and what will go
wrong — multi-input ordering, first-run downloads, layers with no Keras form.
All of it is derived from the graph, so it cannot disagree with what you built.

Shift-click extends the selection, `Cmd/Ctrl+A` takes everything, and shift-drag
on empty canvas sweeps a marquee. A group drags together, and copy-paste
preserves connections between the copied layers. `Cmd/Ctrl+Z` undoes.

The rail on the far left switches pages: Design is the canvas, and Build, Runs,
Chat and Extend take the full window. A status strip along the bottom of the
canvas is green when the graph resolves and red with the first problem when it
does not.

Large graphs get a minimap, `Cmd/Ctrl+K` to find a layer by name and centre on
it, a toggle between curved and right-angled wires, and snap-to-grid. Anything
past about seventy layers wraps into columns rather than running down in one
strip.

Keyboard: `Delete` removes the selection, `Cmd/Ctrl+D` duplicates it, `Escape`
deselects, scroll zooms, dragging empty canvas pans.

## Training on your own data

**Synthetic** generates random tensors matched to every Input. It learns nothing,
but it is the fastest way to confirm a graph runs.

**Built-in image sets** are resized to match your Input layer, converting between
grayscale and RGB as needed.

**A folder of images** wants one subfolder per class. Scan reports what it found
and flags a mismatch against your last Linear layer. Six augmentations, applied
to the training split only.

**A CSV table** hands its columns to your Input layers in order — six columns to
a `[6]` Input, twelve to a `[3, 4]` Input, reshaped to fit. Override per Input by
naming columns. Text columns become integer codes, missing values become the
column mean, and standardization uses training-split statistics only.

**A text corpus** becomes character-level next-token pairs, with the vocabulary
written alongside for the generator.

## Several inputs, several outputs

A graph with more than one Input trains normally. `forward()` takes its arguments
in topological order and the loader feeds them in that same order, so with a CSV
each Input draws from its own columns, and with an image set every Input receives
the same batch — which is what siamese graphs want.

More than one Output also works: the first carries full weight and the rest are
scaled by the extra-head weight in the form, defaulting to 0.3. Every head sees
the same target, so this covers auxiliary classifiers rather than genuinely
independent labels.

## Transfer learning

Drop a `Backbone` on the canvas, follow it with `GlobalAvgPool` and a `Linear`
sized to your classes. The block loads a pretrained torchvision network with its
classifier removed, so what comes out is a feature map rather than ImageNet
logits.

Freezing is a count of trailing stages left trainable. Zero trains only your
head; raise it once the head has settled. A non-RGB Input is handled by averaging
the pretrained stem's kernels across the colour axis.

## Saved designs

Save writes a new version each time rather than overwriting. The selector beside
the design name lists every version with its timestamp, and switching loads that
one onto the canvas. Deleting takes either one version or the whole history.

## Accounts

Accounts are optional. With none registered the app runs exactly as it always
has. *Set up an account*, in the header, turns authentication on; the first
account inherits the work already there, and every account after it gets its own
designs, runs, studies and panel layout. Passwords are hashed with scrypt and a
per-account salt, and changing one signs out every other session.

Every account starts with the example designs — they ship in `examples/` and are
copied in per example, so a design added in a later version reaches workspaces
that already exist. Deleting one sticks; opening a design in an
empty workspace offers to put them back.

Signing out locks the page rather than emptying it — the work stays exactly
where it was, under `saved/`, `runs/` and `studies/`. If you cannot get back in,
`accounts.py` handles it from a terminal:

    python accounts.py list
    python accounts.py passwd <name>     forgotten password
    python accounts.py remove <name>     keeps their workspace
    python accounts.py off               removes every account, keeps every workspace

Accounts separate people from each other, **not from the machine**. Importing
code, importing a folder, and the blocks and recipes folders all execute Python
by design, so anyone who can sign in can run code as this process. Run it on a
network you trust.

## Studies

**Studies** run experiments so you do not have to. A hyperparameter sweep trains
the same network across learning rates, batch sizes and optimizers. An
architecture search proposes wider, narrower and more regularized versions and
ranks them — leaving the final head alone, since the number of classes is not a
hyperparameter. *Try the review's fixes* takes what the assistant's `review`
found, applies each fix on its own, and trains them beside the unchanged network
so you learn which actually helped.

Variants that would not build are discarded during planning. Every trial is an
ordinary run — it appears in the run history and writes checkpoints — and any
variant can be opened on the canvas from the leaderboard.

This is deliberately not "call a language model from inside your network". Those
tasks belong in a workflow orchestrator; they are not differentiable and have no
tensor shape. Running the experiments is the automation a network designer
actually needs.

## Runs

Every training run is recorded to `runs/` as it happens, pinned to the design
name and version that produced it. The Runs tab lists them newest first with
status, epochs, training loop, dataset, best objective and duration; opening one
shows its curve, its full configuration and its checkpoints.

**Open this design** on any run restores the exact graph that run used. A result
from last week can be reproduced rather than reconstructed from memory, which is
the whole reason the record keeps a copy of the design rather than a reference
to it — the design can change, the record should not.

Failed runs are recorded too, with the reason.

## Saved weights

Each run writes its best and final epoch to `checkpoints/`, carrying the weights,
the metrics, and the design that produced them.

**Start from checkpoint** copies every tensor whose name and shape match. What
does not match is reported and left freshly initialized — exactly what you want
when keeping a trunk and resizing the head for a different class count.

## Importing an existing model

The Import button takes pasted PyTorch source, a torchvision architecture by
name, an `.onnx` export, or a `.pt` holding the module itself.

After a scan the folder stays browsable under **Files** in the left panel:
directories, files, and a chip for every model found in each. Clicking a
filename reads its source into the code panel — reading never runs anything —
and clicking a model chip imports that class onto a new sheet.

For a whole project, *Scan a folder of Python files* lists every `nn.Module` it
finds — without running any of it — and each class you pick becomes its own
sheet. Sheets are tabbed at the bottom of the canvas like a spreadsheet: click
to switch, double-click to rename, right-click to choose which sheet is the
model. A `Subgraph` node lets one sheet stand on another — it reads *↪ continues
on "stem"*, double-click follows it — and the referenced sheet is generated as
its own class that the parent instantiates, exactly as the code would be written
by hand. Shapes flow through the reference; circular references are refused with
the circle named.

`GPT2` is one of the example designs: the transformer block on one sheet and the
model on another, 163,037,184 parameters, generating code that runs.

Transformer code imports well: a GPT-2 block written in the usual style arrives
as 21 layers with nothing opaque, shapes resolved and the parameter count exact.
Shape bookkeeping (`B, T, C = x.size()`) is elided rather than drawn, and
reshape targets are measured by running one example through rather than
reconstructed from the traced arithmetic.

What cannot be imported is a `forward` that slices by a traced value — older
GPT-2 attention masks do this — because `torch.fx` cannot trace it into a single
graph. The import says so and names the reason; the same model written with
`F.scaled_dot_product_attention` imports without trouble.

Pasting code is the quickest route for a single model: a class subclassing `nn.Module`, or a
variable holding one such as `model = nn.Sequential(...)`. It is traced by
running it, which is the only way to learn what `forward` does, so it has to be
code you trust — the same assumption the blocks and recipes folders already
make.

PyTorch import traces with `torch.fx`, which keeps layers whole — a traced
`Conv2d` arrives with its real arguments. Eleven of sixteen torchvision
architectures tested rebuild with parameter counts identical to the original,
including every ResNet, both VGGs, both MobileNets, DenseNet121, SqueezeNet,
AlexNet and RegNet.

Tracing has a hard limit: a model whose `forward` branches on tensor values has
no single graph to read, and the import says so. ConvNeXt, ShuffleNet and
Inception import structurally but land with shape problems flagged.

An operation with no equivalent in the registry becomes a stub node that keeps
the original call in its values and appears in the import summary. Emitting a
multi-line module repr inline would produce a file that does not parse, and
substituting something plausible would produce a file that parses and is wrong.

## Blocks: the plug-in system

Everything past the core layers is a file in `blocks/`. Drop one in, press Reload
in the Blocks tab, and it appears — no restart, no edits to the core. A block
that fails to import shows its traceback and is the only thing that breaks.

Each block carries a `prelude`, which is real class source copied verbatim into
the generated file. A block is code you can read and change, not a node with
hidden behaviour.

### Two kinds

**`kind="layer"`** is differentiable. It lives in the graph, declares how it
reshapes an activation, and its code lands inside `forward()`.

**`kind="runtime"`** is everything that is not a tensor transform — tree search,
self-play, sampling. It attaches to the graph so the design shows it exists, but
generates a separate `build_runtime(model)` function and has no output shape.

That distinction is why MCTS is not a layer. AlphaGo is a network *plus* a search
*plus* a self-play loop, and only the first is a tensor graph.

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
that sentence is what appears on the red node. The New button in the Blocks tab
writes this skeleton for you.

## Adding a core layer type

For something that belongs alongside `Conv2d` rather than in `blocks/`, one
`LayerSpec` in `layers.py` defines everything — the parameters shown in the
inspector, the shape rule, and the code emitted for each framework. The palette
and inspector build themselves from that registry, so nothing in the frontend
needs touching:

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

For one-off layers there is also the `Custom` node in the palette, which takes a
shape rule and a code snippet per framework without a server restart. Anything in
braces is evaluated with `shape`, `shapes` and `p` in scope, so
`MyLayer(dim={p['dim']})` picks up whatever you put in its values field.

## Recipes: pluggable training loops

`blocks/` makes the layer set extensible. `recipes/` does the same for the
training loop, which is what was actually blocking the harder workflows. GANs,
diffusion, contrastive pretraining and reinforcement learning all build fine on
the canvas — what stopped them was a loop that assumed one model, one optimizer,
and a loss computed from predictions and labels.

A recipe owns its own backward pass and optimizer steps, so it can run two
optimizers, rewrite the forward pass, construct its own batches, or roll out an
environment. It reports a dictionary of numbers and the trainer charts them.

Three ship:

| Recipe | What it proves |
| --- | --- |
| `Autoencoder` | A target that is not a label — reconstruction, with optional denoising |
| `Contrastive` | A recipe constructing its own batch: two augmented views, NT-Xent loss, no labels at all |
| `Diffusion` | A recipe rewriting the forward pass: noise schedule, timestep conditioning, DDIM sampling |
| `GAN` | Two networks and two optimizers, alternating. The discriminator is a second saved design |
| `Reinforce` | A loop with no dataset at all: the policy acts, CartPole answers, the episode is the batch |
| `Detection` | Variable-length targets assigned to grid cells before any loss can be computed |

Pick one from the top of the Training tab and its settings appear below. Recipes
can refuse a graph they cannot train — the autoencoder names the shapes that
disagree, diffusion tells you exactly what to set the Input to.

### Writing one

```python
from recipes_sdk import Param, Recipe, install


def setup(ctx):
    import torch
    ctx.optimizers["main"] = torch.optim.AdamW(ctx.parameters(), lr=ctx.cfg["lr"])


def step(ctx, xs, y):
    import torch.nn.functional as F
    loss = F.mse_loss(ctx.model(*xs), xs[0])
    opt = ctx.optimizers["main"]
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return {"loss": float(loss.item())}


install(Recipe(name="MyLoop", doc="...", params=[Param("lr", "float", 1e-3)],
               setup=setup, step=step))
```

`ctx` holds the models, the device, your settings and a scratch `state` dict.
Return any numbers you want charted. Add a `check(ctx)` returning a complaint
string to reject graphs that will not work.

Three further hooks cover the harder cases. `extra_models=["discriminator"]`
asks for another network, built from a saved design the user picks.
`data_shape(ctx)` tells the loader what the data looks like when it differs from
the model's Input, which a GAN needs because it takes noise but reads images.
`self_supplied=True` skips the DataLoader entirely and calls `step` a fixed
number of times per epoch, which is the only way to express a reinforcement
learning rollout.

Set `objective` to the metric that matters — `Reinforce` uses `return` with
`lower_is_better=False`, so checkpoints keep the best policy rather than the
lowest loss.

## Shape convention

The graph is channels-first with the batch dimension left out: `[C, H, W]` for
images, `[L, C]` for sequences, `[F]` for vectors, `[L]` for token ids. The Keras
generator converts to channels-last at the Input and lets Keras infer the rest.
`Reshape` and `Permute` are where that difference is visible, and the generated
Keras file flags them.

## Layout

The server finds `index.html` whether `main.py` sits in a `backend/` folder
beside `frontend/`, or flat in the project root next to it.

```
main.py          FastAPI routes
layers.py        core layer registry: parameters, shape rules, code templates
graph.py         topological order, shape propagation, validation
codegen.py       IR to PyTorch and Keras source
train.py         background training jobs, datasets, checkpoints, sampling
importer.py      torch.fx and ONNX import onto the canvas
blocks_sdk.py    the Block/Param surface plug-ins write against
blockloader.py   scans blocks/, hot-reloads, isolates failures
version.py       release number, read by the header and /health
blocks/          plug-in blocks, one file each
recipes/         plug-in training loops, one file each
projects/        the guided project catalogue
projects_sdk.py  the Project/Step surface the catalogue is written against
projectloader.py loads projects and matches free-text requests against them
recipes_sdk.py   the Recipe/Context surface training loops write against
recipeloader.py  scans recipes/, hot-reloads, isolates failures
checkpoints/     saved weights, each carrying the design that made it
runs/            one record per training run, with its design and metrics
studies/         one record per experiment study, with its leaderboard
saved/           designs saved from the Open/Save buttons
uploads/         tables and corpora
tests/           the test suite
frontend/
  index.html     the whole designer
```

## Known limits

Multiple Outputs train against the same target, which suits auxiliary heads but
not independent multi-task labels. Table columns are read as flat numeric
features, so there is no windowing for time series yet. Parameter totals
including a pretrained backbone are marked with a tilde, since that figure is a
published lookup rather than derived from the graph. Custom shape rules and code
templates are evaluated with a restricted `eval`, which is fine for a tool you
run on your own machine and not something to expose publicly.

## Licence

MIT. See `LICENSE`. Version 1.24.1 — see `CHANGELOG.md`.
