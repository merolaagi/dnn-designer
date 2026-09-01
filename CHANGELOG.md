# Changelog

## 1.10.0

- **Panels dock where you want them.** Each panel header carries three controls
  for left, bottom and right. Put the palette along the bottom and give the
  canvas the full width, or stack both panels on one side. Two panels docked to
  the bottom share that row.
- **Everything is resizable.** Drag the seam between a panel and the canvas, or
  the seam above the bottom row. Panels clamp between 150 and 760 pixels so a
  drag cannot lose one off the edge.
- **The arrangement is saved on the server**, not in the browser, so it belongs
  to the project rather than to the machine that opened it. A corrupt
  preferences file reads as empty rather than breaking the app.
- The right panel header names the tab you are looking at.

## 1.9.0

The canvas gets a vocabulary of shapes, and a direction.

- **Shape carries meaning.** Input and Output are circles — entry and exit
  terminals. Merges (Add, Concat, Multiply) are diamonds on a grey pad, because
  they are where paths join. Runtime blocks are dashed hexagons, since they sit
  outside `forward()`. Ordinary layers stay cards. Previously everything was the
  same rectangle, which meant the drawing carried no information the labels did
  not already.
- **A `+` on every node's outgoing port.** Click it to add the next layer, drag
  it to wire by hand. Combined with the `+` already on each wire, a stack can be
  grown without touching the palette.
- **Horizontal flow.** A toolbar toggle lays the graph left to right instead of
  top to bottom: ports move to the sides, wires curve horizontally, and Tidy
  arranges along the new axis. Deep graphs wrap into bands either way.
- **A ruled grid** replaces the dot field, at two levels — fine every 26px,
  heavy every 130px. A toolbar button cycles it through full, half, quarter and
  off.
- **Arrowheads** on every wire, so direction reads without tracing the curve.
- Bounds, minimap, marquee selection, guided steps and insertion all measure
  both dimensions now, since nodes are no longer one size.

## 1.8.1

Fixes a stylesheet I broke in 1.8.0, and takes the chrome closer to the workflow
editor it is modelled on.

- **Fixed: the page was unstyled below a point.** Tidying dead CSS in 1.8.0, I
  filtered by "lines starting with X", which kept the opening line of multi-line
  rules and threw away their bodies and closing braces. One unbalanced brace
  makes a browser discard every rule after it, so the Import dialog rendered
  inline as a dark slab and the canvas disappeared. Three tests now guard this:
  balanced braces, no empty rules, and no script reference to an element nothing
  creates.
- **Header is two rows**: breadcrumb, name and actions on top; a canvas toolbar
  beneath with zoom readout, fit, tidy, routing, snap, minimap, undo and find.
  The floating zoom buttons over the canvas are gone.
- **The design name is the page title**, editable in place, with the version
  selector beside it.
- **Sidebar is named rather than iconic**, grouped into Definitions, Executions
  and Status — the sections are what the app does, so they should be readable.
- Header chrome is light to match the canvas, with the primary action in blue.

## 1.8.0

The layout had two competing panel systems — a bottom drawer *and* a right
panel. The drawer opened over the canvas, and on a laptop the Build project list
was clipped off the bottom of the window with its button half out of view. That
was a layout bug, not a matter of taste.

- **The bottom drawer is gone.** One full-height right panel does the work, the
  way a workflow editor arranges it.
- **Right panel widened to 390px** and gained tabs: Layer, Network, Code, Train,
  Needs. Code and training now live beside the canvas instead of underneath it,
  so you can watch shapes resolve while reading the file they generate.
- **The rail switches pages, not drawers.** Design shows the canvas; Build,
  Runs, Chat and Extend take the full width, which is what the project browser
  and the run history actually needed.
- **Status strip** along the bottom of the canvas: green when the graph
  resolves, red with the first problem when it does not, and a details toggle
  for the rest. Always visible, never covering anything.
- The training panel stacks vertically to suit a column rather than a wide
  drawer.

## 1.7.0

Definitions and executions, separated the way a workflow tool separates them.

Designs were already versioned, but training runs were in-memory only: lost on
restart, tied to nothing, with no history. A run is now a record.

- **Every training run is written to `runs/`** as it happens, pinned to the
  design name and version that produced it, and carrying a full copy of that
  design.
- **Runs tab** on the rail: every execution, newest first, with status, epochs,
  training loop, dataset, best objective and duration. Click one for its loss
  curve, its full configuration, its checkpoints, and anything reported during
  the run.
- **Open this design** on any run restores the exact graph it used, so a result
  from last week can be reproduced rather than reconstructed from memory.
- Runs that fail are recorded too, with the reason. A run that could not start
  because the dataset was not configured is more useful in the history than
  absent from it.
- Records survive a restart, which is the point.
- Fixed while testing: the record was written *after* the event announcing it,
  so a listener could be told a run had finished and then read a file still
  saying it was running. The write now happens first.
- Two more tests, 42 in total.

## 1.6.0

- **Right panel gains tabs**: Layer, Network, Needs — with collapsible sections,
  in the manner of a workflow editor's side rail.
- **Needs answers "what does this design require to run?"** and derives every
  answer from the graph rather than from a declaration, so it cannot go stale.
  It lists the plug-in blocks pulled in and which file each came from, runtime
  components, pretrained weights and whether they download, Python packages and
  why each is needed, which dataset kinds the Inputs can actually accept, the
  training loop this shape wants, and warnings — multi-input argument ordering,
  first-run downloads, and which layers will leave the Keras export incomplete.
- **Network** summarises the design: name, version, layer and connection counts,
  learnables, and every Input and Output with its shape and task.
- Selecting a layer brings the Layer tab forward, so clicking the canvas always
  goes somewhere useful.
- Four more tests, 40 in total.

## 1.5.0

The canvas reworked, taking the parts of a workflow editor that carry their
weight here.

- **Light canvas.** The dark blueprint suited a diagram; it fights a graph you
  read code off. Chrome stays dark, the board is white with a fine dot grid.
- **Nodes carry their code.** Each card now shows a type badge, a reference
  line, and the actual constructor it contributes to the generated file —
  `nn.Conv2d(3, 32, kernel_size=3, padding='same')` on the node itself, rather
  than only in the inspector.
- **Insert into a connection.** Hovering a wire reveals a `+` at its midpoint;
  clicking it opens a picker and drops the chosen layer into that connection,
  rewiring both sides and pushing everything below out of the way. Adding a
  normalization between an existing convolution and its activation is one click
  instead of drop-then-rewire.
- The picker offers only layers that can sit mid-chain — no Inputs, Outputs or
  runtime blocks.
- **Delete on the node.** A hover cross in the corner, rather than select-then-Delete.
- Merge nodes report how many inputs they are actually joining.
- Every layout calculation — fit, minimap, marquee, tidy, guided steps — now
  measures real card height rather than assuming one size.

## 1.4.1

- Fixed: the versioning test used Starlette's `TestClient`, which needs an HTTP
  client library the project does not otherwise depend on, so the suite failed
  in a plain virtual environment. It calls the route functions directly now.
  The suite is meant to run in a bare checkout and pulling in a dependency for
  a filesystem test defeated that.

## 1.4.0

- **Designs are versioned.** Save writes the next version instead of
  overwriting, and a selector beside the name opens any earlier one. A design
  you liked three edits ago is still there. Flat saves from before this are read
  as version 1 and left alone.
- **Navigation rail** down the left, replacing six tabs competing for width in
  the drawer. Design keeps the drawer collapsed so the canvas dominates;
  everything else opens it.
- **Warning count always visible** on the rail, green at zero, rather than only
  when the Problems tab is open.
- Delete removes a single version or the whole history.

## 1.3.0

A guided project catalogue: pick something to build, and it builds a layer at a
time with the reasoning attached.

- **Build tab** with **101 projects** across 15 categories — vision, medical
  imaging, tabular, sequences, language, generative, self-supervised, agents,
  detection, graphs, numerical, audio, anomaly detection, similarity,
  multi-input.
- Each step places one or more layers and says **why that layer**, **what you
  would use instead**, and what to watch for. The reasoning quotes real numbers:
  "this turns [64, 8, 8] into [4,096], which is why the next Linear is wide".
  That is why projects are generated from builders rather than written by hand.
- **Describe what you want** and the catalogue is searched for it. A keyword
  matcher, not a language model, and it says plainly when nothing fits rather
  than returning the least-bad answer as though it were right.
- Projects are a plug-in folder like blocks and recipes.
- Each project carries its data requirements, the recipe to use, settings worth
  starting from, what a working run looks like, and where it goes wrong. The
  pathology project leads with site-based splitting, because that is what
  actually ends those projects.
- Fixed, found by sweeping all 101: a detached branch step became the implicit
  parent of the next step, so graph networks wired the adjacency matrix in as
  node features and the linear solver received one input instead of two. A
  detached step no longer advances the chain, and an explicit connection to a
  step's first node now replaces the implicit one.
- Fixed: the request matcher scored a word by its rarest possible synonym rather
  than the tag it actually matched, so every medical project ranked as highly
  for "slides" as the slide-level project. It now weights by the matched tag's
  rarity, and "molecules" no longer stems to "molecul".
- Six more tests, 35 in total, including one that builds all 101 projects and
  checks each resolves and generates code.

## 1.2.1

- Fixed `release.sh`: it scraped the version with a regex for the first quoted
  string, which matched the docstring at the top of `version.py` rather than the
  number, and then tried to name a git tag after it. It imports the module now.
- The script also refuses anything that is not `N.N.N`, so a malformed version
  stops the release instead of reaching `git tag`.

## 1.2.0

The rest of the out-of-reach list, plus a release script.

- **GAN** recipe. Two networks, two optimizers, alternating updates, the
  non-saturating generator loss. The canvas graph is the generator; the
  discriminator is a second saved design picked in the form. This is the case
  that justified recipes owning their own backward pass. `Generator.json` and
  `Discriminator.json` ship as a working pair.
- **Reinforce** recipe. Policy gradients with a moving baseline and a CartPole
  environment built in, so it runs with no dependencies and no dataset. Solves
  the task: mean return climbs from 14 to the 300-step cap within four epochs.
- **Detection** recipe. Grid-based single-shot detection — objectness, box
  regression and class per cell, with centre-cell assignment. Draws its own
  squares and circles so the loss and the assignment can be exercised without
  annotated images; reaches recall 0.93 and class accuracy 0.95. Real
  annotations still need a loader.
- **`self_supplied` recipes** drive their own loop with no DataLoader at all,
  which is what reinforcement learning needs and nothing in a dataset can
  express.
- **`extra_models`** lets a recipe request further networks, built from saved
  designs.
- **`data_shape`** lets a recipe tell the loader what the data looks like when
  it differs from the model's Input — a GAN takes noise but reads images.
- **`release.sh`**: runs the tests, commits, tags from `version.py`, and pushes.
  It refuses to push if the tests fail.
- Fixed: a recipe metric named `loss` overwrote the objective in the epoch row,
  so REINFORCE reported its policy loss where the return should have been —
  including negative numbers for a quantity that cannot be negative. Non-objective
  metrics are now renamed on collision, and a test asserts the return stays
  positive.
- Six more tests, 29 in total.

## 1.1.0

Training loops become pluggable.

The five things listed as out of reach — GANs, diffusion, reinforcement
learning, contrastive pretraining, detection — were never architecture problems.
Every one of those networks already built on the canvas. What blocked them was
`train.py` assuming one model, one optimizer, and a loss computed from
predictions and labels. So rather than special-casing five workflows, the loop
itself is now a plug-in.

- **`recipes/` folder**, the same shape as `blocks/`: hot reload, isolated
  failures, editable in the app, scaffold from the New button.
- A recipe owns its **backward pass and optimizer steps**. Anything less general
  could not express a GAN, where two updates interleave and each needs its own
  graph handling.
- A recipe can **refuse a graph it cannot train**, with a specific message. The
  autoencoder says which shapes disagree; diffusion says exactly what to set the
  Input to.
- **Autoencoder** recipe — reconstruction and denoising. There is no label, which
  the old loop could not express at all.
- **Contrastive** recipe — SimCLR. Builds its own two augmented views per step
  and uses NT-Xent. Verified on structured images: within-class cosine
  similarity 1.000 against between-class -0.466, with no labels used.
- **Diffusion** recipe — DDPM training with a cosine schedule, timestep carried
  as an extra input channel, DDIM sampling for previews.
- Fixed during development: the first diffusion sampler used per-step DDPM
  coefficients while skipping steps, which under-denoises and diverges — samples
  came out in the range ±150 against data in [0, 1]. Replaced with DDIM, which
  is valid at any stride. The preview reports the sampled range against the data
  range so this class of failure is visible rather than silent.
- Six more tests, 23 in total.

## 1.0.1

- README rewritten as a repository landing page: what the tool is and is not,
  a quickstart above the fold, the worked example as the hook, and the reference
  material after it rather than before. No content dropped — the multi-input,
  core-layer and sampling sections are all still there, further down.

## 1.0.0

Packaged for release.

- **Test suite** in `tests/`, 17 checks, runnable with plain Python or pytest.
  The torch-dependent ones skip themselves when it is absent. They cover the
  things that would make the tool untrustworthy rather than merely broken: that
  generated code runs, that predicted shapes match what PyTorch produces, that
  the inspector shows the same text as the exported file, that the causal mask
  masks, and that resnet18 reimports to the exact parameter count.
- **Continuous integration** on Python 3.10 and 3.12.
- MIT licence, `.gitignore`, and directory placeholders. Weights, uploads and
  downloaded datasets stay out of version control; the worked example and its
  corpus ship.
- Fixed, and found by the new tests: `padding="same"` was accepted alongside a
  stride above 1. Keras allows that, PyTorch does not, so the canvas was
  approving a network the generated code could not build. It is now refused
  with the explicit padding to use instead.

## 0.9.0

Making large graphs navigable. Importing DenseNet121 produced 429 layers laid
out as a single strip 46,000 pixels tall — technically correct and impossible to
work with.

- **Deep graphs wrap into columns.** Past about seventy layers the layout folds
  into columns instead of running straight down. DenseNet121 now lands on a
  canvas of 4138 x 3610 rather than 258 x 46350.
- **Minimap**, bottom right, with the current viewport drawn on it. Click or
  drag to move around. Toggleable.
- **Find a layer** with Cmd/Ctrl+K. Type part of a name or type, arrow through
  the matches, Enter to centre on it and select it.
- **Marquee selection** — shift-drag on empty canvas to sweep up everything in
  the box.
- **Right-angled wire routing** as an option. Curves read better on small
  graphs, right angles on large ones.
- **Snap to a 20px grid**, off by default.
- Shape callouts on wires hide past sixty layers, where they turn into noise.
- Tidy uses the same column wrapping, so an already-open deep graph can be made
  navigable without reimporting.

## 0.8.0

- **Chat tab.** Type a prompt, get an answer from a model you trained. Pick the
  checkpoint, set temperature, top-k and length, and choose whether your text is
  wrapped in the corpus's turn format or continued as-is.
- The checkpoint is rebuilt from the design stored inside it and kept in memory
  after the first message, so replies come back in a fraction of a second.
- Vocabularies now travel inside the checkpoint, so a trained text model is
  self-contained. Older checkpoints fall back to the `.vocab.json` their
  TextGenerator node points at.
- Stop sequences: a dialogue reply ends at the end of its turn instead of
  running on into an invented next question.
- Characters outside the model's vocabulary are reported rather than silently
  dropped.

## 0.7.0

A worked example: a small GPT, built on the canvas and trained in the app.

- **`GPTStack` block** — pre-norm causal transformer blocks. The mask is the
  point: without it a position can read the token it is being asked to predict,
  and the loss collapses while the model learns nothing.
- **`TextGenerator` runtime block** — sampling with temperature and top-k.
  Runtime, not a layer, because generation is a loop that calls the model
  repeatedly.
- **Text corpora as a dataset.** Upload a `.txt`, and it becomes character-level
  next-token pairs. The vocabulary is written alongside it for the generator.
- **`language_modeling` task** on the Output layer: cross entropy over every
  position at once, reported with perplexity and next-character accuracy.
- **Samples during training.** A continuation is drawn after every epoch, so
  progress is legible as text rather than only as a falling number.
- Guard: a corpus vocabulary that disagrees with the Embedding or the final
  Linear is reported before the run wastes your time.
- Ships `saved/MiniGPT.json` and an original 471 KB corpus. Open the design,
  pick the corpus, train.

## 0.6.0

- **Import an existing model.** The Import button takes a torchvision
  architecture by name, an `.onnx` export, or a `.pt` holding a module, and
  rebuilds it as editable layers on the canvas.
- PyTorch import goes through `torch.fx`, so a traced `Conv2d` arrives carrying
  its real arguments rather than as an opaque box. Eleven of sixteen torchvision
  architectures rebuild with parameter counts identical to the originals,
  including all the ResNets, VGGs, MobileNets, DenseNet and RegNet.
- Models whose `forward` branches on tensor values cannot be traced into one
  graph, and the error says so rather than failing obscurely. A `.pt` holding
  only a state_dict is rejected with an explanation: weights carry no
  architecture.
- Operations with no equivalent in the registry become stub nodes that keep the
  original call in their values and are listed in the import summary. A stub
  that is flagged beats a substitution that looks right and is not.
- **Add and Multiply now broadcast**, following PyTorch's rules: axes must match
  or be 1. This is what squeeze-excite and every gating block need, and its
  absence was blocking MobileNetV3 and EfficientNet from importing.
- Fixed: layers named from `Sequential` submodules ("0", "1") generated invalid
  Python identifiers.
- Fixed: braces inside a Custom block's code were evaluated as template
  expressions; `{{` and `}}` are now literal braces.

## 0.5.0

Closing the gap with MATLAB Deep Network Designer.

- **Trained weights persist.** Every run saves its best epoch and its final
  epoch to `checkpoints/`, with the design embedded. Download them, reopen the
  design that produced them, or start a new run from them.
- **Transfer learning.** A new `Backbone` block wraps nine pretrained
  torchvision networks with their classifiers stripped: resnet18/34/50/101,
  vgg16, densenet121, mobilenet_v3_large, efficientnet_b0, convnext_tiny.
  Freezing is expressed as trailing stages left trainable, and a non-RGB input
  re-purposes the pretrained stem by averaging its kernels.
- **Start from checkpoint.** Matching tensors are copied and a resized head is
  reported and left fresh, so swapping the class count is a normal operation
  rather than an error.
- **Folder of images.** Point at a directory of class-named subfolders. Scan
  reports the classes and counts, and warns when the class count disagrees with
  your last Linear layer. Six augmentations, applied to training only.
- **Early stopping** after a configurable number of epochs without improvement.
- **Undo and redo** over the whole editing history, with buttons and
  Cmd/Ctrl+Z. Typing in a field collapses into one entry rather than one per
  keystroke.
- **Multi-select and clipboard.** Shift-click to extend, Cmd/Ctrl+A for all,
  drag to move a group, copy and paste with connections between the copied
  layers preserved and rewired to the new ids.
- Parameter totals that include a pretrained backbone are marked with a tilde,
  since that figure is a published lookup rather than derived from the graph.

## 0.4.0

- Selecting a layer now shows the code it generates. The inspector displays the
  constructor and the forward line for that node, verbatim from the generated
  file rather than reconstructed, so the panel cannot drift from the export.
- Block layers get an Edit button that opens their file in the Blocks tab. Core
  layers get Show in Code, which jumps to the generated file and highlights the
  line.
- Layers that ship a class definition (every block) can expand it inline.
- The inspector states where each layer is defined, and flags the ones with no
  Keras equivalent.

## 0.3.0

- Plug-in block system. Files in `blocks/` install layers into the palette with
  no restart and no edits to the core. A block that fails to import reports its
  traceback and is the only thing that breaks.
- Block editor in the Blocks tab: browse files, edit source, save with
  Cmd/Ctrl+S, hot-reload. New scaffolds a working skeleton.
- Two block kinds. `layer` is differentiable and lives in `forward()`.
  `runtime` wraps the trained model and generates a separate
  `build_runtime()` — search and self-play are not tensor transforms.
- Ten blocks shipped: ResidualBlock, SqueezeExcite, InceptionBlock, PolicyHead,
  ValueHead, MCTSSearch, GraphConv, RidgeSolve, ODEBlock, FixedPoint.
- Blocks can declare `learnables`, so the parameter count in the header stays
  exact when a graph uses them.
- Keras files now open with a header naming any node that has no Keras form,
  instead of silently emitting a different model.
- Fixed: `index.html` was resolved as `HERE.parent / "frontend"`, which broke
  when `main.py` sits at the project root. Four locations are now checked, and
  a missing page reports where it looked rather than raising a 500.

## 0.2.0

- Multi-input graphs train. Any number of Input layers is supported.
- Fixed: `forward()` takes its arguments in topological order, but inputs were
  collected in canvas order, so a two-tower graph could feed the wrong tensor
  into the wrong tower.
- Multiple Outputs train together; the first carries full weight, later heads
  are scaled by a configurable auxiliary weight.
- CSV upload with per-Input column mapping, text-column encoding, mean fill,
  and standardization from training-split statistics only.
- A class-count mismatch between the target and the last layer is reported in
  the training log.

## 0.1.0

- Drag-and-drop canvas with live shape propagation and per-node error messages.
- 34 core layer types.
- PyTorch and Keras generation from one graph.
- Training console with SSE metrics streaming.
- Wire thickness scaled by the log of activation volume.
