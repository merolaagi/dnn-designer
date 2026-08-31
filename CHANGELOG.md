# Changelog

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
