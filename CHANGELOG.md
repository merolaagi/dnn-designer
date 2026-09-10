# Changelog

## 1.68.0

**The codebase panel gains the canvas's views — three of the four, honestly.**

**Measures**, not Maths. The canvas has a Maths tab because a layer *has* an
equation. A Python function does not, and writing one would be inventing
something. What a function does have is structure that can be counted, so it is
counted: paths through it, nesting depth, length, arguments, exits. On MARE's
`engine.py`:

```text
ResearchEngine.prepare_next_round   24 paths   86 lines   3 deep
ResearchEngine.enrich_literature    19 paths   43 lines   4 deep
ResearchEngine.adversarial_review   18 paths   53 lines   5 deep
```

Paths is one plus every decision — each `if`, loop, `except`, comprehension and
`and`. The panel says explicitly that this counts cases to hold in mind and not
quality: a long dispatch table scores badly and reads fine.

**Walkthrough**, replayed rather than inferred. The canvas can walk an example
through a network because the wiring fixes the order. In code it does not —
which function runs next depends on the data — so this replays a recorded run,
step by step, with the depth of the descent preserved. Without a run it says why
it has nothing to show rather than inventing a plausible path. The tracer now
records call order and nesting, not only counts.

**Notes**, which is the Assistant's job here. Grounded in what was counted:
which function has the most paths and why that matters, how many modules import
this one, how many it reaches into, functions nested five deep, undocumented
functions. On `engine.py` it reports 10 modules importing it, 13 reached, and
names the three functions nested five deep. Every line ends with the reminder
that it counted rather than judged.

**Story** is not there, and should not be: the canvas's Story reads a batch
through the layers with the real numbers. The equivalent for code is the
walkthrough, and having both would be one idea wearing two names.

## 1.67.0

**The import dialog fills in what to pass.** The scout has worked this out for
its own imports since 1.61, and leaving a person to guess while the same
program can answer it was not a boundary worth keeping.

On `onnx/models`, scanned against a 3×224×224 input:

```text
ConvLayer            (in_channels, out_channels, kernel_size, stride)   3, 32, 3, 1
ResidualBlock        (channels)                                         3
UpsampleConvLayer    (in_channels, out_channels, kernel_size, stride)   3, 32, 3, 1
CombinedDecoder      (decoder, lm_head, config)                         left blank
SimplifiedT5Encoder  (encoder)                                          left blank
GenerativeT5         (encoder, decoder_with_lm_head, tokenizer)         left blank
```

All three filled ones import: 896, 168 and 896 parameters.

Three things about how it is presented:

- **The placeholder is the signature**, so a box you have to fill yourself says
  what it wants rather than "what to pass".
- **A filled box is marked as a suggestion to check**, not an answer. It is
  inferred from parameter names and your input shape, and a name can lie.
- **A box that cannot be inferred stays empty and says why.** A tokenizer or a
  config is a structural decision; filling it with something plausible would
  produce a model that builds and means nothing.

## 1.66.1

**Fixed: clicking a module on the map did nothing.** My own dragging broke it.
Capturing the pointer makes the SVG the target of the click that follows, so
looking up which box was clicked always found the SVG and never a module. The
pressed node is remembered on the way down instead, and a press that never
moved counts as a click on the thing that was pressed.

**And the page is shaped like the canvas now**, which is what it should have
been from the start: **tree, drawing, panel**. Picking a module fills the panel
and draws its diagram on the stage — the drawing stays put and the panel
changes, exactly as the design canvas works.

The panel has three tabs:

- **module** — classes with bases, methods and docstrings; functions; and what
  it reaches into, each one clickable to jump there.
- **source** — the file itself.
- **run** — the tracing, which used to be buried under the diagram. It states
  what running does and does not protect against before offering the button.

Once a run has happened, the diagram marks in green every definition that was
actually entered, and says how many observed calls the source reading missed —
those being the calls through variables, which is the whole reason for running
it.

## 1.66.0

**Run it and watch.** The diagrams drew only calls that reading could be sure
of, which left out `self.provider.run()` — and in a well-made system that is
most of the interesting calls. MARE's `agents.py` drew no edges at all for
exactly that reason.

Now a file's diagram offers to run the project and record what actually called
what. On a class whose collaborator is injected, the trace produces
`Engine.run → Provider.call`: the edge that cannot be read off the source at
all.

**About running it, plainly.** It happens in a subprocess, never in the
application process, never automatically, and it is killed after a timeout that
is reported rather than disguised as an empty result. That stops an accident,
not an attacker — the code can still read files and open sockets as whoever runs
the server. The panel says so before you press anything, because a subprocess
with a timeout is not a sandbox and calling it one would be worse than saying
nothing.

Results are labelled *observed, not inferred*, and anything the static diagram
does not draw is precisely a call through a variable.

Found while building it: **`<frozen runpy>` is not a path**, but `abspath()`
resolves it against the working directory — which is the project — so every
runpy and posixpath frame looked like project code and the first traces were
mostly the machinery that started the program.

**Also asked for, also done:** the diagram surface is the same squared paper as
the design canvas, and every box can be dragged. A drag is not treated as a
click, so rearranging a module does not open it.

## 1.65.1

**Fixed: the codebase reader broke on any path through a symlink.**

On macOS `/var/folders` is a symlink to `/private/var/folders`. The reader
resolved the *file* but not the *root*, so the two looked like different trees
and every `relative_to()` raised:

```text
'/private/var/.../proj/m.py' is not in the subpath of '/var/.../proj'
```

Both are resolved now, or neither, in all five places that walk the tree. Linux
does not reproduce this by accident, so the test now reaches the project through
a symlink deliberately — and the path-escape refusal is checked in the same
place, because resolving more paths is exactly the change that could have
weakened it.

This is why the suite runs on your machine as well as mine: it is a real
platform difference, and it would have hit any project reached through a linked
directory, not only a temporary one.

## 1.65.0

**A codebase becomes diagrams.** Two of them.

**Every module, as one picture.** Each module a box, each import a line,
ordered so the most depended-upon come first and the entry points last. The
ones more than five modules import are outlined in purple. Click any box to go
inside it. On MARE v0.3: 34 modules, 247 imports, no cycles, with `mare.models`
sitting at the bottom where 114 references land.

**Each file, as its own diagram.** Classes with their methods indented beneath
them, module-level functions below, and a line for every call one makes to
another.

**Only the calls it is sure about.** `self.step()` is certain and is drawn.
`self.provider.run()` is not — which object that provider is cannot be known
without running the program, and drawing it would mean guessing. Those are
counted and reported instead: MARE's `agents.py` draws no edges at all and says
so, because all 29 of its calls go through injected providers. That is a true
fact about the design, and a diagram that invented arrows would have hidden it.

**It is deliberately not the design canvas.** That graph carries shapes and
parameter counts and generates PyTorch from them; a Python module has neither,
and putting one there would make every check downstream meaningless. A test
asserts the module diagrams never touch `state.graph` or open the canvas.

Fixed while building: `from mare.models import A, B, C` was listing three
symbols as three modules reached, making every file look far more entangled
than it is.

## 1.64.0

**Read a codebase.** Open a zip or tar of source and see what is in it: the
folder tree on the left, and for any file its classes, its methods and its
source. Nothing is executed — the archive is unpacked, parsed with `ast`, and
described.

This is deliberately not the model importer. That looks for `nn.Module`
subclasses it can trace onto the canvas, and pointed at a research engine it
finds nothing and says nothing useful. Most code is not a neural network.

What it says about any Python project:

- **The tree**, with `__pycache__`, `.venv` and the rest left out.
- **Per module** — classes with their bases, methods and first docstring line;
  module-level functions; line counts.
- **The import graph**: which modules depend on which, which is most depended
  upon, and whether there are cycles. A cycle between modules is the same
  defect as a cycle in a proof DAG — there is no order in which the pieces can
  be understood one at a time.
- **What it depends on outside itself**, ranked.
- **Files that will not parse**, reported rather than skipped: dropping them
  quietly would make every total a little wrong.

Two things found while building it, on a real 34-file project:

- **`from pkg import b` was recorded as an import of `pkg`**, discarding which
  module `b` is. Every such edge pointed at the package instead of the module,
  which hid import cycles completely — a deliberately cyclic fixture came back
  clean. Fixed, and the real project's edge count went from 81 to 247.
- **An archive naming `../../owned.txt` would have been extracted.** Python's
  own guards here differ by version, so the check is done in the reader where
  it can be relied on, and the same check refuses a request to read a file
  outside the project.

## 1.63.0

**The papers scout keeps going, and proves the papers it keeps.**

A round that returns nothing usable is a failed attempt, not an answer, and
leaving it on screen serves no purpose. So the scout now asks up to five times,
each a different hypothesis about why the last one failed:

    as asked
    asking for a model of it
    asking for the dynamics rather than the task
    asking for a compartmental description
    asking straight for a well-posedness result

**And the test for "modellable" is now reading the paper.** Matching a question
and naming mathematics are cheap tests a title can pass. The scout fetches each
promising candidate, ingests it, and keeps it only if a passage claims something
is unique, stable or conserved — quoting the passage that convinced it. A paper
reporting 91.2% accuracy is read and rejected; one stating a theorem about a
two-compartment system is kept. It stops as soon as it has enough.

- **Papers read and rejected are reported**, so the rounds are legible rather
  than a silence.
- **An unreachable search stops the loop immediately** instead of asking the
  same question five more ways. Nothing is wrong with the question when the
  phone is off the hook, and it says which host refused.
- Fixed while testing: the wrapper that finishes a scout wiped the explanation
  the scout had written, because it assigned the empty string over it.

When every round comes back empty it says so about the subject rather than the
search: a task like classifying images has no uniqueness theorem to find, and
the code scout and Draft a design are the tools for those.

## 1.62.0

**Two real gaps in the papers scout, both mine.**

**It was ranking papers by the wrong thing.** On "differential diagnosis from
chest Xrays" the top result was *Effect of transcriptional delay on ribosome
abundance control* — not a near miss but a different subject. The library works
out how well a paper matches the question and how much mathematics it names, and
my first ranking threw both away, scoring on fetchability and citation
recurrence alone. Under that rule the ribosome paper scored 8 against an
on-topic paper's 1; it now scores 12 against 20. Being readable here is a
convenience and cannot outweigh being about something else.

Each card now says whether the paper matches the question at all, and a search
where none of them do says so rather than ranking the wrong papers quietly.

**Following the reference lists collapsed under 503s.** Six reviews means six
requests in a row, which is a burst, and Europe PMC answers a burst with
*Service Unavailable*. Every one failed and the stage reported "0 distinct
references" — and that is the stage that finds the paper actually worth
modelling, so losing it lost the best half of the search. Requests are paced to
about three a second and temporary refusals are retried with backoff. A 404 is
not retried, because it means never rather than later.

The fetching itself was working, incidentally: that run did return 12 papers with
5 fetchable. What failed was the part that would have found better ones.

## 1.61.0

**The missing step was real, and it was my rule that was wrong.**

The code scout refused to supply any constructor argument, on the grounds that
guessing makes the evidence meaningless. That is true of an *unverified* guess
and wrong about a verified one. `depthwise_separable_conv(nin, nout)` is plainly
two channel counts; if the class builds at exactly PyTorch's parameter count and
a batch goes through it, the hypothesis has passed. My rule left whole
repositories unreachable for want of the number 3.

The scan now records the *names* of the required arguments, not just how many.
A numeric one is inferred from its name and the input shape, then built, counted
and run — and reported, so it can be corrected. On the same search that returned
nothing before:

- `depthwise_separable_conv` — 4 layers, **123 parameters, matching torch
  exactly**, built with `nin=3, nout=32`
- `SeparableConv2d` — 99 parameters, matching exactly
- `Xception` — 170 layers, **22,855,952 parameters, matching exactly**

**A structural argument is still refused**, and the distinction is the point.
`Block` wants `in_filters, out_filters, reps`, and `reps` is a design decision
rather than a number derivable from a shape. Inventing it would make every
figure reported about the result meaningless. It says so, and names which
argument defeated it.

**And a paper now leads somewhere.** The papers scout found papers and stopped.
Findings with fetchable text open in *Paper → spec*, and the results say plainly
that a paper does not go on the canvas directly — it becomes a residual, the
residual is checked, and the result becomes a placeable layer. That is the route
the Bergman glucose model took, and it is now one press from a search result.

## 1.60.1

**Why nothing reached the canvas: none of those results had imported.**

The mechanism works — checked end to end on a real search, not a mock.
`MobileNetV2` from chenyaofo/pytorch-cifar-models imports exactly, carries a
155-node, 164-edge graph through JSON, and lands on the canvas with its
provenance badge.

But every result in the searches so far was a refusal, a class wanting a
config, or an inexact import like `Xception`. A class becomes a button only
once it has been built at the right size, so no button appeared — and the page
never said why, which makes an absent button look like a broken one.

- **The results now say how many can reach the canvas**, and when the answer is
  none, that it is a real answer about this search rather than a missing
  feature.
- **Each repository card says the same** when none of its classes can be
  opened.

No new button was added, and adding one would have been the wrong fix: placing
an unverified graph is exactly what the checks exist to prevent. `Xception`
traced 576 parameters short of the real model, and a canvas holding that would
give wrong numbers for size, precision and ablation while looking perfectly
healthy.

To see it work, search something whose repositories hold plain modules —
"pytorch cifar classification models" returns `MobileNetV2` — rather than
language model code, where nearly everything takes a config.

## 1.60.0

Three things from using the scouts in anger, all fair.

**Errands are kept.** Forty seconds of searching the internet should not be lost
by clicking away, and two searches for the same thing return slightly different
repositories — worth comparing rather than only regretting. Finished scouts are
written to `scouts/`, listed under *already sent out*, and reopened with their
findings intact. This also fixes the vanishing results from 1.59.1 at the other
end: uvicorn reloading emptied the register, and now the errand outlives it.

Found while testing: **the status was set before the file was written**, so a
client that saw "done" and immediately asked for the history could find nothing
there. The errand is now on disk before it is called finished.

**"Explain this design" says which design.** It reads whatever is on the canvas,
which was left to be inferred. The card now names it — *"It will read GPT2 — 18
layers on the canvas now"* — and says plainly when the canvas is empty rather
than offering to explain nothing.

**A design placed by a scout says so, on the canvas.** A badge above the
flowchart reads *"drafted by a scout · Small CNN"* with a link straight back to
the results it came from. That answers the real question — whether the thing you
are looking at is the candidate you chose — and going back no longer costs the
results, since they are kept now. The badge disappears when the canvas holds
something else, so it can never claim a design it did not put there.

## 1.59.1

**The blank panel was a bug. The "refused" lines were not.**

- **Fixed: a scout the server had forgotten painted an empty panel** headed
  *"undefined · undefineds"*. The poll was rendering a 404 body as if it were a
  snapshot. The usual cause is uvicorn reloading on a saved file, which empties
  the register while the scout is out — it now says exactly that, and to send it
  again.
- **It was trying test files.** `tests/test_parametrize.py` and
  `test_linear4bit.py` hold stubs for exercising something else, and importing
  them teaches nothing about the repository. Tests, examples and benchmarks are
  set aside now, so the classes it tries come from `lit_llama/model.py` and
  `bitsandbytes/nn/modules.py` instead. They are still counted, so the totals
  stay honest.
- **Four identical refusals are now one conclusion.** The three kinds have
  different answers and only one is a dead end: classes wanting a config
  (Import can supply one — a scout cannot, because guessing the values would
  make every number it reported meaningless), classes needing packages this
  machine lacks (install them), and classes that cannot be traced at all
  (a real dead end for importing, though the source is still readable).

The refusals themselves are the feature working. `GGMLLayer` needs the `gguf`
package; NVIDIA's `Discriminator` branches on tensor values; lit-llama's four
classes all take a config. Each of those is true, checked by running it, and
more useful than a confident guess would have been.

## 1.59.0

**A fourth scout: "Draft a design for me."** Yes — it puts a network on the
canvas.

Describe what you want, optionally give it an input shape and a number of
classes, and it finds the guided builds closest to the description, assembles
each one whole, refits it to your shape and head, and **checks it before
offering it**: shapes resolve, the parameter count matches PyTorch, and a batch
goes through. Only then is it placeable, in one press.

A real run on "classify small colour images" with `3,32,32` and 10 classes
returned four candidates, all four assembled and ran — 60,554 parameters each
for the convolutional ones, with the input shape and head refitted and both
changes listed on the card.

It assembles from patterns that exist rather than inventing an architecture, and
that is the reason the result can be checked at all. A drafted design that fails
any of the three checks is shown with what went wrong and cannot be placed;
`placeDraft` re-checks the finding rather than trusting that the button was
hidden.

Also added: `assemble()`, which turns a guided build's plan into one graph. The
frontend had always walked those a step at a time and nothing did it in one go,
which is what a scout needs.

## 1.58.0

**Scouts.** Three errands that go and look, and then check what they found.

The useful thing about a scout here is not that it can search — anyone can
search — but that this application can *verify*. So every finding ends with
evidence produced by running it, and the shortlist is ranked on that rather than
on stars or citations.

- **Find code worth importing.** Searches GitHub, downloads each repository,
  scans it, and tries to import the classes it finds. Reports which imported and
  at what size, and the specific reason the rest refused. A real run on
  "residual network cifar" found `GoogleNet` importing as 225 layers with
  6,402,564 parameters matching torch exactly — and `BasicBlock` refusing
  because it takes two constructor arguments, which were not guessed at, since
  a guess would make the evidence worthless.
- **Find a structural result.** Searches Europe PMC and arXiv, follows the
  reference lists of the reviews, and ranks by whether the text can actually be
  fetched — a paper that cannot be read here cannot become a layer here,
  however good it is.
- **Explain this design.** Walks the canvas layer by layer: the equation with
  this network's own numbers, where the parameters are, what each setting does.
  Needs no internet.

**An inexact import is not a success.** One class traced to 6,040,456 parameters
where torch says 6,789,408. The graph is not the model, so every number taken
from it afterwards — size, precision, ablation — would be about a different
network. Those are marked, scored below the ones that refused outright, and
cannot be opened on the canvas.

An exact one can, in one press, because it was already built once to earn its
place on the list.

Also fixed: the Python 3.11 parse check was walking the repositories the scouts
download, and failing on somebody else's Python 2. It now covers our own code
only.

## 1.57.0

**A run now says how much data there is per parameter, and how much of it will
be seen.**

Two numbers that explain most disappointing runs, and neither was on screen
anywhere. A GPT-2 trained on a 470,937-character corpus is **346 parameters per
character**, seeing the corpus **1.09 times** in five epochs of 1,600 crops.
Both are arithmetic, so they are stated rather than guessed at:

> *163,037,184 parameters against 470,937 training values — 346 per value. A
> network with that many parameters per value can memorise the set rather than
> learn from it. Falling validation loss is the thing to watch; if it turns
> while training loss keeps dropping, that is what happened. This run will see
> the data about 1.1 times over. Little of what a model can learn is learned in
> one pass.*

Neither figure refuses anything. Plenty of useful work happens at ratios like
these deliberately, and a tool that blocked it would be wrong more often than
the ratio is. But it should not be invisible, and it was.

Fixed while writing it: the first version called the report before the data
loaders existed, which is an `UnboundLocalError` rather than a report. The test
now pins where it is called as well as what it says.

## 1.56.1

**Fixed: sampling failed on any prompt shorter than the context length.**

`RuntimeError: shape '[1, 64, 12, 64]' is invalid for input of size 15360` —
which is a twenty-character prompt meeting a sixty-four position model. 15360 is
20 × 768.

A model written with explicit reshapes has its context length baked into its
arithmetic: the imported GPT-2 block reshapes to `[B, 64, 12, 64]`, so it cannot
take a shorter sequence at all. The sampler was passing the prompt at whatever
length it happened to be. It now pads the window to the full context and reads
the prediction from the last real position.

Short prompts, an empty prompt and an over-long one are all tested.

This one only surfaced because 1.56.0 made sampling actually run for a design
labelled `classification`. Three versions of fixes each uncovered the next
thing, which is what it looks like when a path has never been exercised end to
end rather than when something has regressed.

## 1.56.0

**One determination, used everywhere.** Last version I made the loss follow the
tensors instead of the Output layer's label, and left four other places still
asking the label: the training accuracy, the validation accuracy, the perplexity
and the sampled continuation. So a design labelled `classification` trained
correctly and then reported no perplexity and produced no sample — which is why
a run that worked still looked like it had not.

The question is now asked once, at the first batch, and the answer read
everywhere. Verified on a deliberately mislabelled model: samples after each
epoch, perplexity 11.55 → 9.69, accuracy climbing.

**Precision can be measured on trained weights.**

Drift on untrained weights describes the shapes rather than the model, and the
difference is real: on the same design, 4-bit drift was 0.131 untrained against
0.112 trained. The Precision view now takes a checkpoint, says which weights a
number came from, and says plainly when the answer is about the shapes instead.
Weights from a different network are refused rather than half-loaded.

**A correction to something I said.** I suggested the validation loss sitting
below the training loss might mean the text split was leaking. It is not: the
loader splits the corpus at 90% and takes disjoint halves. The gap is that the
training figure is an average over the epoch while validation is measured after
it — on a repetitive corpus that alone accounts for it. I should have read the
loader before offering the theory.

## 1.55.0

**Fixed: a model that predicts at every position could not train.**

`RuntimeError: Expected target size [64, 50257], got [64, 64]` — which describes
the tensors and not the mistake. The mistake was mine, in two places, and the
same one both times: the training loop decided how to compute the loss from the
Output layer's **label** rather than from the shapes in front of it.

A design that emits a distribution at every position of a sequence is a language
model whatever its Output layer says. Both the loss and the accuracy now follow
the tensors — and the accuracy check was the second place, producing a different
error from the same cause, which is why fixing the loss alone only moved the
failure.

- **The shipped GPT2 example was mislabelled.** Its Output said
  `classification`; it is `language_modeling`, and now says so.
- **A design still labelled wrongly now trains anyway**, and says once that it
  is being treated as a language model and which setting would make that
  explicit.
- The note fires once. My first version guarded it with `not job.notes`, and
  `emit("note")` appends nothing to that list — so it would have printed on
  every batch.

Verified on a per-position model labelled `classification`: loss 3.17 → 2.32
over three epochs with accuracy climbing, where before it did not start.

## 1.54.1

**Fixed: "blocked" was as silent as "error" had been.**

Last version I made a failed run explain itself and fixed exactly one of the two
branches that can fail. The other — a run the server refuses before it starts —
still wrote its reason to the collapsed log and put one word in the corner. So
"error" became "blocked" and told you no more than before. That is a poor way to
fix something, and the fix now covers every way a run can end badly:

- **Refused before starting**: the reason and every listed problem appear in the
  panel.
- **Failed while running**: unchanged from 1.54.0.
- **The connection dropped**: previously ended the run in silence, which looks
  exactly like a run that finished. It now says the stream was lost and that the
  job may still be going on the server, and shows *disconnected* rather than a
  status that implies it stopped.

A test now walks every place the training status is set and requires the failure
states to explain themselves. Verified by silencing one branch and watching it
fail — which is what I should have done when I wrote the first half of this.

## 1.54.0

**A failed run now says why, above the button that started it.**

The reason was being written into the log strip along the bottom of the panel,
which is usually collapsed. So what a failed run actually said, on screen, was
the word *error* in a corner — which is how a question like "what is the error
still" comes to be unanswerable from a screenshot. The message appears in the
panel now, and a new run clears the last one.

**The corpus hint changes the design instead of describing the change.**

It read "set Embedding vocab and final Linear units to 48" and left you to find
the two layers. It is a button now. On GPT2 that mismatch is 77M parameters of
embedding table serving a 48-symbol alphabet — worth acting on, and worth acting
on in one press. The change is undoable, and it writes back to the workbook
rather than only the canvas, which a sheet-level edit would otherwise lose.

Worth stating plainly: **the vocabulary mismatch is not itself an error.** A
vocabulary larger than the corpus simply leaves those rows untrained, and the
loss starts near log(50257) instead of log(48). It is waste rather than
breakage, and the panel now treats it as such.

## 1.53.0

**Fixed: training GPT2 did nothing, for two reasons at once.**

**A workbook was being trained one sheet at a time.** GPT2's model sheet places
twelve `Subgraph` nodes standing for a class the `block` sheet defines. Only the
open sheet was sent, so the generated file named a class it did not contain and
the run died with *"NameError: name 'Block' is not defined"* — true, and no help
at all. The page now sends the whole workbook when there is one, and the whole
workbook is generated.

**And GPT2 was pointed at MNIST.** A language model reads token indices; MNIST
hands out images. There is no sensible reading of a picture as a sentence, and
the failure was not polite about it: the image loader, asked to resize a picture
into a token sequence, aborted the process. The pairing is now refused before
anything is built, in the endpoint where the message reaches the form:

> *This design reads token indices — its Input is a sequence of 64 whole
> numbers, which an Embedding turns into vectors. MNIST digits hands out images.
> There is no sensible way to read a picture as a sentence…*

The reverse is refused too, and the pairings that do work are untouched.

**Caught by my own test while fixing it:** the new message used a backslash
inside an f-string expression, which Python 3.11 rejects. The same fault as
`mathbook` in 1.22.1, found the same way.

## 1.52.0

Three faults from one session with minGPT, all of them fair.

**The Code tab threw you out when you clicked a layer.** Selecting a layer
switched the panel to Layer unless the open tab was on a list of exceptions —
a list that had grown with every new tab and never gained Code. So clicking a
layer to find its line closed the file you were reading it in. Only two tabs are
useless without a selection, so those are named instead: a list that stays short
rather than one that goes stale.

**The argument box looked like it wanted a count.** It said "1 argument needed",
the box was too narrow to show it, and what survived was "1" — which reads as a
value. It now shows the name from the class's own signature, `config` for
minGPT's three, and a line above the list says plainly that the box is *what to
pass*, not how many: build it in Setup and name it here.

**Training a design with no weights failed with torch's message, not ours.**
Importing `NewGELU` gives exactly that: an elementwise power, an add, a tanh and
a multiply, with nothing learnable anywhere. The optimizer then says "optimizer
got an empty parameter list", which reports what broke and not why. It now says
there is nothing to train, and which kinds of layer would give it something —
before the run starts rather than inside it.

## 1.51.1

**Fixed: after importing, the design was nowhere to be seen.**

Two faults, both of which only appeared once the launch pad became the page that
opens — a regression I introduced in 1.42 and did not go back and check.

- **Importing left you on the launch pad.** The design loaded correctly and
  nothing said so. Anything that loads a whole design — an import, opening a
  saved one, opening a past run, building from a domain — now brings the canvas
  up, which is the point of doing it.
- **The fit was computed against a hidden canvas.** A hidden page has no size,
  so the graph was centred inside a zero box and placed off screen. Pressing
  Canvas afterwards showed an empty-looking canvas with the design outside it.
  A fit asked for while the canvas is hidden is now remembered and performed
  when it appears.

A test holds both: that `fitView` refuses to compute from a zero-sized canvas,
that the deferred request is honoured, and that every route loading a design
reaches the canvas.

## 1.51.0

**The assistant advises now, and it advises about your design.**

Opening the dock reads the graph and answers two questions: what the selected
layer is doing, and what this particular network invites next. A list of general
advice is the same list for every network, which is the same as no advice — so
nothing is suggested that the graph does not actually ask for.

**The selected layer**: its shapes in and out, its parameter count and share of
the whole, what it is for, and what each of its settings does in the terms that
matter — *"5×5 is 2.8× a 3×3; two stacked 3×3 see as far as one 5×5 for fewer
weights"*, *"more heads is free — the width is split, not multiplied"*.

**What the design invites**, worst first, each with what it would cost:

- Two dense layers with nothing between them, which compose to one dense layer.
- A convolution with no normalization after it.
- A `Flatten` handing 65,536 numbers to a dense layer — with the arithmetic:
  *"that one layer is 33,554,432 weights on its own; pooling first would save
  about 25 million."*
- Attention over embeddings with no positions, so word order is invisible.
- Four convolutions deep with nothing carrying the gradient back.
- The layer holding most of the parameters, named.

Each suggestion has a **Do it** button that runs the edit. A clean design is told
so plainly: *"nothing structural stands out — that is not praise, it means the
next question is empirical, which is what Studies is for."*

Nothing here predicts that a change will help. Whether it helps is what a study
measures; this says what is worth measuring.

## 1.50.1

**Fixed: the 8-bit row did not run on Apple silicon.**

It used torch's dynamic quantization, which needs a quantized backend — and
which backends exist differs by machine. On x86 there are four; on Apple silicon
there is qnnpack, and the path this took was not available. So the row reported
"unavailable" for a scheme that is perfectly measurable, and the test asserted
its presence and failed.

Both integer schemes now go on the grid the same way, with no backend involved.
That removes the platform dependency, and it also makes the two rows comparable
— which matters more than using the vendor kernel for one of them. The test now
asserts every scheme runs rather than assuming it: a measurement that depends on
the machine's kernels is not the same measurement everywhere.

## 1.50.0

**A Precision view, in the Run tab.** Reads the design's weights at lower
precision and measures how far the output moves. Nothing is retrained.

This is the transformation behind a release like NVIDIA-Nemotron-…-NVFP4: same
architecture, same training, same weights read at fewer bits. What it saves is
arithmetic and knowable in advance; what it costs can only be run.

- **float16, bfloat16, 8-bit integers, and 4-bit in groups of 64** — the last
  being the shape of an NVFP4 or AWQ release. Size is computed including the
  scale each group carries; drift is measured by pushing the same batch through
  both models.
- **Which layer objects.** Every layer is quantized on its own, with the rest
  left alone, and the results are ordered worst first. On a small convolutional
  network the **head, at 1,280 parameters, moves the output more than the
  802,816-parameter hidden layer**. That is the "keep the last layer in higher
  precision" rule of thumb, arrived at by measurement rather than by repetition —
  and which layer it is depends on the network, which is why it is worth
  measuring.

A mixed-precision release is that list acted on: the layers that tolerate four
bits go to four bits, and the one or two that do not are left alone.

The 4-bit numbers come from putting the weights on the grid a 4-bit format would
use and reading them back as floats. The arithmetic error is therefore exactly
right and the size is calculated rather than observed — reporting a measured
error and a computed size is honest; reporting a measured size for a format
torch cannot store would not be.

## 1.49.0

**Fixed: "Check and save" saved, and said nothing.** The checks passed, the file
was written, the domain registered — and the page showed only the check results,
which reads as a save that did not happen.

Two faults, both mine:

- The failure path was a bare `return`. A save that was refused, or that got no
  answer, produced no message at all.
- The success path refreshed the layer palette *before* writing the
  confirmation, so an exception in that refresh swallowed the only sign that
  anything had been saved. The refresh is now guarded and the confirmation does
  not depend on it.

All four outcomes — saved, refused, no answer, refresh failed — now say so.

**And the step that was missing: "Put it on the canvas."**

Saving a spec registers a domain, which is a file. A saved domain now offers to
build the smallest network that runs it — an input, the layer on its covered
arm, an output — and opens it on the canvas with the Maths tab showing the
theorem it came from. That is the moment a paper becomes a model, and until now
nothing told you it had arrived or how to reach it.

Also: two saved specs can share a title, so the starter list shows each one's
key beside it.

## 1.48.1

**Fixed: the tick bar was still clipped, and stacking it did not help.**

The cause was a name collision. `.pickbar` was already the 13px progress bar in
the walkthrough's prediction list, with `overflow:hidden` — so the panel was
being clipped to one cut-off line by a rule written for something else
entirely. Everything about my own rule was correct, which is why changing it
twice changed nothing.

The panel is `.choosebar` now.

**A test for the whole class of bug.** CSS does not replace a repeated rule, it
merges with it, which is worse than replacing. The check flags a class that is
both a fixed clipped box and a container, since whichever is meant will come out
clipped. Two rules that simply add to each other are left alone — that is normal
and there were two such pairs already, correctly ignored. Verified by putting
the collision back: it reports `.pickbar`.

This is the second time a name reused for two things has cost a version. The
first was two functions called `scanFolder`, where the later definition silently
replaced the earlier.

## 1.48.0

**Start from a spec that works.** The check was right to reject a template with
its placeholder residual still in it — but being told the residual is missing
does not help you write one, and nothing in the app was helping.

- **Two working specs are offered beside the editor**, with their residuals
  shown: the Bergman one for a compartmental system of differential equations,
  the convex-ridge one for a potential whose gradient vanishes at the answer.
  Load whichever matches your paper's shape and edit it.
- **A rejection now appears beside the editor**, where it gets fixed, rather
  than only in the stage that reports it — and it is split into the separate
  things it names instead of one paragraph. Three problems read as three edits.
- The editor scrolls into view when a check fails.

**Two layout faults fixed:**

- The tick bar above the passages was a rigid row, so the sentence squeezed the
  button until both clipped. It stacks now.
- The three stages were equal columns, so the reading column ran to several
  screens while the other two sat empty at the top. The reader scrolls within
  itself and the other two stay in view beside it.

## 1.47.0

**The bridge from reading a paper to having a spec.** Until now the passages
appeared and then you were on your own with an empty editor, which is a gap I
should have closed when I built the search.

- **Tick the passages that carry the result**, then *Draft a spec from these*.
  The choice matters more than it looks: the top-ranked passage is not always
  the load-bearing one, and a draft made from the wrong theorem models the wrong
  thing confidently. So the choice is the reader's and it is carried through —
  `propose` takes the chosen indices rather than the whole paper.
- **With a proposer** (`ANTHROPIC_API_KEY` set) it drafts a candidate spec from
  those passages alone.
- **Without one** it fills the skeleton with what is actually known — the title,
  the source, your chosen passages, and any displayed expressions found — and
  leaves the residual blank. That is the part nothing here can honestly guess,
  and a plausible guess to correct is worse than a blank to fill.
- The passages you chose are shown beside the editor while you write, so the
  theorem is in front of you rather than scrolled away.

## 1.46.1

**Fixed: pressing "Read this one" looked like it did nothing.**

It was working — the passages were being fetched and ranked correctly. The
trouble was where the answer appeared: in the reading stage, which sits below a
results list that can run to thirty papers. Press a button, get no visible
reaction, and it does not matter that something happened somewhere.

- The button now says **reading…** while it works and goes back afterwards.
- A note appears beside it: how many passages were found, or the reason it
  failed. A paper outside the open access subset returns a 404 from the full
  text service, and that sentence now lands next to the button rather than only
  where the passages would have been.
- The reading stage scrolls into view when the text arrives.
- All three outcomes — read, refused, no answer — leave the button usable again.
  The first version could strand it saying "reading…" after a failure.

## 1.46.0

**Search for the paper, from inside the app.** Paper → spec now opens with a
question box: ask for the property or the system you want, and it searches
Europe PMC and arXiv, follows the reference lists of what it finds, and returns
a reading list you choose from.

Results come in four groups, because they are found four different ways:

- **Directly on the question** — matched the query itself.
- **What the reviews point back at** — turned up repeatedly in the reference
  lists of the well-cited reviews. This is usually where a modellable result
  actually is; the paper that states the theorem is rarely the top hit.
- **Reviews and surveys** — good for orientation, rarely the result.
- **Preprints** — from arXiv, where the source is available so the equations
  come through exactly rather than through a PDF.

Each result shows who, when, how often cited, how many reference lists it
recurred in, and why it was kept. **Read this one** pulls the full text and
ranks its passages straight into the reading stage.

- **A paper whose full text cannot be fetched offers no button.** Only the open
  access subset has retrievable text; a PMCID is not enough. A button that can
  only fail is worse than no button.
- **A network failure is not reported as an empty result.** The search library
  swallows an unreachable host and returns nothing, which reads as "no papers
  matched" when the truth is "nothing was asked". Those are different answers
  and the page now says which one it got, quoting the error.

This is the only part of the app that reaches the internet, and it says so.

## 1.45.0

**The six architectures that were missing are here.** All twenty-five on the
chart can now be built from layers in the palette.

- **GraphAttention** — attention over a node's neighbours, learned per pair
  rather than fixed by the degree. Non-edges are masked before the softmax, so
  it attends only where the adjacency allows; attention over a fully connected
  graph would just be attention.
- **MessagePassing** — the general form the others are instances of: build a
  message from each connected pair, aggregate, update.
- **CapsuleLayer** — dynamic routing by agreement, three rounds by default.
- **SpikingDense** — leaky integrate-and-fire neurons over simulated time, with
  a surrogate gradient, because a spike is not differentiable.
- **RBM** — with `gibbs()` and `free_energy()`, so a recipe can do contrastive
  divergence. Stacked these are a Deep Belief Network's body; the layer-wise
  pretraining that makes it one is a training procedure and the block says so.
- **Sampling** — the reparameterisation trick for a VAE. Random while training,
  the mean when evaluating, and it owes the loss a KL term, which `kl()` gives.

Each is tested for the behaviour that makes it that architecture, not only for
producing a tensor of the right shape.

**Two real defects found by testing behaviour rather than shape:**

- **Capsule routing was inert.** At the initialisation I first wrote, the
  predictions were so small that squash returned nearly zero and three rounds of
  routing differed from one by 5.6e-6. Shapes were right, parameter count was
  right, and the defining feature of the layer did nothing. Scaled by fan-in it
  now moves the output by 0.41 and the capsule norms sit near 1 as squash
  intends.
- **A design named after a layer shadowed it.** Calling a design `ResidualBlock`
  put a class of that name in the same file as the prelude defining the layer,
  and the model silently shadowed it — surfacing as a TypeError about arguments
  the model does not take. Reserved names now get `Net` appended, and the
  emitter and the loader take the name from one function instead of two, which
  is how they came to disagree.

## 1.44.0

- **The launch pad is what opens.** The canvas is one click away and always was,
  but arriving at an empty canvas tells you nothing about what this can do.
- **The assistant dock is about two and a half times the size** — 620 wide, up
  to 760 tall, with room to read a review rather than scroll one.
- **An Implicit domains page**, which is the detail that was missing.

Each domain is listed with its theorem and its source, and every arm gets a card
showing what it actually is: state width, trainable parameters, whether the
first parameter's transform is free or constrained, whether the state is
confined to the positive orthant, the residual expression itself, and either
*Guaranteed* with the claim or *No guarantee* with the condition that was
broken.

Three things worth saying about how it is built:

- **The facts come from the built layer describing itself**, not from the spec
  that asked for it. The page cannot claim a width or a parameter count the
  layer does not have.
- **The controls are shown as prominently as the covered arm.** They are the
  same size, the same shape and one condition apart; seeing them side by side is
  the whole point, because the guarantee is then the only difference.
- **Place this on the canvas** puts the arm down as a layer, so reading about a
  structure and using it are the same gesture.

*Check them all* runs the bench's validation across every registered domain and
reports each one's counts as it goes.

A test asserts every domain has exactly one covered arm, that all its arms cost
the same, and that each control says which condition it breaks — the page would
otherwise be able to show an unfair comparison and look fine doing it.

## 1.43.0

**The launch pad is a full page now**, and the assistant follows you around.

- **Say what you want and see what matches.** The box searches the 101 guided
  builds and ranks them as you type. It searches rather than generates, because
  searching is what can actually be delivered — a box that promises to build
  whatever you describe and then does not is worse than one that says what it
  does. When nothing matches it says so and offers the empty canvas.
- **Three worked things to try**, one starter, one working, one involved, taken
  from the guided builds rather than written out again.
- **Six routes**, each saying what it is for.
- **What is already in the workspace**, one click from opening.

**An assistant dock.** The assistant already lived in a tab beside the canvas,
which is the right place while designing and no place at all on the launch pad
or the projects list. It now also floats, reachable from anywhere, with the same
commands and a button to move it back beside the canvas.

It opens by saying what it is for: it reads the design and can change it, and it
does not invent networks from a description. Saying so up front is better than
letting somebody find out by asking.

**A bug the test caught before you did.** The dock posted `text` where the
endpoint reads `message`. A reply comes back either way — the wrong one, the
"ask me to change something" fallback — so it would have looked like a working
assistant that never understood anything. The check now asserts the field the
dock sends is the field the endpoint declares.

## 1.42.0

**A launch pad**, first in the sidebar: six routes in, each saying plainly what
it is for — design a network, learn how one works, bring a model in, see what it
actually does, train and compare, turn a paper into a layer. Below them, the
designs already in the workspace.

A door rather than a dashboard. Somebody arriving does not want a summary of
their account; they want to know what this can do and to be one click from doing
it.

- **A test asserts every route goes somewhere real** — that the page exists, that
  every function a route calls is defined, and that no route fakes a click on a
  control it does not own. The import route did exactly that, so opening the
  import dialog is now a function the app exposes rather than something that
  happens to a button.

**The bench is upgraded to 0.9.5.**

- **The Bergman glucose–insulin minimal model is now a placeable layer** — a
  1979 physiology paper, through the spec pipeline, into the palette. That is
  the loop working end to end on something that was never about neural networks.
- Twelve arms across five domains now, with `positive_rates`, `no_decay` and
  `signed_decay` joining.
- `paper_to_spec.py` comes with it: research question to candidate papers to
  spec skeleton, searching Europe PMC and arXiv. The library is here; it is not
  yet wired to the Paper page, which is the obvious next piece.

## 1.41.0

**The Paper → spec page has a real design**, and the reason it looked like that
was a one-word omission.

- **Fixed: the page painted no background**, so the dark page surface showed
  through and its heading was dark text on dark. Every other page sets one and
  this one was simply never added to the list.
- **A test now asserts every page paints its own background.** It immediately
  found a second one — the Studies page had the same gap, less visibly. This is
  the third time the dark surface has shown through something; it should be the
  last.

The page itself is now three numbered stages rather than two loose columns:

1. **Read it** — a drop zone that takes a PDF by drag or click and names the
   file once it has one, or a box for pasting the theorem.
2. **Write the spec** — Propose beside the editor, with its answer, refusal
   included, shown next to it rather than below the fold.
3. **Check it, then keep it** — Check, or Check and save, with the result in
   place. Before anything runs it says what the checking is for rather than
   sitting empty.

Every control was driven headlessly after the rewrite to confirm the handlers
still attach, since all the element ids changed.

## 1.40.1

**Fixed: the paper test failed on any install that has an account.**

It drove the endpoints over HTTP, and those routes require a signed-in user once
a workspace has one. My container had no accounts, so the test passed there and
failed everywhere real — the same class of mistake as the release archive
carrying my own state.

The test now calls the endpoint functions directly. Signing in would have meant
registering an account in the user's own data to test something that is not
about authentication; the auth behaviour is covered by its own checks. The suite
now passes both with accounts enabled and without, which is what I should have
verified before shipping it.

## 1.40.0

**Paper → spec, in the sidebar.** The last piece of the bench that was not here.

Drop a PDF or paste the theorem, read the ranked passages, write the spec, check
it, save it — and the ImplicitEquilibrium layer offers it as a domain
immediately, with no restart. A paper becomes a layer you can place on the
canvas.

The three stages are kept apart, because only one of them involves judgment:

- **Reading** ranks the passages that plausibly carry a structural result. It
  decides nothing; it means reading fifteen ranked passages instead of forty
  pages. A result that never says "unique" will rank low and still be the right
  one.
- **Proposing** is the judgment step and the one to trust least. Without an
  `ANTHROPIC_API_KEY` in the server's environment there is no proposer, and the
  page says so and gives the blank template — the same artifact by a slower
  route. A proposal that refuses is a correct output.
- **Checking** is what makes a proposal worth having at all. You write only the
  residual and the Jacobian is differentiated from it, so the two cannot
  disagree; the rest is physics, which the checks are for.

**A spec that fails is now withdrawn.** Loading a spec registers it, and the
first version left a broken one in the registry offering itself as a layer even
though its check had failed. Anything registered during a check is removed
unless it passes, and a test asserts the registry is left exactly as it was
found.

With this, everything from the bench except its own server and `runs.db` is
here: the domains as layers, the validation from the Layer tab, the ablation as
a study, and now the paper loop.

## 1.39.0

**The ablation runs from Studies: "Does the guarantee help?"**

The bench compares a domain's arms on its own fixed model, which answers whether
the structure helps in general. Run here it answers a different and often more
useful question: whether it helps in the architecture you actually have. Each
arm is your design with one layer's variant swapped, so the parameter counts
match by construction — every trial is an ordinary run, with its own history and
checkpoints, and can be reopened on the canvas.

- The covered arm is badged **covered** in the results table, and each row says
  which condition it satisfies or breaks.
- **A verdict that does not overclaim.** Beating every control reads as the
  guarantee buying something. Beating none says it cost nothing and bought
  nothing. Beating *some* says so and names the control it lost to: "beating one
  control is not beating the condition — whatever that one preserves may be what
  actually matters."

On a first run of the convex domain the covered arm held at 0.693 while the
`flat` control — the same layer with α set to zero, no longer strongly convex —
diverged to 8e10. That is what losing a uniqueness guarantee looks like from the
outside.

**Training in double precision, which the above needed.**

The study failed three times before it ran, and each failure was worth fixing
rather than working around:

- The model is now built in **float64 whenever any of its parameters are**, since
  a layer whose solver runs to 1e-10 cannot be single precision. Warning about
  it in the Needs panel was not enough — the study could not run at all.
- Two helpers used `torch` at module level, where this file imports it inside
  functions. A plain `NameError`, and a reminder that a convention in a file is
  worth reading before adding to it.
- **The validation batches were still float32** after the training batches were
  fixed, so the first epoch trained and then died at evaluation. A test now
  asserts no batch path is left unconverted, rather than trusting that I found
  them all by eye.

## 1.38.0

**The bench's own checks, from the Layer tab.** Select an ImplicitEquilibrium
and press **Check the mathematics**.

*Test layer* asks whether the canvas's arithmetic matches torch. This asks
something harder: whether the layer's mathematics agrees with itself. It runs
the bench's `validate_domain` — the analytic Jacobian against autograd, the
implicit gradients against finite differences, the equilibrium from several
starting points, conservation, scale robustness, training stability — and shows
every check with its measured number.

On the reaction-network domain that is 21 checks, including the Jacobian
matching autograd to 1.85e-17 and the solution staying in its compatibility
class to 4.81e-16.

This is worth having in the designer rather than only in the bench because a
Jacobian with a sign error still converges. The solver looks fine, the loss goes
down, and the gradients are quietly wrong — which the bench's own notes record
as having happened once already.

**What is and is not here.** The `dnn_bench` library is vendored whole,
including `sweep`, `verdict` and `validate`. Its server, its browser UI and its
`runs.db` are not: the standalone bench remains its own application, and the
ablation sweep is not yet wired to Studies.

## 1.37.0

**Every implicit domain is now a layer**, from the 0.7 bench, vendored as
`dnn_bench/` with its `specs/`.

One layer, **ImplicitEquilibrium**, offers each domain and each of its arms:

| domain | where the guarantee comes from | covered arm | controls |
|---|---|---|---|
| `crn` | the topology of a reaction graph | weakly reversible, δ=0 | edge reversed; deficiency 1 |
| `contraction` | a constraint on the weights (monDEQ) | margin 0.05 | margin 0; W free |
| `convex` | the sign of a coefficient vector | coefficients ≥ 0, α>0 | α=0; coefficients free |
| `convex_ridge` | the same, declared as a spec rather than coded | | |

Three different mechanisms buying the same certificate, which is the point: if
the structured arm only separates in one of them, the separation is about that
mechanism rather than about having a guarantee at all.

- All **twelve** domain-and-arm combinations were built through the designer:
  shapes resolve, the canvas parameter count matches torch exactly, and the
  generated file runs.
- **A test asserts the arms within a domain cost the same.** Matched counts are
  what make a control a control; if they drifted the comparison would be
  confounded and nothing else would notice.
- **The Maths tab names the theorem and says whether your arm satisfies it** —
  "theorem applies: yes", or "no — this is a control", with the paper it came
  from.
- **The Needs panel warns when a control is selected**, naming the layer and the
  arm. Choosing one is legitimate and is how you find out whether the guarantee
  was buying anything; it should just never be an accident.
- Domains declared as JSON in `specs/` register alongside the coded ones and
  differentiate their Jacobian from their residual, so the two cannot disagree.

The earlier `EquilibriumCRN` stays: it exposes the reaction graph's shape —
species, linkage classes, extra edges — which the general layer does not.

## 1.36.0

**The Deficiency-Zero equilibrium layer is now a layer you can place**, under a
new *Implicit* category, from work done separately and vendored unchanged as
`crn_deq.py`.

It does not compute its output; it solves for it. The input seeds a chemical
reaction network and the layer returns where that network settles. What makes it
worth having is where the guarantee comes from: most implicit layers need a
constraint on the weights — a spectral norm, a monotonicity margin, a projection
after every step — to be sure a fixed point exists and is unique. The Feinberg /
Horn–Jackson Deficiency Zero Theorem gives it from the *topology of the graph*
instead, so the rates train as `k = exp(θ)` with θ free over all of ℝ and no
projection anywhere.

- Shapes, parameter counts and generated code work as for any other layer:
  a test builds a design containing one and confirms torch agrees with the
  canvas exactly, and that the file runs.
- **The Maths tab carries the theorem** with this graph's own integers:
  `δ = m − l − s = 6 − 2 − 4 = 0`, and says whether the graph is weakly
  reversible, since the guarantee needs both.
- Setting deficiency to 1 keeps the parameter count and drops the guarantee,
  which is what makes it a comparison rather than a smaller model.
- **The Maths tab animates it** as a trajectory spiralling into its fixed point
  from anywhere — the diagram the theorem describes.
- **The Needs panel warns that this design must train in float64.** The solver
  runs to 1e-10, which single precision cannot reach, so it is a property of the
  layer rather than a preference — and without the warning the first batch fails
  on a dtype mismatch with nothing to explain it.

**A wrong number caught before shipping.** The maths panel first printed
`δ = m − l − s = 6 − 2 − 6 = 0`, which does not add up: I had used the length of
the stoichiometric basis, which is the number of species, where the rank was
wanted. A test now checks the printed arithmetic actually holds across several
graphs — a displayed equation that does not balance is worse than no equation.

## 1.35.0

**An End to end tab**, beside Maths: a narrated walk through what happens to the
data, one layer at a time.

Each step gives what arrives, the mathematics applied, what leaves, and what the
values actually look like afterwards — measured from one real example passing
through, not described in the abstract. A ReLU is not said to remove the
negatives; it is run, and the step reports that 49% of the tensor is now exactly
zero and the minimum is 0.

- **Back, Next and Play** step through it; the canvas highlights whichever layer
  is being read about.
- **The closing step reads the output the way the task means it.** A classifier's
  logits become the top classes with their probabilities. A model that scores
  every position in a sequence is recognised as such, and the last position's
  distribution is presented as the next token it would choose. A regression head
  keeps its numbers and says they are the prediction, not a score. Which reading
  applies comes from the Output layer, so this works for whatever is on the
  canvas rather than for one architecture.
- Token indices are reported as a range and a count of distinct values, never
  averaged — the mean of a set of token ids means nothing.
- A layer with no module of its own says its values were not captured, rather
  than showing a blank that reads as nothing having happened.

Tested on a small classifier, a sequence model and a regression head, and on the
GPT-2 block from the examples, which walks through in 23 steps.

## 1.34.0

**Run a batch and watch it happen.** A new **Run** tab, and a rail entry beside
Train.

A workflow engine can show tasks going green because they execute one at a time
and report back. A forward pass has the same shape — modules run in order, each
taking a measurable time and producing a definite tensor — so this is measured
with hooks on the generated model rather than animated from a guess.

- **The canvas follows the run.** Each layer takes a green border and a tick as
  the pass reaches it, with what it actually cost printed beside it. Layers not
  yet reached are dimmed.
- **Layers**: a table of every layer in execution order with its time, a bar
  showing its share, the shape it produced and the memory that holds. Clicking a
  row selects that layer on the canvas.
- **Timeline**: the same run as a Gantt, so a layer that dominates is obvious at
  a glance.
- **This run**: batch size, total time, the slowest layer and its share, total
  activation memory, parameters, and the output shape.
- **Timed after a warm-up.** Unwarmed, a small convolution reports several
  hundred milliseconds of one-off kernel setup and looks like the bottleneck it
  is not — on a test network that was 766ms against a true 0.26ms. Reporting the
  first number would send you optimising the wrong layer.
- A layer with no module of its own — an Add, a Flatten in some graphs — is
  listed as untimed rather than as taking zero, and never dropped from the
  table: a missing row reads as a layer that failed.

## 1.33.3

**One line between a panel and the canvas, and it is the one you drag.**

There were three. The panel drew its own border. The drag handle drew an amber
bar when you pointed at it. And between them sat a 5px transparent splitter,
through which the dark page background showed as a black rule — the app was not
drawing that line at all, it was a gap.

- The splitter is the divider now: it matches the canvas, carries a single
  hairline down its centre, and thickens to amber under the pointer or while
  being dragged. The panels no longer draw an edge of their own.
- The same for the horizontal splitter above a bottom-docked panel.
- Every gap in the layout is painted, so nothing can reveal the page background
  again.
- Dragging is unchanged and verified: the width follows the pointer and still
  clamps between 150 and 760.

## 1.33.2

- **A folded panel now leaves nothing behind.** The 26px strip that held a
  chevron was a second control for a job the rail already does — Layers and
  Files open the left panel, and Code, Maths, Train, Assistant or Checks open
  the right one. It also occupied room in exactly the state the fold exists to
  clear.
- **Asking for a tab reopens its panel.** Pressing Code in the rail while the
  right panel is folded now unfolds it rather than doing nothing, which is what
  removing its strip requires.

## 1.33.1

**Fixed: pressing Layers or Files blanked the whole canvas.**

The rail's click handler was bound to every button in the rail. The new Layers
and Files entries carry no `data-page`, so pressing one called
`showPage(undefined)`, which removed the display class from every page and added
it to none. With no page shown, the dark page background appeared where the
canvas should be — and because the palette was toggling correctly underneath, it
looked like the panel was coming and going at random.

- All seven rail handlers are scoped to entries that actually name a page.
- A test asserts none of them is bound to the whole rail, and that a palette
  entry never carries `data-page`. Verified against the broken version: it
  reproduces the empty screen exactly.

Nothing was wrong with the markup, the styling or the folding logic, which is
why it took a reproduction rather than a reading to find.

## 1.33.0

- **Layers and Files are entries in the left sidebar**, under Definitions with
  Canvas and Code, each carrying a chevron. Pressing one opens the palette on
  that section; pressing the one already showing folds it away, so the same
  control does both — which is what a chevron promises.
- The palette reads as part of the sidebar rather than a second column beside
  it: no border between them, and the rail entry stays lit while its section is
  open.
- Folding the palette any other way — its own chevron, its strip, a dock change —
  updates the rail entry too, so the two can never disagree about what is
  showing.
- Docking, resizing and folding are unchanged, and the test that holds all three
  together still passes.

## 1.32.1

- **The heavy black rule down each side of the canvas was a scrollbar.** Only
  the thumb had ever been styled, so the track fell back to the browser default —
  which on a machine set to dark mode is nearly black. The page now declares
  itself light and styles the track, the corner and the Firefox equivalent, so
  no control renders in a colour the app never chose.
- Panel edges are a hairline plus a soft shadow rather than a hard border, so
  the canvas reads as sitting under the panels rather than being fenced off.
- **The toolbar floats over the canvas**, top left, instead of occupying a band
  across the window — which puts the header on one line, as it should be.
- **Tabs read as tabs**: sentence case at 12px, light grey when idle, near-black
  and semibold with a blue rule when active.
- **Fixed while checking the above**: `applyLayout` redrew without reapplying
  the fold state, so re-docking or resizing a folded panel silently brought it
  back. Folding is part of the layout, and a test now holds docking, resizing
  and folding together.

## 1.32.0

- **Both side panels fold away.** A chevron in each panel's header collapses it
  to a 26px strip carrying the chevron that brings it back, so the canvas can
  have the window without the controls becoming unreachable. The right panel's
  tab row carries the same control at its left, where a tab strip usually does.
- The chevron points the way the panel will actually move: `«` for a panel on
  the left, `»` on the right, `▼` and `▲` when docked at the bottom. A folded
  panel no longer holds the bottom row open, and its splitter goes with it.
- Folding is remembered with the rest of the layout.
- **The canvas toolbar is a floating card** — rounded, shadowed, sitting over
  the canvas rather than as a bar across it, with the controls grouped by
  separators and the zoom shown as a number.
- A **home** button resets the view to the origin at 100%, separate from *fit*,
  which frames whatever is drawn.
- The active tab is underlined in blue rather than amber, which reads as
  selection rather than as a warning.

## 1.31.3

- **`GUIDE.md`** — what the app is, everything it can do, what is unusual about
  it, and a beginner's first half hour. Every figure in it was checked against
  the code rather than written from memory.

## 1.31.2

**Fixed: the repository box suggested exactly the address you wanted.**

Its placeholder read `https://github.com/karpathy/minGPT` — grey, so the field
was empty, but it looked filled in. Pressing Fetch then answered "paste a
repository address" about a box that appeared to contain one.

- The placeholder now reads *owner/name, for example karpathy/minGPT*, which
  cannot be mistaken for an entered value.
- **`karpathy/minGPT` works as typed**, without the address around it.
- The folder box had the same trap with `/path/to/project`; it now describes
  what to enter rather than showing something that looks entered.
- **A test rejects any placeholder that is a usable value** — no URLs, no
  paths. It found the folder box, which I had not noticed.
- The shorthand introduced a hole the same test caught:
  `github.com/onlyowner` parsed as a repository called `onlyowner` owned by
  `github.com`. Refused now, with a message saying what was missing.
- The hint says where the download lands (`data/github/`) and that the classes
  appear below.

## 1.31.1

**Fixed: Import fell through to the architecture dropdown when nothing was
chosen.**

With no class ticked and nothing pasted, pressing Import fetched resnet18 —
because the dropdown always held a value, so "nothing chosen" was not a state
the dialog could be in. The failure that surfaced was then about torchvision,
which had nothing to do with what the person was doing.

- The architecture list starts at **— none —**, so choosing one is a decision.
- Import considers every route before doing anything, and says which is missing:
  *"Press Fetch first, then tick the classes you want"* when a repository
  address is typed but not fetched, *"Tick at least one class"* when a scan is
  showing, and a plain list of the routes when nothing at all is filled in.
- **A missing package says how to install it**, naming the interpreter:
  `/path/to/.venv/bin/python -m pip install torchvision`. Anything else is
  passed through unchanged rather than dressed up.

## 1.31.0

**Import straight from GitHub.** Paste a repository address into the Import
dialog and press Fetch; it is downloaded, unpacked and scanned, and the classes
it holds appear in the same browser as a local folder.

- Accepts what people paste: `github.com/karpathy/minGPT`, the full https
  address, a `.git` suffix, or `/tree/<branch>/<folder>` to go straight to one
  part of a repository. Tries `main` then `master` when no branch is named.
- Downloaded once and cached under `data/github/`, so a second look is instant.
- Archive members whose paths point outside the unpack directory are skipped, as
  are symlinks — an archive can name anywhere on disk, and this one is coming
  from the internet.
- Downloading and reading run nothing. Execution happens only when you pick a
  class to import, which is the same rule as for a folder already on disk.

**A setup box**, because one expression is not a configuration. minGPT wants a
default config with half a dozen fields set on it before anything can be built.
Whatever the setup defines can be named in a row's argument box.

### minGPT, measured

`NewGELU` imports cleanly. `CausalSelfAttention`, `Block` and `GPT` build
correctly with the setup box and then refuse to trace, naming the reason: the
attention slices its mask with `self.bias[:, :, :T, :T]`, where `T` came from
another tensor. That is the same limit older GPT-2 code hits, and the same code
written with `F.scaled_dot_product_attention` imports without trouble.

## 1.30.3

The check added in 1.30.2 did its job and found the rest of the problem:
`.seeded` was **committed** in an earlier release, and adding a name to
`.gitignore` does not untrack a file already in the repository. So it would have
kept shipping.

- `release.sh` untracks workspace state before it tests and tags, so this cannot
  recur through inattention.
- The test's message now names the files and gives the command to fix them,
  rather than only reporting that something is wrong.

To clear it on a machine that has already committed one:

    git rm --cached .seeded && rm -f .seeded

## 1.30.2

**Fixed: the test added in 1.30.1 failed on any machine actually running the
app.**

It asserted that `.seeded` and `prefs.json` were not on disk. That is right for
a clean checkout and wrong for a working install, where those files are supposed
to exist — so the check reported a problem on precisely the machines that had
none.

It now verifies what it meant to: that such files are git-ignored and untracked,
and that the marker sits with the designs rather than at the project root.
Absence from disk was never the property worth testing; exclusion from the
release is. Verified with those files both present and absent.

## 1.30.1

**Fixed: MicroGo could not arrive, because the release said it already had.**

The file recording which examples a workspace has been given sat at the top of
the project, next to the source — so it was swept into the release archive.
Unpacking it overwrote the receiving machine's own record with mine, which
listed MicroGo as delivered. The app then correctly declined to deliver
something it had been told was already there.

- The marker lives in `saved/` now, beside the designs it describes, where a
  release cannot pick it up. An existing marker at the old location is moved
  automatically.
- It is git-ignored, along with the trained weights `play.py` writes.
- **A test asserts a release carries no workspace state**, checking both the
  files and the ignore rules — the packaging step was the only thing standing
  between my machine's state and yours, and it was not enough.

Verified against a workspace in exactly the state this broke: old marker, no
MicroGo. It arrives, the old marker is migrated, and an example deleted on
purpose still stays deleted.

## 1.30.0

**A micro AlphaGo: the rules, the search, the network — and an honest account
of what came out of it.**

- **`microgo.py`** is Go on a small board with the rules done properly:
  liberties, single and group capture, suicide refused, the ko rule, two passes
  to end, and Chinese area scoring with komi. Tested against positions worked
  out by hand.
- **PUCT search**, the AlphaZero kind: priors and a value from an evaluator,
  values negated up the tree, visit counts returned as the policy target.
- **`MicroGo`** is an example design — a convolutional trunk with residual
  blocks feeding a policy head over 26 actions and a value head, 129,589
  parameters, matching PyTorch exactly.
- **`python play.py`** runs self-play, trains on it, and reports games won
  against a baseline, every game scored by the rules engine.

### What actually happened

Self-play training **did not produce a stronger player**, and the reason is
measurable rather than mysterious. Against a random-move opponent the untrained
network won 74/100 and the trained one 18/100 — training made it worse.

The search is too weak to teach. At 40 simulations the visit counts over 25
legal moves have an entropy of 2.96 where uniform is 3.22; at 800 simulations it
is still 2.88. The policy target is very close to noise, and training a
reasonable initialization toward noise degrades it. AlphaZero's numbers are
hundreds of simulations per move and hundreds of thousands of games; this is
three orders of magnitude short, and the shortfall shows up exactly where the
theory says it would.

Two real bugs were found and fixed on the way, both of which would have made
things worse regardless of budget:

- **BatchNorm in a reinforcement-learning loop.** Positions within a self-play
  batch come from one game and are highly correlated, so the running statistics
  described nothing — and `eval()` then used them, meaning the network the
  search consulted was not the network that was trained. `ResidualBlock`,
  `PolicyHead` and `ValueHead` take a `norm` choice now (batch, group or none),
  and `MicroGo` uses group.
- **A policy that learned to pass.** Passing is legal in every position, so
  offering it at every node gave it prior mass everywhere, and the greedy policy
  passed on move one. The search now offers it only when there is nothing else
  to play — which is also when a person would consider it.

The machinery is verified. The player is not strong, and this says so.

## 1.29.1

The Code tab shows the whole generated file — every sheet, every class — because
that is what you would save and run. It does not change per layer, and there was
nothing to say so.

- **Selecting a layer now jumps to its line** when the Code panel is open, the
  way the Layer tab's *Show in Code* button always did. Selecting another layer
  moves to that one.
- **The header says where you landed** — `line 13 in Block`, `line 57 in Model` —
  because a workbook generates several classes and a line number alone does not
  place you.
- Verified across the MicroLean design: 21 of its 23 block layers locate their
  line, the two that do not being Input and Output, which have no constructor.

## 1.29.0

**A micro proof assistant, and a network that learns to drive it.**

Lean itself is not a neural network — a trusted kernel, an elaborator,
dependent types and a tactic framework cannot be made of layers, and pretending
otherwise would be the wrong answer to the question. What *is* learnable is the
part a prover cannot do alone: choosing the next step. That is what this adds.

- **`microlean.py`** is a small proof kernel over equational logic: terms,
  twelve axioms, one-way matching, rewriting at a position, and `check`, which
  replays a proof and decides whether it holds. It is the only thing in the
  project that decides truth, and it does not trust the model.
- **A tactic space of 84** — twelve rules by seven positions — so every
  prediction is a legal thing to attempt and the head is an ordinary classifier.
- **Theorems are generated by walking backwards** from a random term, which
  guarantees a proof exists and hands you that proof as the label. Every one is
  verified before it is used for training.
- **`microlean` is a dataset** in the Train tab: proof states in, tactics out.
- **`MicroLean` is an example design**: a three-block transformer built from the
  app's own layers — Embedding, LearnedPositions, RMSNorm, Split, Transpose,
  Attention — 609,620 parameters, matching PyTorch exactly.
- **`python prove.py`** trains it and reports proofs the kernel accepted.

On theorems that never appear in training, after six epochs:

| | 1–3 steps | up to 5 steps |
|---|---|---|
| random policy | 3/120 (2%) | 3/120 (2%) |
| trained policy | 119/120 (99%) | 114/120 (95%) |

**The first measurement was wrong and worth recording.** A differently-seeded
sample scored 120/120, which was too good. It shared **32%** of its theorems
with the training set — the reachable space is small enough that independent
seeds collide. `corpus` takes an `exclude` set now, the numbers above are from
genuinely disjoint theorems, and a test asserts that naive sampling still
overlaps, so the guard cannot quietly stop being needed.

## 1.28.2

**Fixed: "will not trace" was shown for every kind of failure.**

A class that was never given its config got the same badge as one whose
`forward` genuinely cannot be traced. Only the first is the user's to fix, and
it is fixed by typing in the box on the same row — so the label was sending
people to look at the wrong thing entirely.

The badge now says which happened: **needs arguments**, **cannot be traced**,
**import failed**, **shapes unresolved** or **not found**, with the full reason
on hover. A class that needs a config and has not been given one shows its box
in amber before you press anything.

## 1.28.1

- **The import dialog resizes.** Drag the corner grip to make it as large as the
  window allows; the class list and the source preview both grow with it, so a
  38-line class can be read without scrolling in a strip.
- **Drag it by its header** to move it out of the way of the canvas behind.
- **The split between the list and the source is draggable**, for when the
  source is what you are reading rather than the names.
- The size, position and split are remembered with the rest of the panel layout.
  A size remembered on a large screen is clamped when the dialog opens on a
  smaller one, rather than opening off the edge.
- **Fixed: the argument boxes stretched across the whole row**, squashing every
  class name to a single letter. The dialog's generic field rule sets every text
  input to full width and outranked the class on those boxes.

## 1.28.0

Scanning a model library returns hundreds of classes, so the Import dialog is a
browser now rather than a list.

- **The dialog is 920px wide** when importing, with a layered shadow — a tight
  contact shadow, a mid one and a wide soft one — instead of a single heavy
  drop, and a slight blur behind it.
- **A filter** across every class found. Scanning the whole `transformers`
  models tree finds 691; typing `MLP` narrows it immediately.
- **Hover a class and see it**: its `__init__` signature, its docstring, how many
  lines it is, its methods, and the source itself with syntax highlighting. Read
  by syntax tree, so the preview ends where the class ends.
- **A Try button per class.** Most classes in a library will not trace, and
  finding that out should not mean committing to an import. Try reports layers,
  opaque nodes and parameter count on success, and the specific reason on
  failure.
- Classes needing constructor arguments get a box on their row, and those
  arguments are evaluated **where the class lives** — so `LlamaConfig(hidden_size=256)`
  resolves without you spelling out the import.

**Two fixes that made the above work at all:**

- A file inside an installed package is now imported by its dotted module name,
  so its relative imports resolve. Every `transformers` module failed with
  "attempted relative import with no known parent package" before this.
- Verified against the real library: `LlamaMLP`, `Qwen3MLP` and `DeepseekV3MLP`
  all import as 7 layers with nothing opaque and 393,216 parameters exactly.

## 1.27.3

**Fixed: scanning a folder inside a virtual environment read nothing.**

The scanner ignores `.venv`, `site-packages`, `node_modules` and the like so
that pointing it at a project root does not crawl into the environment. It
judged the whole absolute path, so a folder that merely *sat* inside one had
every file skipped — "Read 0 files and found no nn.Module classes", with nothing
to say why.

Those names are now matched only *below* the folder you gave. Pointing
deliberately at a library inside a virtual environment works; pointing at a
project still leaves its environment alone. Both are tested.

## 1.27.2

**Fixed: the Import dialog's Scan button ran the wrong scanner.**

Two functions were called `scanFolder` — the one that reads Python files for
model classes, and an older one that counts images in a training dataset folder.
The second definition replaced the first, so pressing Scan with a path typed
into the Import dialog read a different, empty field and answered "Type a folder
path first".

- The code scanner is `scanCodeFolder` now; the dataset one keeps its name.
- **A test that no two functions on the page share a name.** Every existing
  check passed on this bug, because nothing was undefined — the wrong thing was
  defined twice, which is invisible to a check for missing definitions.

## 1.27.1

`doctor.py` now answers the question behind "but I installed it".

- It names **which interpreter it checked**, and says so explicitly when that is
  a virtual environment — where anything installed outside does not count.
- When a package is absent, it looks for it on the other interpreters on the
  machine and reports **"it is installed for /usr/bin/python3 — but not here"**,
  which is almost always the real situation.
- Install lines use `<that interpreter> -m pip install`, so the packages land
  where the check looked rather than wherever `pip` happens to point.

## 1.27.0

- **`python doctor.py`** reports what this machine has and what each thing
  unlocks — Python version, the three packages the server needs, and the
  optional ones with a plain statement of what is lost without each. Nothing is
  fatal except the first three, so it describes rather than alarms.
- It also reports what torch can train on (CPU, CUDA, Apple GPU), how many model
  families the installed `transformers` makes readable and the exact path to
  scan them from, whether the storage directories are writable, free disk space,
  and which accounts exist.
- Tested with the optional packages hidden as well as present, because a check
  that has only ever seen a working machine is not a check.

## 1.26.1

Tested against the real Hugging Face `transformers` library rather than
reimplementations of it, which changed some of what 1.26.0 claimed.

**Confirmed against library code**, with parameter counts matching exactly:
`LlamaMLP`, `Qwen3MLP`, `DeepseekV3MLP`. `LlamaRMSNorm` and `DeepseekV3RMSNorm`
import with nothing opaque, and the parameter check correctly reports their
learned scale as belonging to no layer.

**Confirmed as out of reach**: every attention module in the library, and
`MixtralSparseMoeBlock`. Their `forward` takes `**kwargs`, which `torch.fx`
cannot trace through. Wrapping them does not help — the same pattern appears
inside, which I found by trying it.

- **Refusals now name the obstacle** instead of offering one generic guess.
  A length taken from another tensor, more than one required tensor argument,
  and a variadic signature are three different problems, and only some have a
  way round. Each says which one it hit.

## 1.26.0

Work on importing larger language models, prompted by asking whether something
like DeepSeek could be analysed. Measured rather than assumed, on faithful
implementations of the pieces that make those models distinctive.

**What now imports cleanly**

- **Multi-head latent attention** — the low-rank KV compression DeepSeek-V2
  introduced — comes through with nothing opaque: 17 layers, shapes resolved.
- **SwiGLU** feed-forwards: 7 layers, nothing opaque.
- **RMSNorm** written by hand as `x * rsqrt(x.pow(2).mean(-1)) * weight` now
  resolves, via three new layers: `RMSNorm` itself for building with,
  `Elementwise` (pow, rsqrt, sqrt, exp, log, abs, neg) and `Reduce` (mean, sum,
  max, min over an axis). A whole dense block — norm, latent attention, SwiGLU,
  residuals — imports as 32 layers with nothing opaque and runs.
- Fixed: `mean` was being read as global average pooling even when given an
  explicit axis, which is a different operation entirely.

**What is now refused rather than misdrawn**

A mixture of experts cannot be honestly drawn as a fixed graph. `torch.fx`
unrolls the loop over experts, so all of them appear as though all of them run —
and the parameter count comes out at 158,613,504 for a model holding 89,411,584.
The diagram is not wrong about the code; it is wrong about the model, which is
worse.

- Operations that route at run time — `topk`, `where`, `scatter`, `gather`,
  `nonzero` and others — are recognised, and an import containing them says
  plainly that every branch is drawn as though it always runs.
- **Imports now check their own parameter count** against the model's. A gap
  means a learned tensor is used in plain arithmetic rather than through a
  layer, so it belongs to no node — that is now reported with the exact
  difference instead of quietly undercounting.

## 1.25.1

**Fixed: the GPT2 model sheet drew almost nothing.**

It was an error, not a slow render. The continuation label on a Subgraph node
read a variable declared later in the same function — `Cannot access 'inset'
before initialization`. Rendering stops at the first exception, so the first
block came out as an empty white box and the remaining sixteen layers never
drew at all: arrows pointing at nothing, with stale shape labels left over from
the previously rendered sheet.

Nothing was wrong with the design, the analysis or the generated code. Only the
drawing.

- The label moved below the declaration it depends on, and now replaces the
  node's subtitle rather than being an extra line.
- **A test that renders a real sheet.** It drives the page's own `render()` over
  the GPT2 model sheet in Node and asserts every layer produced a title.
  Verified against the broken version: it fails with the exact ReferenceError
  and reports 2 of 16 layers drawn.

That last part is the point. Every existing check passed on the broken build —
they tested the analysis, the code generation and the parameter counts, all of
which were correct. Nothing tested that the canvas actually drew.

## 1.25.0

- **Open is a list you click.** It was a browser prompt that showed you the
  names and then made you type one of them into a box, which is a strange thing
  to ask of someone who is looking straight at the list.
- Each row shows the design, how many versions it has and when it was last
  saved. Designs with more than one version get a version selector on the row,
  so an older one can be opened without a second dialog.
- Search filters as you type; arrow keys move; Enter opens; Escape closes;
  clicking outside closes.
- Delete is on the row, behind a hover, rather than being a separate flow.
- An empty workspace offers the example designs as a button rather than a
  yes/no question.
- Opening a workbook restores its sheets properly — the old handler predated
  sheets and would have flattened a multi-sheet design like GPT2 to whichever
  arrays happened to be at the top level.

## 1.24.1

**Fixed: GPT2 would not have appeared for anyone already using the app.**

Seeding recorded a single "this workspace has had the examples" flag, so a
workspace set up when three examples existed never received a fourth. GPT2
shipped in 1.24.0 and was invisible to every existing account — which is most of
the point of shipping it.

- The marker now records **which** examples have been delivered, not merely that
  some were. Anything newly shipped arrives; anything deleted stays deleted,
  because the record is of what has been offered rather than what is present.
- Old markers are understood: for those, whatever is in `saved/` is taken as the
  record, which is correct — at that point the only way an example could be
  there is that it was delivered.
- Delivery happens when the design list is opened, so it reaches accounts that
  already existed rather than only new ones.

## 1.24.0

**GPT-2 ships as an example design.** Open `GPT2` and it is there as a workbook:
the transformer block on one sheet, imported from real code, and the model on
another — embeddings, twelve `block` references, final norm, language-model
head. 163,037,184 parameters, which is what PyTorch reports for the same
architecture, and the generated file runs.

- **A `LearnedPositions` layer**, since attention is blind to order and GPT-2's
  learned position table had no equivalent here.
- Select any layer in the block and the Maths panel derives it with GPT-2's own
  numbers: attention showing √64 ≈ 8, the 12 × 64² = 49,152 scores held at once,
  the quadratic cost of doubling the context.

### A correctness bug this found, and fixed

Reconstructing `q, k, v = t.split(n, dim=2)` gave **all three branches piece
zero**. fx models that line as one split followed by three getitems, and the
importer emitted the Split when it saw the split — so every branch took index 0.
The shapes were right. The parameter count was right. The network computed
something else entirely.

Splits are now recorded and a node emitted per selection, carrying the index it
actually takes. Two tests came out of it: one asserting the branches take pieces
0, 1 and 2, and one that loads an original model's weights into the
reconstruction and compares outputs — because matching shapes is not the same as
matching behaviour. On the GPT-2 block the difference is now exactly zero.

## 1.23.0

**Transformer blocks import cleanly.** A GPT-2 block, written the way nanoGPT
writes it, now arrives as 21 layers with nothing opaque and every shape
resolved — 7,087,872 parameters, matching PyTorch exactly.

Before this it was 26 nodes with 11 of them marked "no layer equivalent", which
told you nothing about the architecture.

- **Shape bookkeeping is no longer drawn.** `B, T, C = x.size()` and
  `C // self.n_head` are arithmetic the author does in order to reshape. They
  were appearing as layers, burying the architecture. They are tracked so they
  can be left out rather than guessed at.
- **Pass-through operations are elided** — `.contiguous()`, `.detach()`, `.to()`
  add a node and say nothing.
- **Three new layers for what was left**: `Transpose` (swap two axes),
  `Split` (cut into equal pieces, which is how one projection becomes Q, K and
  V), and `Attention` (`softmax(QKᵀ/√d_k)V` itself, holding no weights — the
  projections around it hold them). All three have full entries in the Maths
  panel.
- **Reshape targets are measured, not reconstructed.** The import now runs one
  example through the traced model and reads the real shape of every
  intermediate, so `view(B, T, h, C // h)` becomes `[64, 12, 64]` rather than a
  guess.

### What still cannot be imported

A model whose `forward` slices by a traced value — the classic
`att.masked_fill(self.bias[:, :, :T, :T] == 0, -inf)` in older GPT-2 code —
cannot be traced into a graph at all, and says so. The same model written with
`F.scaled_dot_product_attention` imports perfectly. That is a property of
`torch.fx`, not a limitation this app can wish away, and the refusal names the
reason.

## 1.22.2

- **Maths is in the sidebar**, under Definitions beside Canvas and Code. It was
  reachable only from the right panel's tab strip, which is a lot of tabs to
  scan. Grouped with the design rather than with Status, since it explains what
  the network does rather than reporting on its condition.

## 1.22.1

**Fixed: `mathbook.py` would not parse on Python 3.11, so the server would not
start at all.**

Backslashes inside an f-string expression — `f"{'\u00d7'.join(parts)}"` — only
became legal in Python 3.12 (PEP 701). I wrote the maths panel on 3.12, where it
imports fine. On 3.11 it is a `SyntaxError` at import, which takes the whole
application down before it serves anything.

- Four of them, all in `mathbook.py`. The escapes are named constants now and
  the shape formatting went into a helper. The panel's output is unchanged.
- **A test that catches this whichever version runs it.** It walks the syntax
  tree of every module looking for backslashes inside f-string expressions, so a
  3.12 machine can still find a mistake that only breaks 3.11. Verified against
  the broken file: it fails and names the file, line and expression.
- Also swept for other 3.12-only syntax — PEP 695 generics, `itertools.batched`,
  same-quote nesting in f-strings. None present.

`README` now states the minimum as Python 3.11.

## 1.22.0

**A Maths tab.** Select any layer and see what it actually computes — with this
network's numbers in it, not a textbook's.

Four parts, for every layer:

- **The equation**, in the notation the literature uses.
- **What each symbol is**, bound to this node: not "W is the weight matrix" but
  "W is 64 learned filters, each 3×3×3".
- **The arithmetic worked through.** Not `1,792 parameters` but
  `(C_in/g)·k²·C_out + C_out = 3×9×64 + 64 = 1,792`, and
  `⌊(32 + 2 − 2 − 1)/2⌋ + 1 = 16` for the output size. You can see where every
  number came from, which is the difference between reading a result and being
  able to change it.
- **Where the freedom is** — what can be varied and what varying it does to the
  mathematics. Dilation widens the receptive field at no parameter cost; groups
  divide the parameters and at g=C_in become a depthwise convolution;
  factorising W ≈ UV is cheaper below a rank this panel computes for you.

**An animated diagram per operation family**, drawn rather than illustrated: a
convolution window walking the input while output cells fill behind it, a batch
of activations sliding to zero mean, the actual curve of whichever activation
you picked with a point travelling along it, an attention score matrix filling
in, a lookup table returning one row.

Covers convolution, pooling, dense, normalization, activation, attention,
recurrent, embedding, dropout, merges and reshapes. **A layer whose mathematics
is not written up says so** rather than producing something plausible — that
seemed the only honest option for a panel meant to be trusted while you tweak.

## 1.21.0

- **Every account starts with the example designs.** They used to live only in
  the first account's workspace, so anyone registering afterwards arrived at an
  empty app. They are shipped assets now, in `examples/`, copied into each new
  workspace when it is created.
- **Copied once, at registration.** Deleting them is a decision, not an accident
  to undo on the next sign-in, so a workspace that has been seeded is never
  seeded again.
- **Asking for them back is easy.** Opening a design in an empty workspace
  offers to add them, and `POST /api/examples/restore` does it directly.
  Restoring skips any name already present, so it can never overwrite your own
  work — verified with a design saved under an example's name.

## 1.20.1

**Fixed: signing out looked exactly like losing everything.**

Nothing was ever deleted. The page still loaded, but every API call answered
401, so the app drew itself with no layers, no designs and no projects — and
said nothing about why. That is an alarming thing to show someone about their
own work.

- A missing or expired session now shows the sign-in box over a locked page
  instead of an empty one, and says plainly that the designs, runs and studies
  are still on disk.
- Every API call goes through one interception point, so a session that expires
  mid-session raises the sign-in box rather than quietly blanking the screen.
- Startup stops before loading anything when there is no session, rather than
  firing a dozen requests that cannot succeed.
- The sign-in box has no close button when there is nothing behind it to return
  to.

**`accounts.py`, for managing accounts from the command line** — including the
case this release exists for, being unable to get back in:

    python accounts.py list
    python accounts.py passwd <name>     forgotten password
    python accounts.py remove <name>     keeps their workspace
    python accounts.py off               removes every account

Turning accounts off leaves every workspace on disk and reopens the shared one.
Whoever runs the server can already read everything it stores, so a reset here
costs nothing and being locked out of your own designs costs a lot.

## 1.20.0

**A scanned folder stays browsable**, and **accounts keep one person's work
separate from another's**.

### Project files

- The left panel has a **Files** tab beside Layers. After scanning a folder, its
  tree stays there: directories, files, and a chip for every `nn.Module` found
  in each one. Files that would not parse are marked rather than hidden.
- Click a filename to read its source in the code panel, with highlighting.
  Reading never runs anything — only importing a class does.
- Click a model chip to import that class straight onto a new sheet. Classes
  needing constructor arguments ask for them.
- Reading is confined to the scanned folder: a path that climbs out of it is
  refused.

### Accounts

- **Optional.** With none registered the app behaves exactly as before, on the
  workspace it always used. Registering the first account turns authentication
  on, and that first account inherits the existing designs, runs and studies
  rather than hiding them.
- Each further account gets its own designs, runs, studies and panel layout.
- Passwords are hashed with scrypt and a per-account salt. Changing a password
  signs out every other session.
- **What this is not.** Accounts separate users from each other, not from the
  machine. Importing code, importing a folder, and the blocks and recipes
  folders all execute Python by design — that is what they are for — so any
  account that can reach them can run code as this process. This belongs on a
  network you trust, and the sign-up screen says so rather than leaving it to be
  assumed.

Two bugs found while building this, both the same shape and both silent: the
account was first bound in middleware, then in a synchronous dependency, and in
each case every account resolved to the same workspace. Starlette runs
middleware's `call_next` in a separate task and FastAPI runs sync dependencies
in a worker thread, so a context variable set in either is invisible to the
endpoint. It is an async dependency now, and a test asserts that, because the
failure produced no error — just quietly shared data.

## 1.19.0

Multi-file projects arrive as multi-sheet workbooks.

- **Scan a folder.** Import → *Scan a folder of Python files* lists every
  `nn.Module` in a project — class, file, line, and how many constructor
  arguments it needs. Discovery reads syntax trees only; nothing runs until a
  class is actually picked, and a file whose import would explode cannot hurt
  the scan. Classes needing arguments take them as they would in code
  (`width=64`). Modules import with their package on the path, so the relative
  imports between a project's files resolve the way they do when the project
  runs.
- **Sheets, tabbed like a spreadsheet**, in a strip at the bottom of the canvas.
  Click to switch, double-click to rename, right-click to make a sheet the
  model, `+` for a new one. Each picked class from a folder scan becomes its own
  sheet.
- **One sheet can stand on another.** A `Subgraph` node references a sheet by
  name; the node carries *↪ continues on "stem"* and double-clicking it opens
  that sheet. Shapes flow through the reference, so the parent sheet knows what
  comes back out.
- **The reference is real, not decorative.** A referenced sheet is generated as
  its own `nn.Module` class and the parent instantiates it — a class per sheet,
  the way the code would be written by hand. Verified: the generated two-sheet
  model runs, and its parameter count matches the canvas exactly. Renaming a
  sheet renames every reference to it; deleting a sheet something references is
  refused and says which sheets those are.
- Sheets nothing references are kept in the workbook but stay out of the
  generated code and the parameter total — a class nothing constructs is dead
  code.
- Sheets referencing each other in a circle are refused, with the circle named:
  the generated code would not terminate.
- Old flat saves still open; a one-sheet workbook saves flat, so nothing about
  existing designs changes shape.
- Five tests, including one that plants a booby-trapped file to prove scanning
  executes nothing.

## 1.18.0

- **Paste PyTorch source and get a diagram.** The Import dialog takes code now,
  alongside a torchvision name, an `.onnx` file or a `.pt`. Paste a class that
  subclasses `nn.Module`, or a variable holding one — `model = nn.Sequential(...)`
  works. `torch`, `nn` and `F` are already in scope.
- It runs the code to trace it, which is the only way: no static reader can tell
  you what `forward` does. Same trust assumption as the blocks and recipes
  folders, and the dialog says so.
- Refusals explain themselves: code that does not run reports the syntax error,
  code with no `nn.Module` says what to paste instead, and a class needing
  constructor arguments says to assign one to a variable.
- Fixed while testing this: `F.max_pool2d` and `F.avg_pool2d` were coming in as
  opaque stubs, because the tracer only recognised pooling as a module. The
  functional forms are at least as common in real code. A two-convolution network
  written with them now imports with no stubs at all.

## 1.17.0

A **Studies** page: agents that run experiments for you.

Worth being precise about the word. A workflow tool's agentic tasks orchestrate
calls to external AI services — chat completions, embeddings, MCP tools. Those
are not layers: they are not differentiable, carry no tensor shape, and nothing
downstream could consume them. The automation this app was actually short of is
different, and it is this: somebody still had to sit there changing one number
and pressing Train.

- **Hyperparameter sweep** — the same network trained across learning rates,
  batch sizes and optimizers.
- **Architecture search** — wider, narrower and more regularized versions of the
  network, each trained and ranked. The final head is left alone, because the
  number of classes is not a hyperparameter.
- **Try the review's fixes** — takes what `review` found, applies each fix on its
  own, and trains them beside the network as drawn, so you learn which of them
  actually helped rather than assuming. On a test network it ranked adding
  dropout above adding batch norm, and both above leaving it alone.
- Variants that would not build are discarded during planning, before anything
  is trained.
- **Every trial is an ordinary run**: it appears in the run history, writes
  checkpoints, and can be reopened. The study adds a leaderboard over the top,
  and *open* puts any variant on the canvas.
- Studies persist to `studies/`, so a leaderboard survives a restart.
- Three tests, including one asserting every proposed variant builds and one
  that width search never resizes the head.

## 1.16.0

- **A quick-add picker anchored to the `+` you pressed**, rather than a list in
  the middle of the window. Twelve tiles for the layers people reach for most,
  a search box, and *All layers* to open the full set grouped by category. It
  flips to the other side of the button when it would run off the edge.
- **Each tile carries a drawn glyph** — a grid for convolution, stacked bars for
  dense, a step for an activation, a parallelogram for shape surgery. Drawn as
  inline paths rather than an icon font, and there is a test that every category
  in the registry has one, so a new category cannot ship with blank tiles.
- The picker only offers layers that can go where you are putting them: no
  Outputs or runtime blocks inserted mid-chain, though both are offered when
  appending at the end.
- Arrow keys move through the tiles, Enter places one, Escape closes.
- **Layered shadows** on every node shape — a tight contact shadow plus a soft
  ambient one, and a stronger pair on hover. The single flat shadow read as
  pasted-on.

## 1.15.1

- **Fixed: the status strip was covering the bottom of the side panels.** It is
  30px tall and was pinned to the bottom of the whole page rather than to the
  canvas, so it lay across the last 30px of anything docked beside it. That
  buried the assistant's input box, and the end of the palette with it. It now
  lives inside the canvas and spans only the canvas. The minimap moved up to
  clear it.
- Two tests: one asserts nothing pinned to the page edge sits outside the
  canvas, the other that the assistant's log can shrink and its input bar
  cannot. Verified against the broken layout — the first fails on it and names
  the culprit.

## 1.15.0

An **Assistant** tab, and an honest account of what it is.

Theirs is backed by a language model. This one is not — the MiniGPT in this app
is an 805,000-parameter character model and could only produce fluent nonsense
about a network. So the assistant is rule-based: it matches a small set of
phrasings exactly, and says so when it does not recognise something rather than
guessing.

- **It edits the graph.** `add dropout after the activation`, `remove the
  flatten`, `set units to 64 on linear_1`, `freeze conv2d_2`, `rename linear_1
  to head`, `note on conv2d: widened for the tiles`. Edits come back as a whole
  graph, so the change appears on the canvas and Cmd+Z undoes it like any other.
- Layers resolve by label, by type, or by the name in the generated file — which
  is what you actually read off a node.
- **`review` finds real problems**, by inspection rather than opinion. Two Linear
  layers with no activation between them compose to a single linear layer. A
  Flatten of a 32×32×32 map hands the next Linear 8,388,608 weights, and the
  message says what GlobalAvgPool would cost instead. A convolution going
  straight to its activation with no normalization. A spatial map already at 1×1
  with convolutions still below it. Everything frozen, so training would change
  nothing. A single value arriving at a classification Output.
- It refuses settings a layer does not have, and lists the ones it does.
- **Optional model.** Set `ASSISTANT_API_KEY` and anything it does not recognise
  is forwarded to an Anthropic-compatible endpoint with a summary of the graph.
  Without it, no pretending.
- Two more tests, 60 in total.

## 1.14.0

The Code tab was a dark block of unstyled text. It is a viewer now.

- **Light background**, matching the canvas beside it. Reading code against one
  background and a diagram against another is tiring.
- **Line numbers** in a gutter.
- **Python syntax highlighting** — keywords, strings, comments, numbers,
  decorators, `self`, and the names in `def` and `class` lines. The file is
  tokenised in one pass before being cut into lines, so a docstring stays a
  single string token across all eleven of its lines rather than falling apart
  at the first newline.
- **A map of the whole file** down the right edge: one bar per line, its width
  the line's length, its indent the code's indent, its colour what the line
  mostly is. The current viewport is marked, and clicking jumps there.
- Jumping from a layer's *Show in Code* highlights the line and scrolls to it.
- Everything is HTML-escaped, so a `<` in the generated code stays a `<`.

## 1.13.0

The Layer tab listed parameters and little else. Three things it now does, each
the working counterpart of something in a task editor rather than a decoration.

- **Test layer.** Builds the selected layer on its own, pushes a tensor of the
  right shape through it, and reports what came out: actual shape, parameter
  count, time and dtype — and whether that agrees with what the canvas
  predicted. The canvas computes shapes arithmetically; this checks the
  arithmetic against PyTorch, per layer, with your settings.
- **Reference.** Core layers link straight to their `torch.nn` documentation
  page. Blocks link to their source file instead, since they have no PyTorch
  page.
- **Freeze this layer.** The weights keep their values and the optimizer leaves
  them alone. The generated code emits the `requires_grad_(False)` loop, the
  header count drops to the trainable total, and the panel says how many
  parameters were parked. Verified against PyTorch: canvas and framework agree
  on the trainable count.
- The panel header carries the layer type, so what you are editing is stated
  rather than inferred.

## 1.12.0

- **Seven node shapes, mapped to flowchart convention** rather than chosen for
  variety. A circle is a terminator, so Input and Output are circles. A diamond
  is a decision, so merges are diamonds. A parallelogram is the data symbol, so
  Flatten, Reshape and Permute — which reinterpret the same values — are
  parallelograms. A rectangle with side bars is a predefined process, so every
  plug-in block gets one. A stadium is a simple step, so activations are pills.
  A hexagon is preparation, which suits a runtime component. Everything else is
  an ordinary process rectangle.
- **Drag the whole graph from anywhere.** The hand tool in the toolbar turns it
  on, holding space turns it on while held, and the middle mouse button always
  pans. Because layers here are individually draggable — unlike a workflow
  editor, where the layout is automatic — this had to be a mode rather than the
  default, or you could never move a single layer again.
- The pan handler captures, so a node never starts moving when you meant to move
  the canvas.

## 1.11.0

You could not see what "Add this layer" was doing, because the Projects page
covered the canvas. That made the guided build useless — the whole point is
watching shapes resolve as each layer lands.

- **The guided build moved to a Guide tab beside the canvas.** Press *Build step
  by step* on a project and you are taken to the canvas with the plan in the
  right panel. Each press places the next layer, selects it, and the canvas is
  in front of you the whole time.
- **Bulk import.** *Add all layers* brings the whole plan in at once. Every step
  has a checkbox, so *Add selected* brings in just the ones you ticked — take
  the backbone and head from a transfer-learning project and skip its dropout,
  or lift the aggregator out of the pathology project.
- **Keep what is already on the canvas**, so a project's layers can be appended
  to a network you are already building rather than replacing it.
- Selecting a project no longer clears your canvas. Nothing is touched until you
  choose how to bring it in.
- *Add the rest* in the Guide finishes a build you started stepping through.
- Stepping and bulk import share one placement routine, so the same plan gives
  the same graph either way. A selection that skips a step drops the connections
  that would have dangled rather than leaving broken edges.

## 1.10.1

**Restores the Projects, Runs, Chat and Import pages, which were dead since
1.8.0.**

Removing the bottom drawer, I replaced a span of the script bounded by two
comment markers. Four whole sections sat between them and went with it. The
markup survived, so every page still rendered its shell and every other test
passed — but clicking Projects, Runs or Chat called a function that no longer
existed, and Import did nothing. That state shipped in 1.8.0, 1.8.1, 1.9.0 and
1.10.0.

- All four sections restored and brought up to the current layout: the guided
  stepper places layers along whichever direction the canvas is flowing, and the
  Train shortcut uses the side panel rather than the drawer that no longer
  exists.
- **Two new tests that would have caught it.** One asserts every page and panel
  entry point is defined; the other reads `PAGE_SETUP` and checks each page's
  loader points at a real function. Verified against the broken build — both
  fail on it, naming the missing functions.

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
