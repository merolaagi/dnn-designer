"""The project catalogue.

Entries are generated from the shared builders so each one's rationale quotes
its own numbers. A 32x32 classifier and a 128x128 one are genuinely different
builds — different flatten widths, different pooling depths, different advice —
not the same entry listed twice.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _builders import (char_language_model, cnn_classifier, mlp_tabular,  # noqa: E402
                       resnet_classifier, sequence_model, transfer_classifier)
from _builders2 import (autoencoder, detector, diffusion_unet,  # noqa: E402
                        gan_generator, graph_classifier, mil_aggregator,
                        policy_network, simclr_encoder)
from projects_sdk import Project, install  # noqa: E402


def P(pid, name, category, summary, steps, tags, **kw):
    install(Project(id=pid, name=name, category=category, summary=summary,
                    steps=steps, tags=tags, **kw))


# ---------------------------------------------------------------- vision
IMAGE_TASKS = [
    ("mnist", "Handwritten digits", [1, 28, 28], 10, "mnist",
     "The standard first network. Small, fast, and it works.", "starter"),
    ("fashion", "Clothing categories", [1, 28, 28], 10, "fashion_mnist",
     "Same shape as digits but genuinely harder — textures rather than strokes.", "starter"),
    ("cifar", "Ten everyday objects", [3, 32, 32], 10, "cifar10",
     "Colour, clutter and real backgrounds. Where a digit classifier stops being enough.", "starter"),
    ("leaf", "Plant disease from leaf photos", [3, 64, 64], 6, "folder",
     "Field photographs sorted into folders by disease.", "intermediate"),
    ("waste", "Recycling sorting", [3, 64, 64], 5, "folder",
     "Sort waste photographs into material types.", "intermediate"),
    ("parts", "Manufactured part inspection", [3, 96, 96], 4, "folder",
     "Tell good parts from the common defect modes.", "intermediate"),
    ("food", "Dish recognition", [3, 96, 96], 12, "folder",
     "Photographs of prepared food, one folder per dish.", "intermediate"),
    ("satellite", "Land use from satellite tiles", [3, 64, 64], 8, "folder",
     "Overhead tiles labelled by what covers the ground.", "intermediate"),
]
for pid, name, shape, classes, data, blurb, level in IMAGE_TASKS:
    P(f"cnn-{pid}", f"{name} — small CNN", "Vision",
      f"{blurb} Two convolution blocks and a dense head.",
      cnn_classifier(shape, classes, blocks=2), 
      ["image", "classification", "cnn", pid],
      difficulty=level, data=f"Dataset: {data}.",
      training="AdamW at 1e-3, batch 64, 10 epochs to start.",
      expect="Training loss should fall within the first epoch. If it does not, the learning rate is wrong.")

for pid, name, shape, classes, data, blurb, level in IMAGE_TASKS[2:]:
    P(f"resnet-{pid}", f"{name} — residual network", "Vision",
      f"{blurb} A residual trunk, which trains deeper without the gradient dying.",
      resnet_classifier(shape, classes, blocks=3),
      ["image", "classification", "resnet", "residual", pid],
      difficulty="intermediate", data=f"Dataset: {data}.",
      training="AdamW at 1e-3. Residual nets tolerate more depth and more epochs than a plain stack.",
      expect="Should beat the plain CNN on the same data, and keep improving for longer.")

for arch in ("resnet18", "resnet50", "efficientnet_b0", "mobilenet_v3_large", "densenet121"):
    P(f"transfer-{arch}", f"Transfer learning with {arch}", "Vision",
      f"Freeze {arch}'s ImageNet features and train only a small head. The right "
      f"move whenever you have hundreds of images rather than tens of thousands.",
      transfer_classifier([3, 224, 224], 5, arch),
      ["image", "classification", "transfer", "pretrained", arch],
      difficulty="starter", data="A folder of images, one subfolder per class.",
      training="1e-3 or lower. Far fewer epochs than training from scratch.",
      expect="Useful accuracy from a few hundred images per class.",
      caution="Weights download on the first run.")

# ---------------------------------------------------------------- medical
MEDICAL = [
    ("chest-xray", "Chest X-ray findings", [1, 224, 224], 4,
     "Frontal chest radiographs labelled by finding."),
    ("skin-lesion", "Skin lesion categories", [3, 128, 128], 7,
     "Dermatoscopic images across lesion types."),
    ("retina", "Diabetic retinopathy grading", [3, 224, 224], 5,
     "Fundus photographs graded by severity."),
    ("cell", "Blood cell types", [3, 64, 64], 4,
     "Microscopy crops of individual cells."),
]
for pid, name, shape, classes, blurb in MEDICAL:
    P(f"med-{pid}", name, "Medical imaging",
      f"{blurb} Transfer learning, because medical datasets are almost always small.",
      transfer_classifier(shape, classes, "resnet18"),
      ["medical", "image", "classification", pid.split("-")[0]],
      difficulty="intermediate",
      data="One folder per class. Record the scanner and site for every image.",
      training="1e-3, and hold out an entire site rather than a random split.",
      caution=("Split by site and equipment, not at random. A model that learns "
               "the scanner rather than the disease posts excellent internal "
               "numbers and fails everywhere else. This is the single most common "
               "way medical imaging projects go wrong."))

P("med-mil-slide", "Whole-slide differential diagnosis", "Medical imaging",
  "Attention over thousands of tile features from one slide, with a multi-label "
  "head. The standard approach when you have one label per slide and no labels "
  "per tile.",
  mil_aggregator(1024, 8), ["medical", "pathology", "mil", "attention", "slide", "diagnosis"],
  difficulty="advanced",
  data=("Precomputed tile features, not slides. Tile at 20x into 224px patches, "
        "run them through a frozen pathology encoder once, cache the vectors."),
  training="Low learning rate, and evaluate per class rather than on an average.",
  expect="Attention weights should land on tissue a pathologist would also look at. If they land on pen marks, fix the data.",
  caution=("The architecture is the easy part. Label quality, site "
           "generalization and clinical validation are where this lives or "
           "dies, and a diagnostic tool is a regulated device in most places."))

P("med-anomaly-ae", "Find abnormal scans without labels", "Medical imaging",
  "Train an autoencoder on normal studies only. Anything it reconstructs badly "
  "is unlike what it saw, which is a usable abnormality signal when you have "
  "plenty of normals and few labelled abnormals.",
  autoencoder([1, 64, 64], 32), ["medical", "anomaly", "autoencoder", "unsupervised"],
  difficulty="intermediate", recipe="Autoencoder",
  data="Normal studies only. Never show it an abnormal during training.",
  expect="Reconstruction error should separate held-out normals from abnormals.",
  caution="This finds unusual, not abnormal. Rare-but-healthy variation scores just as high.")

# ---------------------------------------------------------------- tabular
TABULAR = [
    ("churn", "Customer churn", 20, 2, "classification", "Predict who leaves next month."),
    ("credit", "Credit default risk", 24, 2, "classification", "Score loan applications."),
    ("fraud", "Card fraud", 30, 2, "classification", "Flag fraudulent transactions."),
    ("house", "House price", 16, 1, "regression", "Predict sale price from property features."),
    ("demand", "Fuel demand", 12, 1, "regression", "Forecast volume from station and calendar features."),
    ("yield", "Crop yield", 18, 1, "regression", "Predict yield from soil and weather."),
    ("quality", "Product quality score", 14, 1, "regression", "Predict a measured quality outcome."),
    ("triage", "Support ticket priority", 22, 4, "classification", "Route tickets by urgency."),
]
for pid, name, feats, out, task, blurb in TABULAR:
    P(f"tab-{pid}", name, "Tabular",
      f"{blurb} A small MLP over {feats} columns.",
      mlp_tabular(feats, out, task), ["tabular", "csv", task, pid],
      difficulty="starter",
      data=f"A CSV with roughly {feats} feature columns and the target in its own column.",
      training="AdamW at 1e-3, batch 64. Standardize features, which the CSV loader does for you.",
      expect="Compare against a gradient-boosted tree before assuming a neural network is the right tool here.",
      caution="Class imbalance is the usual problem on fraud and churn, not architecture.")

# ---------------------------------------------------------------- sequence
SEQUENCE = [
    ("ecg", "Heartbeat classification", 256, 1, 5, "lstm", "Single-lead ECG windows."),
    ("har", "Activity from phone sensors", 128, 6, 6, "lstm", "Accelerometer and gyroscope windows."),
    ("machine", "Machine failure prediction", 200, 8, 2, "lstm", "Vibration and temperature traces."),
    ("energy", "Energy load forecast", 168, 4, 1, "conv", "A week of hourly readings."),
    ("stock", "Price movement direction", 60, 5, 3, "conv", "Recent bars with derived features."),
    ("weather", "Rain tomorrow", 72, 6, 2, "conv", "Three days of hourly observations."),
    ("gesture", "Gesture recognition", 100, 9, 8, "transformer", "Motion capture windows."),
    ("eeg", "EEG state classification", 250, 16, 4, "transformer", "Multi-channel brain recordings."),
]
for pid, name, length, ch, out, kind, blurb in SEQUENCE:
    P(f"seq-{pid}", f"{name} — {kind}", "Sequences",
      f"{blurb} Built with a {kind} over {length} steps of {ch} channel"
      f"{'s' if ch > 1 else ''}.",
      sequence_model(length, ch, out, kind), ["sequence", "timeseries", kind, pid],
      difficulty="intermediate",
      data=f"A CSV where each row holds {length * ch} values, reshaped to [{length}, {ch}].",
      training="AdamW at 1e-3. Sequences overfit fast; watch the validation curve from epoch one.",
      caution=("If you are predicting the future, do not use a bidirectional layer "
               "and do not shuffle across the time boundary — both leak the answer."))

# ---------------------------------------------------------------- language
for pid, name, ctx, depth, heads, blurb in [
    ("mini", "MiniGPT — character language model", 128, 4, 4,
     "The worked example. Learns to continue text one character at a time."),
    ("tiny", "Tiny language model", 64, 2, 2,
     "Half the size, trains in about a minute. Good for checking a corpus."),
    ("deep", "Deeper character model", 192, 8, 8,
     "More capacity for a larger corpus. Slower, and needs more text to justify itself."),
]:
    P(f"lm-{pid}", name, "Language",
      f"{blurb} Embedding, positional encoding, causal transformer blocks, and a "
      f"head back to the vocabulary.",
      char_language_model(ctx, 48, depth=depth, heads=heads),
      ["text", "language", "gpt", "transformer", "generative", pid],
      difficulty="advanced" if depth > 4 else "intermediate",
      data="A .txt corpus. The loader builds the character vocabulary for you.",
      training="AdamW at 3e-3, batch 32.",
      expect="Perplexity near 1 on repetitive text; the Chat tab is where you judge it.",
      caution="Set the Embedding vocab and the final Linear to the corpus's character count, or it will not train.")

for pid, name, length, blurb in [
    ("sentiment", "Sentiment from characters", 128, "Positive or negative from raw characters, no tokenizer."),
    ("spam", "Spam detection", 96, "Flag unwanted messages."),
    ("intent", "Intent classification", 64, "Route a short message to one of several intents."),
]:
    P(f"text-{pid}", name, "Language",
      f"{blurb} An Embedding then a transformer encoder, pooled to one decision.",
      sequence_model(length, 64, 2 if pid != "intent" else 6, "transformer"),
      ["text", "classification", "transformer", pid],
      difficulty="intermediate",
      data="Character ids in a CSV, or adapt the text corpus loader.",
      caution="Character-level models need far more data than word-level ones to reach the same accuracy.")

# ---------------------------------------------------------------- generative
for pid, name, shape, blurb in [
    ("faces", "Generate faces", [3, 32, 32], "Small face crops."),
    ("textures", "Generate textures", [3, 32, 32], "Repeating material patterns."),
    ("digits", "Generate digits", [1, 32, 32], "The easiest generative target — start here."),
]:
    P(f"gan-{pid}", f"{name} — GAN", "Generative",
      f"{blurb} A generator turning noise into images, trained against a "
      f"discriminator that learns to tell real from fake.",
      gan_generator(shape), ["generative", "gan", "adversarial", pid],
      difficulty="advanced", recipe="GAN",
      data="A folder of images. No labels are used.",
      training="Both learning rates near 2e-4. Raise generator steps if the discriminator dominates.",
      expect="d_accuracy hovering near 0.5. Climbing to 1.0 means the generator has stopped learning.",
      caution="GANs are the least stable thing in this catalogue. Expect to tune.")
    P(f"diffusion-{pid}", f"{name} — diffusion", "Generative",
      f"{blurb} A denoiser trained to predict the noise added to an image, then "
      f"run backwards from pure noise.",
      diffusion_unet(shape), ["generative", "diffusion", "denoising", pid],
      difficulty="advanced", recipe="Diffusion",
      data="A folder of images. No labels.",
      training="2e-4, 200 noise steps. Slower than a GAN but far more stable.",
      expect="The per-epoch preview reports the sampled range; it should sit inside the data range.")

for pid, name, shape, code, blurb in [
    ("mnist-ae", "Compress digits", [1, 28, 28], 16, "Squeeze a digit into 16 numbers and back."),
    ("image-ae", "Compress images", [3, 32, 32], 64, "A colour image through a 64-number bottleneck."),
    ("tab-ae", "Compress tabular rows", [32], 4, "Find a four-number summary of a wide row."),
]:
    P(f"ae-{pid}", name, "Generative",
      f"{blurb} The target is the input, so no labels are needed.",
      autoencoder(shape, code), ["autoencoder", "compression", "unsupervised", pid],
      difficulty="starter", recipe="Autoencoder",
      data="Any dataset. Labels are ignored.",
      expect="Reconstruction loss should fall steadily. Raise the noise setting to make it a denoising autoencoder.")

# ---------------------------------------------------------------- self-supervised
for pid, name, shape, blurb in [
    ("images", "Pretrain on unlabelled images", [3, 32, 32], "General purpose."),
    ("medical", "Pretrain on unlabelled scans", [1, 64, 64], "Where labels are expensive and images are not."),
]:
    P(f"ssl-{pid}", name, "Self-supervised",
      f"{blurb} Two augmented views of the same image should land close together "
      f"in the embedding, everything else far apart. No labels at all.",
      simclr_encoder(shape), ["selfsupervised", "contrastive", "simclr", "pretraining", pid],
      difficulty="advanced", recipe="Contrastive",
      data="Unlabelled images in a folder.",
      training="Large batches matter — the negatives all come from within the batch.",
      expect="Judge it by starting a classifier from the checkpoint, not by the loss.")

# ---------------------------------------------------------------- agents
P("rl-cartpole", "Balance a pole", "Agents",
  "Policy gradients on CartPole. The network acts, the environment answers, and "
  "the episode is the training batch — no dataset at all.",
  policy_network(4, 2), ["reinforcement", "policy", "cartpole", "agent", "game"],
  difficulty="intermediate", recipe="Reinforce",
  data="None. The environment is built in.",
  training="1e-2, 40 episodes per epoch.",
  expect="Return climbing from about 10 to the step cap within a few epochs.")

P("rl-bigger-policy", "A wider policy network", "Agents",
  "The same task with more capacity, to see how little it helps. Policy "
  "gradients are noisy and a bigger network fits the noise faster.",
  policy_network(4, 2, width=256), ["reinforcement", "policy", "agent", "capacity"],
  difficulty="intermediate", recipe="Reinforce",
  expect="Often slower and less stable than the small one. That is the lesson.")

for pid, name, size, actions, blurb in [
    ("tictactoe", "Tic-tac-toe", 3, 9, "Small enough to verify the search by hand."),
    ("connect4", "Connect Four", 7, 7, "Where search starts to matter more than the network."),
    ("gomoku", "Gomoku on 9x9", 9, 82, "Go-like, with a pass move."),
]:
    from _builders import resnet_classifier as _rc
    steps = _rc([17, size, size], actions, width=64, blocks=1)[:-2]
    from projects_sdk import Step, node as _n
    steps.append(Step(
        title="Shared trunk output",
        why=("Both heads read the same features. One trunk, two heads: the policy "
             "and the value are different questions about the same position, and "
             "sharing everything below them is most of why this trains at all."),
        nodes=[_n("Activation", {"kind": "relu"}, id="trunk")]))
    steps += [
        Step(title=f"Policy head — {actions} moves",
             why=(f"One logit per legal move. Trained against the visit counts the "
                  f"search produces, not against a human's choice — the search is "
                  f"the teacher."),
             nodes=[_n("PolicyHead", {"actions": actions}),
                    _n("Output", {"task": "classification"}, id="pout")],
             connect_from="trunk"),
        Step(title="Value head",
             why=("A single number in [-1, 1] estimating who is winning from the "
                  "mover's point of view. This is what lets the search stop before "
                  "the end of the game."),
             nodes=[_n("ValueHead", {"hidden": 128}, id="value"),
                    _n("Output", {"task": "regression"}, id="vout")],
             connect_from="trunk"),
        Step(title="Monte Carlo tree search",
             why=("A runtime block. Search is not a tensor transform — it calls the "
                  "network hundreds of times per move and picks by visit count. It "
                  "generates a separate section of the exported file."),
             nodes=[_n("MCTSSearch", {"simulations": 200}, id="mcts")],
             connect_from="__none__",
             connect=[("pout", "mcts", 0), ("vout", "mcts", 1)],
             watch=("The generated file defines an Environment with five methods "
                    "and leaves them for you. Search cannot run without your game.")),
    ]
    P(f"game-{pid}", f"{name} — AlphaZero shape", "Agents",
      f"{blurb} A residual trunk with policy and value heads, plus tree search "
      f"attached past the outputs.",
      steps, ["game", "alphazero", "mcts", "reinforcement", pid],
      difficulty="advanced",
      data="Self-play. You supply the game rules.",
      caution="The network trains supervised on search output; the self-play loop is not in the trainer yet.")

# ---------------------------------------------------------------- detection & graphs
for pid, name, shape, classes, blurb in [
    ("shapes", "Find shapes", [3, 64, 64], 2, "Squares and circles, drawn synthetically."),
    ("objects", "Find objects", [3, 128, 128], 4, "A larger grid for a busier scene."),
]:
    P(f"det-{pid}", name, "Detection",
      f"{blurb} A grid head predicting objectness, a box and a class at every cell.",
      detector(shape, classes), ["detection", "boxes", "objects", pid],
      difficulty="advanced", recipe="Detection",
      data="Synthetic, drawn by the recipe. Real annotations need a loader that does not exist yet.",
      expect="Recall above 0.9 on the synthetic shapes within a few epochs.")

for pid, name, n, feats, classes, blurb in [
    ("molecule", "Molecule property", 20, 16, 2, "Atoms as nodes, bonds as edges."),
    ("social", "Community classification", 50, 32, 4, "People as nodes, connections as edges."),
]:
    P(f"graph-{pid}", name, "Graphs",
      f"{blurb} Two rounds of neighbour averaging, then a graph-level head.",
      graph_classifier(n, feats, classes), ["graph", "gnn", "network", pid],
      difficulty="advanced",
      data="Two Inputs: node features and an adjacency matrix. No graph loader ships yet.",
      caution="More than three graph convolutions usually makes every node look the same.")

# ---------------------------------------------------------------- numerical
from projects_sdk import Step as _S, node as _N  # noqa: E402

P("num-ode", "Continuous-depth classifier", "Numerical",
  "Replaces a stack of layers with one field integrated over time. Depth becomes "
  "a number you set rather than parameters you add.",
  [_S(title="Input — 3x32x32", why="Ordinary images.", nodes=[_N("Input", {"shape": [3, 32, 32]})]),
   _S(title="Lift to 32 channels",
      why="The ODE block preserves its input shape, so the channel count is fixed before it.",
      nodes=[_N("Conv2d", {"filters": 32, "kernel": 3, "padding": "same"}),
             _N("Activation", {"kind": "silu"})]),
   _S(title="ODE block — RK4, 4 steps",
      why=("Integrates dh/dt = f(h) with the same f at every step. More steps means "
           "more computation and identical parameter count, which is the whole "
           "trade this offers."),
      nodes=[_N("ODEBlock", {"field": "conv", "hidden": 64, "steps": 4})],
      alternatives="Raise steps for accuracy, lower for speed. Nothing about the network changes."),
   _S(title="Pool and classify", why="Standard head.",
      nodes=[_N("GlobalAvgPool", {}), _N("Linear", {"units": 10}, label="head"),
             _N("Output", {"task": "classification"})])],
  ["numerical", "ode", "continuous", "image"], difficulty="advanced",
  data="Any image dataset.", expect="Comparable accuracy to a residual net with far fewer parameters.")

P("num-deq", "Solve for a fixed point", "Numerical",
  "Runs one transform until it stops changing, rather than stacking copies of it. "
  "Stops early when the state settles.",
  [_S(title="Input — 32 features", why="Tabular or pooled features.",
      nodes=[_N("Input", {"shape": [32]})]),
   _S(title="Fixed point — up to 16 iterations",
      why=("Iterates z toward a solution of z = f(z, x). Effective depth varies "
           "with the input: easy cases converge in three iterations, hard ones use "
           "all sixteen."),
      nodes=[_N("FixedPoint", {"hidden": 128, "iterations": 16})]),
   _S(title="Head", why="Reads the settled state.",
      nodes=[_N("Linear", {"units": 4}, label="head"), _N("Output", {"task": "classification"})])],
  ["numerical", "equilibrium", "deq", "fixedpoint"], difficulty="advanced")

P("num-solve", "Learn to pose a linear system", "Numerical",
  "The network builds a matrix and a right-hand side, and a differentiable solver "
  "returns the solution. Gradients flow through the solve, so upstream layers "
  "learn to pose a well-conditioned problem.",
  [_S(title="Matrix input — [8, 8]", why="A square system.",
      nodes=[_N("Input", {"shape": [8, 8]}, id="A", label="A")]),
   _S(title="Right-hand side — [8]", why="A second Input holding b.",
      nodes=[_N("Input", {"shape": [8]}, id="b", label="b")], connect_from="__none__"),
   _S(title="Ridge solve",
      why=("Solves (A + lambda I)x = b with lambda learned. The ridge term is what "
           "keeps a near-singular A from blowing the gradients up."),
      nodes=[_N("RidgeSolve", {}, id="solve")],
      connect_from="__none__",
      connect=[("A", "solve", 0), ("b", "solve", 1)]),
   _S(title="Head", why="Reads the solution.",
      nodes=[_N("Linear", {"units": 2}, label="head"), _N("Output", {"task": "classification"})])],
  ["numerical", "linear", "solver", "differentiable"], difficulty="advanced")

# ---------------------------------------------------------------- fusion
P("fusion-two-tower", "Combine sensors and context", "Multi-input",
  "Two Inputs, each with its own tower, concatenated before the head. The right "
  "shape when your features come from genuinely different sources.",
  [_S(title="Sensor input — 6 readings", why="One tower's worth of columns.",
      nodes=[_N("Input", {"shape": [6]}, label="sensors")]),
   _S(title="Sensor tower", why="Learns from the sensor block alone.",
      nodes=[_N("Linear", {"units": 32}), _N("Activation", {"kind": "relu"}, id="s2")]),
   _S(title="Context input — 4 columns", why="A second Input for the other source.",
      nodes=[_N("Input", {"shape": [4]}, id="ctx", label="context")], connect_from="__none__"),
   _S(title="Context tower", why="Its own path, so the two are not forced into a shared representation too early.",
      nodes=[_N("Linear", {"units": 16}, id="c1"), _N("Activation", {"kind": "relu"}, id="c2")],
      connect=[("ctx", "c1", 0)]),
   _S(title="Concatenate",
      why=("Joins the two towers into one vector. Both have learned their own "
           "representation first, which is the point of keeping them separate."),
      nodes=[_N("Concat", {"axis": 0}, id="join")],
      connect_from="__none__",
      connect=[("s2", "join", 0), ("c2", "join", 1)]),
   _S(title="Head", why="Decides from the combined representation.", connect_from="join",
      nodes=[_N("Linear", {"units": 32}), _N("Activation", {"kind": "relu"}),
             _N("Linear", {"units": 3}, label="head"), _N("Output", {"task": "classification"})],
      watch=("forward() takes its Inputs in topological order and the CSV loader "
             "feeds them in that order. Map the columns explicitly if you want to "
             "be sure which tower gets which."))],
  ["multiinput", "fusion", "tabular", "towers"], difficulty="intermediate",
  data="One CSV; assign column groups to each Input in the training form.")
