"""Catalogue, second half: audio, anomaly detection, similarity, and more of the
domains where the framing genuinely differs rather than just the numbers."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _builders import (cnn_classifier, mlp_tabular, sequence_model,  # noqa: E402
                       transfer_classifier)
from _builders2 import autoencoder, simclr_encoder  # noqa: E402
from projects_sdk import Project, Step, install, node  # noqa: E402


def P(pid, name, category, summary, steps, tags, **kw):
    install(Project(id=pid, name=name, category=category, summary=summary,
                    steps=steps, tags=tags, **kw))


# ---------------------------------------------------------------- audio
AUDIO_NOTE = ("Audio becomes an image before it reaches the network. Compute a "
              "mel spectrogram — time along one axis, frequency along the other — "
              "and save it as a single-channel image. Every convolution below "
              "then works exactly as it would on a photograph.")

for pid, name, shape, classes, blurb in [
    ("keyword", "Spoken keyword spotting", [1, 64, 64], 10, "Short commands like yes, no, stop."),
    ("urban", "Urban sound classification", [1, 128, 128], 10, "Sirens, drilling, dogs, traffic."),
    ("music", "Music genre", [1, 128, 128], 8, "Thirty-second clips by genre."),
    ("cough", "Respiratory sound screening", [1, 96, 96], 3, "Cough and breath recordings."),
    ("engine", "Engine fault from sound", [1, 64, 64], 4, "Machinery recordings by fault mode."),
]:
    P(f"audio-{pid}", name, "Audio",
      f"{blurb} A convolution stack over spectrograms.",
      cnn_classifier(shape, classes, blocks=2 if shape[1] <= 64 else 3),
      ["audio", "sound", "spectrogram", "classification", pid],
      difficulty="intermediate",
      data=f"{AUDIO_NOTE} One folder per class.",
      training="AdamW at 1e-3. Time-axis shifts are the most useful augmentation; horizontal flips are meaningless here.",
      caution="Do not flip a spectrogram horizontally — it reverses time. Vertical flips scramble pitch. Most image augmentations are wrong for audio.")

P("audio-raw-1d", "Classify raw waveforms", "Audio",
  "Skips the spectrogram and convolves along the samples directly. More data "
  "hungry, but nothing is discarded by the transform.",
  sequence_model(1024, 1, 6, "conv"), ["audio", "waveform", "conv1d", "raw"],
  difficulty="advanced",
  data="Fixed-length waveform windows as [1024, 1].",
  caution="Needs considerably more data than the spectrogram route to reach the same accuracy.")

# ---------------------------------------------------------------- anomaly
for pid, name, feats, blurb in [
    ("network", "Network intrusion", 40, "Connection records, mostly benign."),
    ("sensor", "Sensor drift", 24, "Readings from equipment running normally."),
    ("transaction", "Unusual transactions", 32, "Payment records without fraud labels."),
]:
    P(f"anom-{pid}", name, "Anomaly detection",
      f"{blurb} Train an autoencoder on normal rows only; whatever it "
      f"reconstructs badly is unlike what it has seen.",
      autoencoder([feats], max(4, feats // 8)),
      ["anomaly", "outlier", "autoencoder", "unsupervised", "tabular", pid],
      difficulty="intermediate", recipe="Autoencoder",
      data="A CSV of normal records only. Never include known anomalies in training.",
      training="Set noise above zero — a denoising autoencoder gives a sharper separation.",
      expect="Reconstruction error should separate held-out normals from anomalies. Pick the threshold on validation data, not on the test set.",
      caution="This detects unusual, not bad. Rare-but-legitimate records score exactly as high as genuine anomalies.")

P("anom-image-defect", "Visual defect detection", "Anomaly detection",
  "Train on defect-free product photographs only. The reconstruction error map "
  "shows where the model was surprised, which is usually where the defect is.",
  autoencoder([1, 64, 64], 64), ["anomaly", "defect", "image", "manufacturing", "autoencoder"],
  difficulty="intermediate", recipe="Autoencoder",
  data="Photographs of good parts only, consistently lit and framed.",
  expect="High reconstruction error localises the defect without ever being shown one.",
  caution="Inconsistent lighting or framing produces high error everywhere and drowns the signal.")

# ---------------------------------------------------------------- similarity
P("sim-siamese", "Tell whether two images match", "Similarity",
  "Two Inputs sharing the same downstream shape, each embedded, then compared. "
  "The right shape when the classes are not fixed in advance — signatures, faces, "
  "product matching.",
  [Step(title="First image", why="One half of the pair.",
        nodes=[node("Input", {"shape": [1, 64, 64]}, label="left")]),
   Step(title="Encoder for the first",
        why="An ordinary convolution trunk producing an embedding.",
        nodes=[node("Conv2d", {"filters": 32, "kernel": 3, "padding": "same"}),
               node("Activation", {"kind": "relu"}),
               node("MaxPool2d", {"kernel": 2}),
               node("GlobalAvgPool", {}),
               node("Linear", {"units": 64}, id="e1")]),
   Step(title="Second image",
        why="The other half. With a built-in image dataset both Inputs receive the same batch, which is exactly the siamese setup.",
        nodes=[node("Input", {"shape": [1, 64, 64]}, id="right", label="right")],
        connect_from="__none__"),
   Step(title="Encoder for the second",
        why=("A separate path here. True weight sharing needs one encoder applied "
             "twice, which the canvas cannot express yet — this approximates it "
             "and is the honest limitation of building a siamese network this way."),
        nodes=[node("Conv2d", {"filters": 32, "kernel": 3, "padding": "same"}, id="r1"),
               node("Activation", {"kind": "relu"}, id="r2"),
               node("MaxPool2d", {"kernel": 2}, id="r3"),
               node("GlobalAvgPool", {}, id="r4"),
               node("Linear", {"units": 64}, id="e2")],
        connect=[("right", "r1", 0)]),
   Step(title="Compare the embeddings",
        why="Concatenates the two and lets a small head decide whether they match.",
        nodes=[node("Concat", {"axis": 0}, id="pair")],
        connect=[("e1", "pair", 0), ("e2", "pair", 1)]),
   Step(title="Match or not",
        why="One output, binary.",
        nodes=[node("Linear", {"units": 32}), node("Activation", {"kind": "relu"}),
               node("Linear", {"units": 1}, label="head"),
               node("Output", {"task": "binary"})])],
  ["similarity", "siamese", "matching", "verification", "multiinput"],
  difficulty="advanced",
  data="Pairs with a match/no-match label.",
  caution="Weight sharing between the two branches is the usual design and is not expressible here yet.")

P("sim-embedding", "Learn an image embedding", "Similarity",
  "Train a general purpose embedding with contrastive learning, then compare "
  "anything by cosine distance without ever training a classifier.",
  simclr_encoder([3, 64, 64], 128), ["similarity", "embedding", "contrastive", "search"],
  difficulty="advanced", recipe="Contrastive",
  data="Unlabelled images.",
  expect="Within-class cosine similarity should clearly exceed between-class. That gap is the whole result.")

# ---------------------------------------------------------------- more vision
for pid, name, shape, classes, blurb, level in [
    ("document", "Document type", [1, 128, 128], 6, "Scanned pages sorted by kind.", "intermediate"),
    ("crack", "Structural crack detection", [3, 64, 64], 2, "Concrete surfaces, cracked or not.", "starter"),
    ("weather-cam", "Weather from a camera", [3, 64, 64], 5, "Outdoor images by condition.", "starter"),
    ("pill", "Pill identification", [3, 96, 96], 10, "Medication photographs.", "intermediate"),
    ("fuel-gauge", "Read a dial or gauge", [1, 64, 64], 10, "Digit or needle position from equipment photographs.", "intermediate"),
    ("receipt", "Receipt classification", [1, 128, 128], 4, "Sort scanned receipts by vendor type.", "intermediate"),
]:
    P(f"cnn-{pid}", name, "Vision",
      f"{blurb} A convolution stack sized to the input.",
      cnn_classifier(shape, classes, blocks=3 if shape[1] >= 96 else 2),
      ["image", "classification", "cnn", pid],
      difficulty=level, data="One folder per class.",
      training="AdamW at 1e-3, batch 32 or 64.",
      expect="If the training loss will not fall, check the class folders before touching the architecture.")

for arch in ("vgg16", "convnext_tiny"):
    P(f"transfer-{arch}", f"Transfer learning with {arch}", "Vision",
      f"Freeze {arch} and train a head on top.",
      transfer_classifier([3, 224, 224], 8, arch),
      ["image", "classification", "transfer", "pretrained", arch],
      difficulty="intermediate", data="A folder of images, one subfolder per class.",
      caution=f"{arch} is heavier than resnet18; check it trains at a workable speed before committing to it.")

# ---------------------------------------------------------------- more tabular
for pid, name, feats, out, task, blurb in [
    ("maintenance", "Predictive maintenance", 26, 2, "classification", "Flag equipment likely to fail."),
    ("pricing", "Dynamic price suggestion", 18, 1, "regression", "Suggest a price from market features."),
    ("staffing", "Shift demand", 14, 1, "regression", "Predict how many staff a shift needs."),
    ("inventory", "Stockout risk", 20, 2, "classification", "Flag products about to run out."),
    ("segment", "Customer segmentation", 24, 6, "classification", "Assign customers to segments."),
]:
    P(f"tab-{pid}", name, "Tabular",
      f"{blurb} An MLP over {feats} columns.",
      mlp_tabular(feats, out, task), ["tabular", "csv", task, pid],
      difficulty="starter",
      data=f"A CSV with about {feats} feature columns.",
      caution="Try a gradient-boosted tree first. On tabular data it usually wins, and knowing that saves you weeks.")

# ---------------------------------------------------------------- more sequences
for pid, name, length, ch, out, kind, blurb in [
    ("traffic", "Traffic volume forecast", 144, 3, 1, "conv", "Two days of readings at ten-minute spacing."),
    ("sleep", "Sleep stage scoring", 300, 3, 5, "lstm", "Epochs of physiological signal."),
    ("keystroke", "Typing pattern identity", 64, 4, 8, "lstm", "Timing between keystrokes."),
    ("vibration", "Bearing wear stage", 512, 2, 4, "conv", "Accelerometer traces from rotating equipment."),
    ("sales", "Weekly sales forecast", 52, 6, 1, "transformer", "A year of weekly figures with calendar features."),
]:
    P(f"seq-{pid}", f"{name} — {kind}", "Sequences",
      f"{blurb} Built with a {kind}.",
      sequence_model(length, ch, out, kind), ["sequence", "timeseries", kind, pid],
      difficulty="intermediate",
      data=f"Each row holds {length * ch} values reshaped to [{length}, {ch}].",
      caution="Split by time, not at random. A random split lets the model see the future and every metric becomes meaningless.")
