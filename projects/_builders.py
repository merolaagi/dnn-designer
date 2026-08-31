"""Step builders shared by the catalogue.

Each returns a list of Steps whose rationale quotes the actual numbers for that
instantiation, so "CNN on 32x32" and "CNN on 128x128 tiles" read differently
because they genuinely are different builds.
"""

from projects_sdk import Step, after_pools, commas, node


def cnn_classifier(shape, classes, width=32, blocks=2, norm=True):
    c, h, w = shape
    steps = [Step(
        title=f"Input — {c}x{h}x{w}",
        why=(f"Everything downstream is sized from here. Channels first, batch "
             f"left out: {c} channel{'s' if c > 1 else ''} at {h} by {w}. Get "
             f"this wrong and every shape after it is wrong too."),
        nodes=[node("Input", {"shape": list(shape)})],
        alternatives=("Smaller images train faster and lose fine detail; larger "
                      "ones cost time quadratically. 32 to 96 pixels is the usual "
                      "range for a first pass."),
    )]

    channels, size = width, h
    for i in range(blocks):
        prev = c if i == 0 else channels // 2
        steps.append(Step(
            title=f"Convolution block {i + 1} — {channels} filters",
            why=(f"A 3x3 convolution over {prev} channels produces {channels} "
                 f"feature maps, each looking at a 3x3 neighbourhood. Padding is "
                 f"'same', so the {size}x{size} grid survives this layer intact — "
                 f"only the pooling below shrinks it."
                 + (" Batch normalization sits between the convolution and the "
                    "activation, where it steadies the scale of what the "
                    "activation sees." if norm else "")),
            nodes=([node("Conv2d", {"filters": channels, "kernel": 3, "padding": "same"})]
                   + ([node("BatchNorm2d", {})] if norm else [])
                   + [node("Activation", {"kind": "relu"})]),
            alternatives=("A 5x5 kernel sees more at once but costs nearly three "
                          "times the parameters; two stacked 3x3 layers see the "
                          "same area more cheaply. Swap ReLU for GELU if you are "
                          "copying a modern architecture."),
        ))
        size = after_pools(size, 1)
        steps.append(Step(
            title=f"Pool to {size}x{size}",
            why=(f"Halves the grid to {size}x{size}, which cuts the work in the "
                 f"next block fourfold and widens what each later filter can see. "
                 f"Channels double as resolution halves — the standard trade, "
                 f"keeping roughly constant cost per block."),
            nodes=[node("MaxPool2d", {"kernel": 2})],
            alternatives=("A stride-2 convolution does the same downsampling and "
                          "learns how, at the cost of parameters. Average pooling "
                          "is gentler and occasionally better on smooth images."),
        ))
        channels *= 2

    flat = (channels // 2) * size * size
    steps.append(Step(
        title=f"Flatten — {commas(flat)} numbers",
        why=(f"The classifier head needs a vector, not a grid. This turns "
             f"[{channels // 2}, {size}, {size}] into [{commas(flat)}]. That is a "
             f"lot of numbers to hand a Linear layer, which is why the pooling "
             f"above matters so much."),
        nodes=[node("Flatten", {})],
        alternatives=("GlobalAvgPool instead would give just "
                      f"[{channels // 2}] — far fewer parameters and usually "
                      "better generalization on small datasets. Worth trying both."),
    ))
    steps.append(Step(
        title="Dropout",
        why=("Zeroes half the activations during training only. On a small "
             "dataset the head memorizes before the trunk generalizes, and this "
             "is the cheapest thing that helps."),
        nodes=[node("Dropout", {"rate": 0.5})],
        alternatives="Drop the rate to 0.2 if the training loss stops falling at all.",
    ))
    steps.append(Step(
        title=f"Classifier head — {classes} outputs",
        why=(f"One number per class, left as raw logits. Cross-entropy applies "
             f"its own softmax, so adding one here would apply it twice and "
             f"flatten the gradients."),
        nodes=[node("Linear", {"units": classes}, label="head"),
               node("Output", {"task": "classification"})],
        alternatives=("Add a hidden Linear before this if the task is hard, but "
                      "on most image problems the trunk does the work and a wider "
                      "head just overfits."),
        watch=f"The head must be exactly {classes} wide, matching your class count.",
    ))
    return steps


def resnet_classifier(shape, classes, width=64, blocks=3):
    c, h, w = shape
    steps = [
        Step(
            title=f"Input — {c}x{h}x{w}",
            why="Channels first, no batch dimension.",
            nodes=[node("Input", {"shape": list(shape)})],
        ),
        Step(
            title=f"Stem — {width} filters",
            why=(f"One plain convolution to lift {c} channels up to {width} before "
                 f"the residual blocks start. Residual blocks add their input to "
                 f"their output, so the channel count has to be established first."),
            nodes=[node("Conv2d", {"filters": width, "kernel": 3, "padding": "same"}),
                   node("BatchNorm2d", {}),
                   node("Activation", {"kind": "relu"})],
        ),
    ]
    size, channels = h, width
    for i in range(blocks):
        stride = 1 if i == 0 else 2
        if stride == 2:
            size = after_pools(size, 1)
            channels *= 2
        steps.append(Step(
            title=f"Residual block {i + 1} — {channels} filters"
                  + (f", stride 2 to {size}x{size}" if stride == 2 else ""),
            why=(f"Two convolutions with the input added back around them. That "
                 f"skip is the whole point: the gradient reaches earlier layers "
                 f"directly instead of being multiplied down through every layer "
                 f"in between, which is what made networks past about twenty "
                 f"layers trainable at all."
                 + (f" The stride of 2 halves the grid to {size}x{size}, and the "
                    f"block learns a 1x1 projection on the skip path so the "
                    f"shapes still line up." if stride == 2 else "")),
            nodes=[node("ResidualBlock", {"filters": channels, "stride": stride})],
            alternatives=("Add a SqueezeExcite after each block for channel "
                          "attention — cheap, and usually worth a point of accuracy."),
        ))
    steps.append(Step(
        title="Global average pool",
        why=(f"Averages each of the {channels} feature maps down to one number, "
             f"giving [{channels}]. Compare a Flatten here, which would give "
             f"[{commas(channels * size * size)}] and a head with orders of "
             f"magnitude more parameters. Global pooling is why modern "
             f"classifiers have small heads."),
        nodes=[node("GlobalAvgPool", {})],
    ))
    steps.append(Step(
        title=f"Head — {classes} outputs",
        why="One logit per class, straight off the pooled features.",
        nodes=[node("Linear", {"units": classes}, label="head"),
               node("Output", {"task": "classification"})],
    ))
    return steps


def transfer_classifier(shape, classes, arch="resnet18"):
    c, h, w = shape
    return [
        Step(
            title=f"Input — {c}x{h}x{w}",
            why=(f"{arch} was trained at 224x224. It will accept {h}x{w}, but the "
                 f"further you drift the less its learned features fit."),
            nodes=[node("Input", {"shape": list(shape)})],
            alternatives="224x224 matches the pretraining exactly and is the safe choice.",
        ),
        Step(
            title=f"Backbone — {arch}, frozen",
            why=(f"Loads {arch} with its ImageNet weights and strips the classifier, "
                 f"so what comes out is a feature map rather than ImageNet logits. "
                 f"Trainable stages is 0, meaning nothing in the backbone updates — "
                 f"you are training only your head on top of features that already "
                 f"know edges, textures and parts."),
            nodes=[node("Backbone", {"arch": arch, "weights": "DEFAULT",
                                     "trainable_stages": 0})],
            alternatives=("Raise trainable stages to 1 or 2 once the head has "
                          "settled, and drop the learning rate when you do. "
                          "Unfreezing early destroys the pretrained features "
                          "before the head can make use of them."),
            watch="First run downloads the weights, so it needs a network connection.",
        ),
        Step(
            title="Global average pool",
            why=("Collapses the feature map to one number per channel. The "
                 "backbone has already done the spatial reasoning."),
            nodes=[node("GlobalAvgPool", {})],
        ),
        Step(
            title="Dropout",
            why=("With a frozen backbone the head is the only thing learning, and "
                 "a small head on rich features overfits fast."),
            nodes=[node("Dropout", {"rate": 0.3})],
        ),
        Step(
            title=f"Head — {classes} outputs",
            why=(f"The only part being trained at first. {classes} logits, one per "
                 f"class."),
            nodes=[node("Linear", {"units": classes}, label="head"),
                   node("Output", {"task": "classification"})],
            watch=("Transfer learning wants a much lower learning rate than "
                   "training from scratch. Start at 1e-3 and go down, not up."),
        ),
    ]


def mlp_tabular(features, outputs, task="classification", width=64, depth=2):
    steps = [Step(
        title=f"Input — {features} features",
        why=(f"One number per column, so the Input is [{features}]. The training "
             f"form hands columns to Inputs in order, so this width has to match "
             f"how many feature columns you have after removing the target."),
        nodes=[node("Input", {"shape": [features]})],
    )]
    for i in range(depth):
        units = width // (2 ** i) or 1
        steps.append(Step(
            title=f"Hidden layer {i + 1} — {units} units",
            why=(f"A fully connected layer mixing every feature with every other. "
                 f"{units} units is wider than the {features} inputs"
                 if units > features else
                 f"A fully connected layer narrowing {features} inputs to {units} "
                 f"units, forcing it to keep only what matters"),
            nodes=[node("Linear", {"units": units}),
                   node("Activation", {"kind": "relu"})],
            alternatives=("Add BatchNorm1d before the activation if training is "
                          "unstable. On tabular data a gradient-boosted tree is "
                          "often still the stronger baseline — worth knowing "
                          "before investing in a deep model."),
        ))
    steps.append(Step(
        title="Dropout",
        why="Tabular models overfit quickly because the features are few and dense.",
        nodes=[node("Dropout", {"rate": 0.2})],
    ))
    if task == "regression":
        steps.append(Step(
            title="Output — one number",
            why=("A single unit with no activation. Squashing it would cap what "
                 "the model can predict."),
            nodes=[node("Linear", {"units": 1}, label="head"),
                   node("Output", {"task": "regression"})],
            watch="Standardize the target as well as the features, or the loss will be dominated by scale.",
        ))
    else:
        steps.append(Step(
            title=f"Output — {outputs} classes",
            why=f"One logit per class.",
            nodes=[node("Linear", {"units": outputs}, label="head"),
                   node("Output", {"task": "classification"})],
        ))
    return steps


def sequence_model(length, channels, outputs, kind="lstm", units=128):
    steps = [Step(
        title=f"Input — {length} steps of {channels}",
        why=(f"Sequences are [L, C]: {length} time steps, {channels} value"
             f"{'s' if channels > 1 else ''} at each. Time along the first axis, "
             f"features along the second."),
        nodes=[node("Input", {"shape": [length, channels]})],
    )]
    if kind == "lstm":
        steps.append(Step(
            title=f"LSTM — {units} units, last step only",
            why=(f"Walks the {length} steps in order, carrying a state that lets "
                 f"it remember what happened earlier. Return sequences is off, so "
                 f"it emits only the final state — one summary of the whole "
                 f"sequence, which is what a single prediction needs."),
            nodes=[node("LSTM", {"units": units, "return_sequences": False})],
            alternatives=("A GRU is cheaper and often just as good. Turn "
                          "bidirectional on if the whole sequence is available at "
                          "once — but not if you are predicting the future, where "
                          "reading backwards is cheating."),
        ))
    elif kind == "conv":
        steps.extend([
            Step(
                title="Conv1d — 64 filters over time",
                why=(f"Slides a kernel along the {length} steps, learning local "
                     f"patterns regardless of where they occur. Far faster than a "
                     f"recurrent layer because every position computes at once."),
                nodes=[node("Conv1d", {"filters": 64, "kernel": 5, "padding": "same"}),
                       node("Activation", {"kind": "relu"})],
                alternatives="Stack a second Conv1d with dilation to see further back cheaply.",
            ),
            Step(
                title="Pool the time axis away",
                why="Averages across time, leaving one vector describing the whole window.",
                nodes=[node("GlobalAvgPool", {})],
            ),
        ])
    else:
        steps.extend([
            Step(
                title=f"Widen to {((channels + 3) // 4) * 4} channels",
                why=(f"Attention splits channels across heads, so the width has to "
                     f"divide by the head count. {channels} does not divide by 4, "
                     f"so a Linear lifts it to {((channels + 3) // 4) * 4} first. "
                     f"Skip this and the graph will refuse to resolve."),
                nodes=[node("Linear", {"units": ((channels + 3) // 4) * 4})],
            ),
            Step(
                title="Positional encoding",
                why=("Attention has no inherent sense of order — shuffle the steps "
                     "and it returns the same answer. This adds a fixed signal "
                     "encoding each position, which is what makes order visible."),
                nodes=[node("PositionalEncoding", {})],
            ),
            Step(
                title="Transformer encoder",
                why=(f"Every step attends to every other, so a value at step 1 can "
                     f"directly influence step {length} without passing through "
                     f"anything in between. That is the advantage over an LSTM, "
                     f"and it costs memory quadratic in the {length} steps."),
                nodes=[node("TransformerEncoder", {"heads": 4, "ff_dim": 256, "depth": 2})],
                watch=f"Channels must divide by the head count: {channels} into 4 heads.",
            ),
            Step(
                title="Pool across time",
                why="Averages the per-step outputs into one vector for the head.",
                nodes=[node("GlobalAvgPool", {})],
            ),
        ])
    steps.append(Step(
        title=f"Head — {outputs} output{'s' if outputs > 1 else ''}",
        why="Maps the sequence summary to the answer.",
        nodes=[node("Linear", {"units": outputs}, label="head"),
               node("Output", {"task": "classification" if outputs > 1 else "regression"})],
    ))
    return steps


def char_language_model(context, vocab, dim=128, depth=4, heads=4):
    return [
        Step(
            title=f"Input — {context} token ids",
            why=(f"Shape [{context}] with dtype long. These are integer ids, not "
                 f"numbers to do arithmetic on, which is why the dtype matters — "
                 f"an Embedding looks them up rather than multiplying them."),
            nodes=[node("Input", {"shape": [context], "dtype": "long"}, label="tokens")],
            alternatives=("Longer context sees further back and costs memory "
                          "quadratically through attention."),
        ),
        Step(
            title=f"Embedding — {vocab} x {dim}",
            why=(f"A learned vector of {dim} numbers per character, giving "
                 f"[{context}, {dim}]. The table has {commas(vocab * dim)} "
                 f"parameters and is often the largest single layer in a small "
                 f"language model."),
            nodes=[node("Embedding", {"vocab": vocab, "dim": dim})],
            watch=f"Vocab must match your corpus exactly — {vocab} distinct characters.",
        ),
        Step(
            title="Positional encoding",
            why=("Attention is order-blind. Without this, 'abc' and 'cba' produce "
                 "identical outputs, which makes next-character prediction "
                 "impossible."),
            nodes=[node("PositionalEncoding", {})],
        ),
        Step(
            title=f"GPT stack — {depth} blocks, {heads} heads",
            why=(f"{depth} pre-norm decoder blocks. Causal masking means position "
                 f"{context} can see everything before it and nothing after — "
                 f"without that mask the model reads the answer and the loss goes "
                 f"to zero while it learns nothing."),
            nodes=[node("GPTStack", {"depth": depth, "heads": heads, "dropout": 0.1})],
            watch=f"{dim} channels must divide by {heads} heads.",
        ),
        Step(
            title=f"Language head — {vocab} outputs",
            why=(f"Projects each position back to a distribution over the {vocab} "
                 f"characters. Applied at every position, so one forward pass "
                 f"produces {context} predictions and {context} training signals."),
            nodes=[node("Linear", {"units": vocab, "bias": False}, label="lm_head"),
                   node("Output", {"task": "language_modeling"})],
        ),
        Step(
            title="Text generator",
            why=("A runtime block, not a layer. Sampling is a loop that calls the "
                 "model once per character — it produces no activation and sits "
                 "outside forward()."),
            nodes=[node("TextGenerator", {"block_size": context, "temperature": 0.8})],
        ),
    ]
