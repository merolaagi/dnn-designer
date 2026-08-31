"""Builders for the generative, self-supervised, structural and agent projects."""

from projects_sdk import Step, after_pools, commas, node


def autoencoder(shape, code, denoising=False):
    flat = shape[0] if len(shape) == 1 else shape[0] * shape[1] * shape[2]
    steps = [Step(
        title=f"Input — {shape}",
        why=("With an autoencoder the input is also the target. There is no label "
             "anywhere in this project."),
        nodes=[node("Input", {"shape": list(shape)})],
    )]
    if len(shape) == 3:
        steps.append(Step(
            title="Flatten",
            why=f"Turns the image into [{commas(flat)}] so the encoder can be dense layers.",
            nodes=[node("Flatten", {})],
            alternatives="A convolutional encoder and ConvTranspose2d decoder keeps spatial structure and works better on larger images.",
        ))
    steps += [
        Step(
            title=f"Encoder — down to {code}",
            why=(f"Squeezes {commas(flat)} numbers into {code}. The bottleneck is "
                 f"the entire point: the network cannot copy its input through "
                 f"{code} numbers, so it has to learn what is worth keeping."),
            nodes=[node("Linear", {"units": max(code * 8, code)}),
                   node("Activation", {"kind": "relu"}),
                   node("Linear", {"units": code}, label="code"),
                   node("Activation", {"kind": "tanh"})],
            alternatives=(f"A wider code reconstructs better and compresses less. "
                          f"At {code} you are asking for roughly "
                          f"{flat // max(code, 1)}x compression."),
        ),
        Step(
            title="Decoder — back up",
            why=("Mirrors the encoder. It only ever sees the code, so whatever it "
                 "reconstructs is what the code managed to carry."),
            nodes=[node("Linear", {"units": max(code * 8, code)}),
                   node("Activation", {"kind": "relu"}),
                   node("Linear", {"units": flat})],
        ),
    ]
    if len(shape) == 3:
        steps.append(Step(
            title="Reshape back to the image",
            why=f"The recipe compares output against input, so the shape has to return to {shape}.",
            nodes=[node("Reshape", {"shape": list(shape)})],
        ))
    steps.append(Step(
        title="Output",
        why="The Autoencoder recipe supplies the target, so the task here is ignored.",
        nodes=[node("Output", {"task": "regression"})],
        watch=("Choose the Autoencoder recipe in the Training tab. The standard "
               "loop expects labels and there are none."
               + (" Set noise above zero to make it a denoising autoencoder, which "
                  "learns far more useful features." if denoising else "")),
    ))
    return steps


def gan_generator(shape, latent=64):
    c, h, w = shape
    start = max(4, after_pools(h, 2))
    return [
        Step(
            title=f"Input — {latent} random numbers",
            why=(f"A generator takes noise, not an image. Every different draw of "
                 f"{latent} numbers should become a different picture."),
            nodes=[node("Input", {"shape": [latent]}, label="noise")],
        ),
        Step(
            title=f"Project to a {start}x{start} grid",
            why=(f"Expands {latent} numbers into {commas(128 * start * start)} and "
                 f"reshapes them into 128 feature maps at {start}x{start}. This is "
                 f"where a vector becomes something spatial that convolutions can "
                 f"work on."),
            nodes=[node("Linear", {"units": 128 * start * start}),
                   node("Reshape", {"shape": [128, start, start]}),
                   node("Activation", {"kind": "relu"})],
        ),
        Step(
            title=f"Upsample to {start * 2}x{start * 2}",
            why=("A transposed convolution doubles the resolution and learns how "
                 "to fill in the detail, rather than interpolating it."),
            nodes=[node("ConvTranspose2d", {"filters": 64, "kernel": 4, "stride": 2, "padding": 1}),
                   node("BatchNorm2d", {}),
                   node("Activation", {"kind": "relu"})],
            alternatives="Upsample2d followed by a plain Conv2d avoids the checkerboard artefacts transposed convolutions can produce.",
        ),
        Step(
            title=f"Upsample to {start * 4}x{start * 4}",
            why="Second doubling, reaching the target resolution.",
            nodes=[node("ConvTranspose2d", {"filters": 32, "kernel": 4, "stride": 2, "padding": 1}),
                   node("Activation", {"kind": "relu"})],
        ),
        Step(
            title=f"To {c} channels, squashed",
            why=(f"A 1x1 convolution mixes {32} feature maps down to the {c} the "
                 f"image needs, and sigmoid bounds the result to [0, 1] — the same "
                 f"range the real images arrive in. A generator whose output range "
                 f"differs from the data is trivially detectable."),
            nodes=[node("Conv2d", {"filters": c, "kernel": 3, "padding": "same"}),
                   node("Activation", {"kind": "sigmoid"}),
                   node("Output", {})],
            watch=("Save a discriminator separately — image in, one number out — "
                   "and pick it in the GAN recipe. Watch d_accuracy: near 0.5 is "
                   "balanced, near 1.0 means the discriminator has won and the "
                   "generator has stopped learning."),
        ),
    ]


def diffusion_unet(shape, width=32):
    c, h, w = shape
    return [
        Step(
            title=f"Input — {c + 1} channels, not {c}",
            why=(f"The extra channel carries the timestep. The model has to know "
                 f"how noisy its input is — predicting the noise at step 5 and at "
                 f"step 500 are different problems — and this is the simplest way "
                 f"to tell it without a special conditioning layer."),
            nodes=[node("Input", {"shape": [c + 1, h, w]})],
            watch=f"Input {c + 1} channels, Output {c}. The recipe checks this and will refuse otherwise.",
        ),
        Step(
            title=f"Encoder — {width} filters",
            why="First convolution, kept at full resolution so the skip connection later has fine detail to reuse.",
            nodes=[node("Conv2d", {"filters": width, "kernel": 3, "padding": "same"}),
                   node("Activation", {"kind": "silu"}, id="fine")],
            alternatives="SiLU is the usual choice in diffusion models; ReLU's hard zero discards information the denoiser needs.",
        ),
        Step(
            title=f"Downsample to {h // 2}x{h // 2}",
            why="Halving the resolution lets the next convolutions see a wider area for the same cost.",
            nodes=[node("MaxPool2d", {"kernel": 2}),
                   node("Conv2d", {"filters": width * 2, "kernel": 3, "padding": "same"}),
                   node("Activation", {"kind": "silu"})],
        ),
        Step(
            title=f"Upsample back to {h}x{h}",
            why="A transposed convolution returns to full resolution so the output can be a full-size noise prediction.",
            nodes=[node("ConvTranspose2d", {"filters": width, "kernel": 4, "stride": 2, "padding": 1},
                        id="up")],
        ),
        Step(
            title="Skip connection",
            why=("Concatenates the full-resolution features from before the "
                 "downsample onto the upsampled ones. Without this the fine detail "
                 "destroyed by pooling never comes back, and a denoiser that "
                 "cannot resolve detail produces mush."),
            nodes=[node("Concat", {"axis": 0}, id="skip")],
            connect_from="__none__",
            connect=[("fine", "skip", 0), ("up", "skip", 1)],
        ),
        Step(
            title=f"Predict the noise — {c} channels",
            why=(f"Outputs {c} channels the same size as the image: an estimate of "
                 f"exactly the noise that was added. Sampling subtracts this, "
                 f"repeatedly, from pure noise."),
            nodes=[node("Conv2d", {"filters": width, "kernel": 3, "padding": "same"}, id="tail"),
                   node("Activation", {"kind": "silu"}),
                   node("Conv2d", {"filters": c, "kernel": 1, "padding": "same"}),
                   node("Output", {})],
            connect_from="skip",
            watch="Choose the Diffusion recipe. The preview reports the sampled range against the data range each epoch.",
        ),
    ]


def simclr_encoder(shape, embedding=128):
    c, h, w = shape
    return [
        Step(
            title=f"Input — {c}x{h}x{w}",
            why="Unlabelled images. This project never uses a label at any point.",
            nodes=[node("Input", {"shape": list(shape)})],
        ),
        Step(
            title="Convolution trunk",
            why=("An ordinary feature extractor. What makes this self-supervised "
                 "is the training loop, not the architecture — the same trunk "
                 "would work for classification."),
            nodes=[node("Conv2d", {"filters": 32, "kernel": 3, "padding": "same"}),
                   node("BatchNorm2d", {}),
                   node("Activation", {"kind": "relu"}),
                   node("MaxPool2d", {"kernel": 2}),
                   node("Conv2d", {"filters": 64, "kernel": 3, "padding": "same"}),
                   node("Activation", {"kind": "relu"})],
        ),
        Step(
            title="Pool to a vector",
            why="Collapses the spatial grid so what remains describes the whole image.",
            nodes=[node("GlobalAvgPool", {})],
        ),
        Step(
            title=f"Projection head — {embedding}",
            why=(f"The contrastive loss is computed on these {embedding} numbers. "
                 f"Two augmented views of one image should land close together, "
                 f"and every other image in the batch should be far away — which "
                 f"is why batch size matters here more than usual: the batch "
                 f"supplies all the negatives."),
            nodes=[node("Linear", {"units": embedding}, label="projection"),
                   node("Output", {})],
            watch=("Choose the Contrastive recipe. After training, start a "
                   "classifier from this checkpoint — that is the payoff, not the "
                   "loss number itself."),
        ),
    ]


def policy_network(observations, actions, width=64):
    return [
        Step(
            title=f"Input — {observations} observations",
            why=(f"Whatever the environment reports each step. CartPole gives "
                 f"{observations}: position, velocity, angle and angular velocity."),
            nodes=[node("Input", {"shape": [observations]}, label="observation")],
        ),
        Step(
            title=f"Hidden layer — {width} units",
            why=("Small on purpose. Policy gradients are noisy, and a large "
                 "network fits that noise faster than it fits the task."),
            nodes=[node("Linear", {"units": width}),
                   node("Activation", {"kind": "tanh"})],
            alternatives="Tanh is conventional in policy networks — bounded activations keep the logits from swinging wildly early on.",
        ),
        Step(
            title=f"Action logits — {actions}",
            why=(f"One logit per action. The recipe samples from these rather than "
                 f"taking the maximum, which is where exploration comes from: a "
                 f"policy that always picks its current best never discovers "
                 f"anything better."),
            nodes=[node("Linear", {"units": actions}, label="policy"),
                   node("Output", {})],
            watch=("Choose the Reinforce recipe. Watch the return, not the loss — "
                   "the loss in policy gradients is not a quantity that "
                   "meaningfully decreases."),
        ),
    ]


def detector(shape, classes=2, grid=8):
    c, h, w = shape
    downs = 0
    size = h
    while size > grid:
        size //= 2
        downs += 1
    steps = [Step(
        title=f"Input — {c}x{h}x{w}",
        why=f"Images to find objects in. The head below will predict on a {grid}x{grid} grid over this.",
        nodes=[node("Input", {"shape": list(shape)})],
    )]
    filters = 32
    for i in range(downs):
        steps.append(Step(
            title=f"Downsample {i + 1} — to {h // (2 ** (i + 1))}x{h // (2 ** (i + 1))}",
            why=(f"A stride-2 convolution rather than pooling, so the downsampling "
                 f"is learned. Detection needs to preserve where things are, and a "
                 f"learned stride does that better than max pooling, which throws "
                 f"position away within each window."),
            nodes=[node("Conv2d", {"filters": filters, "kernel": 3, "stride": 2, "padding": 1}),
                   node("BatchNorm2d", {}),
                   node("Activation", {"kind": "relu"})],
        ))
        filters *= 2
    steps.append(Step(
        title=f"Detection head — {5 + classes} channels on {grid}x{grid}",
        why=(f"A 1x1 convolution producing {5 + classes} numbers at each of the "
             f"{grid * grid} cells: one objectness score, four box numbers, and "
             f"{classes} class logits. Every cell is responsible for objects whose "
             f"centre falls inside it."),
        nodes=[node("Conv2d", {"filters": 5 + classes, "kernel": 1, "padding": "same"},
                    label="head"),
               node("Output", {})],
        watch=("Choose the Detection recipe. It draws its own shapes, so this "
               "trains without annotated data — real annotations still need a "
               "loader that reads them."),
    ))
    return steps


def mil_aggregator(features, classes, heads=4):
    return [
        Step(
            title=f"Input — a bag of tile features [N, {features}]",
            why=(f"Not an image. Each slide has already been cut into tiles and "
                 f"each tile turned into {features} numbers by a frozen encoder. "
                 f"This is what makes gigapixel slides tractable: the expensive "
                 f"part happens once, offline."),
            nodes=[node("Input", {"shape": [64, features]}, label="tiles")],
            alternatives="The first dimension is tiles per bag. Fix it for now; variable bags need a loader that pads.",
        ),
        Step(
            title="Project the tile features",
            why=(f"Brings {features} encoder dimensions down to something the "
                 f"attention layer can work with cheaply, and lets the model adapt "
                 f"features that were learned for a different task."),
            nodes=[node("Linear", {"units": 256}),
                   node("Activation", {"kind": "relu"})],
        ),
        Step(
            title=f"Attention across tiles — {heads} heads",
            why=("This is the multiple-instance step. There is one label for the "
                 "whole slide and no label per tile, so the model has to work out "
                 "which tiles matter. Attention weights give you that, and they "
                 "map back onto the slide as a heatmap — which is the only reason "
                 "a pathologist would trust the number."),
            nodes=[node("SelfAttention", {"heads": heads})],
            alternatives="A plain mean over tiles is the baseline. If attention does not beat it, the problem is the features or the labels, not the aggregator.",
        ),
        Step(
            title="Pool the bag to one vector",
            why="Averages the attended tile features into a single slide-level representation.",
            nodes=[node("GlobalAvgPool", {})],
        ),
        Step(
            title=f"Differential head — {classes} outputs",
            why=(f"{classes} independent sigmoid outputs rather than a softmax, "
                 f"because a differential is a set of possibilities that can "
                 f"co-occur, not a single winner."),
            nodes=[node("Dropout", {"rate": 0.25}),
                   node("Linear", {"units": classes}, label="head"),
                   node("Output", {"task": "binary"})],
            watch=("Split by site and scanner, not by slide. Models learn the "
                   "scanner's colour profile and then collapse on outside data — "
                   "this is the failure that ends most computational pathology "
                   "projects, and it is invisible if you split randomly."),
        ),
    ]


def graph_classifier(nodes_per_graph, features, classes):
    return [
        Step(
            title=f"Node features — [{nodes_per_graph}, {features}]",
            why=f"{nodes_per_graph} nodes, each described by {features} numbers.",
            nodes=[node("Input", {"shape": [nodes_per_graph, features]}, id="feats",
                        label="nodes")],
        ),
        Step(
            title=f"Adjacency — [{nodes_per_graph}, {nodes_per_graph}]",
            why=("A second Input holding who connects to whom. Graph convolution "
                 "needs both, which is why this project has two Inputs."),
            nodes=[node("Input", {"shape": [nodes_per_graph, nodes_per_graph]}, id="adj",
                        label="adjacency")],
            connect_from="__none__",
        ),
        Step(
            title="Graph convolution — 64 units",
            why=("Averages each node's neighbours, then projects. After one round "
                 "every node knows about its immediate neighbours; the receptive "
                 "field grows one hop per layer."),
            nodes=[node("GraphConv", {"units": 64}, id="g1")],
            connect_from="__none__",
            connect=[("feats", "g1", 0), ("adj", "g1", 1)],
        ),
        Step(
            title="Second graph convolution",
            why="Two hops. Most small-molecule and citation tasks saturate around two or three — deeper tends to smear every node toward the same value.",
            nodes=[node("Activation", {"kind": "relu"}, id="ga"),
                   node("GraphConv", {"units": 32}, id="g2")],
            connect_from="g1",
            connect=[("adj", "g2", 1)],
        ),
        Step(
            title="Pool the nodes",
            why="Averages across nodes to describe the whole graph rather than any one node.",
            nodes=[node("GlobalAvgPool", {})],
        ),
        Step(
            title=f"Head — {classes} classes",
            why="One logit per graph-level class.",
            nodes=[node("Linear", {"units": classes}, label="head"),
                   node("Output", {"task": "classification"})],
            watch="No graph dataset loader ships yet, so train this on synthetic data for now.",
        ),
    ]
