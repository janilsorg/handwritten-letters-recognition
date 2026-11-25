import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    # Reading guide for readers of the article "From Digits to Letters..." (reports/v2/03_article_ieee_v2).
    mo.md(r"""
    # From Digits to Letters: annotated copy of the extension notebook

    This is the experiment behind the article. The code is the same as in
    `../02_extension_letters_chars74k.py`. Only the comments and the markdown cells are new. They
    are written for someone who has the article open next to the notebook.

    **How to read the comments.** Every cell starts with an `ARTICLE ▸` line naming the section,
    table, figure or equation it produces or supports. A `NOTE ▸` line marks the places where the
    number in the article is *not* the number this cell prints, and says where the article's number
    comes from.

    **One notebook, two machines.** The article pools two complete runs of this notebook, one on a Mac
    (Apple GPU, MPS) and one on a Windows PC (RTX 4060 Ti, CUDA), with the same seeds 0–4 (Section
    III-C, Table III). A single run of the notebook gives **5 runs per model**, while the article's
    tables show the **mean of 10**. The numbers will be close but not identical. Some statistics in the
    article need both machines or the saved models, and are computed after the notebook by the report
    scripts in `report/py_file/v2/`:

    | In the article | Computed by |
    |---|---|
    | Means over 10 runs, 95% CIs of the effects (Table V, Fig. 2), Holm's correction | `data.py` (pools both machines' CSVs) |
    | Writer-level bootstrap on Chars74K, probit fit and effective robustness (Table IX, Fig. 5) | `data.py`, using `reanalysis.py` output |
    | Training accuracy with dropout off (Sec. IV-B, claim C7) | `reanalysis.py` |
    | Chars74K accuracy at stroke levels 2–6, I/l merged (Sec. V-B, Fig. 4a) | `reanalysis.py` |
    | Deployable 99% threshold fixed on EMNIST validation (Sec. V-E) | `reanalysis.py` |
    | Seed-vs-machine comparison, flipped verdicts (Section VI, Tables X–XI, Fig. 8) | `data.py` |

    **Map of this notebook to the article.**

    | Notebook part | Article |
    |---|---|
    | Setup, protocol, seeds | III-C Two Machines, III-D Protocol and Statistics, Table III |
    | 1. Data | III-A Two Letters Datasets, Table II |
    | 2. Models | II The Base Model, III-B Models |
    | 3. Training | III-D |
    | Part A: accuracy and cost | IV-A The Fall, IV-B The Search for a Fix, IV-C The Price of Each Point; Tables IV–VI; Figs. 1–3 |
    | Part G: reliability | III-E Reliability Metrics, IV-D Can the Confidence be Trusted?; Tables VII–VIII |
    | Part E: Chars74K | V Comparing on a Second Letters Dataset; Table IX; Figs. 4–7 |
    | Save results | Input to every table and to Section VI (one folder per machine) |

    **Names.** The notebook and the article name the models slightly differently:
    `Baseline 8/16` = the *baseline*; `BatchNorm 8/16` = *batch normalization*; `LeNet-5` = the
    *LeNet-5-style network*; `KNN (k=3)` = *k-NN*; `mnist` = *digits*; `letters` = *EMNIST Letters*.
    Claims C1–C9 are those of Table I.

    **Running it.** A full run is 2 datasets × 10 models × 5 seeds = 100 short trainings (1–2 h on a
    laptop GPU). `EXT_QUICK=1` runs a 2-epoch smoke test into a separate folder. Its numbers are
    not the article's.
    """)
    return


@app.cell
def _():
    # ARTICLE ▸ Table III (Execution Environments): Python, PyTorch and scikit-learn versions and the GPU.
    #           These values are saved to environment.json by the last cell of the notebook, and the
    #           article reads them from there for each machine.
    #
    # Imports for the whole notebook. Compared with the replication notebook, the new ones are for:
    #   math, string                 McNemar p-values (Eq. 2) and letter names
    #   json, Path                   saving and reloading trained runs and result tables
    #   f1_score, roc_auc_score      macro-F1 (Part A) and OOD detection (Table VIII)
    #   LogisticRegression, KNeighborsClassifier, SVC   the classical baselines of Table IV
    #   PIL, scipy.ndimage, download_and_extract_archive   Chars74K download and preprocessing (Sec. III-A)
    import copy
    import json
    import math
    import os
    import platform
    import random
    import string
    import time
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import sklearn
    import torch
    import torch.nn as nn
    import torchvision
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix, f1_score,
                                 precision_recall_fscore_support, roc_auc_score)
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from torch.utils.data import DataLoader, TensorDataset
    from PIL import Image, ImageFilter
    from scipy import ndimage
    from torchvision import datasets
    from torchvision.datasets.utils import download_and_extract_archive

    # Training device: CUDA on the Windows machine, MPS on the Mac (Table III); CPU as a fallback
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")

    print(f"Python       : {platform.python_version()}")
    print(f"NumPy        : {np.__version__}")
    print(f"PyTorch      : {torch.__version__}")
    print(f"torchvision  : {torchvision.__version__}")
    print(f"scikit-learn : {sklearn.__version__}")
    print(f"Platform     : {platform.platform()}")
    print(f"CPU          : {platform.processor()}")
    print(f"Device       : {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU          : {torch.cuda.get_device_name(0)}")
    return (
        ConfusionMatrixDisplay,
        DEVICE,
        DataLoader,
        Image,
        ImageFilter,
        KNeighborsClassifier,
        LogisticRegression,
        Path,
        SVC,
        TensorDataset,
        confusion_matrix,
        copy,
        datasets,
        download_and_extract_archive,
        f1_score,
        json,
        math,
        ndimage,
        nn,
        np,
        os,
        pd,
        platform,
        plt,
        precision_recall_fscore_support,
        random,
        roc_auc_score,
        sklearn,
        string,
        time,
        torch,
        torchvision,
    )


@app.cell
def _(Path, os):
    # ARTICLE ▸ Sec. III-D Protocol and Statistics: "All models use the recipe of [1] unchanged ... Each
    #           configuration is trained with seeds 0–4 on each machine". Also Sec. III-E (ECE over 15
    #           bins), Sec. III-B (the SVM uses a 20,000-image subsample) and Sec. III-A (stroke levels).
    #
    # Every experimental setting of the study, in one place. Everything marked "paper" is copied
    # unchanged from Mohammed [1]. The study adds the seeds, the datasets and the metrics, not a new
    # training recipe. This is why the article says its architectural conclusions hold "at a 10-epoch
    # budget".
    QUICK = os.environ.get("EXT_QUICK", "0") == "1"   # smoke test only; never used for the article

    DATASETS = ["mnist", "letters"]   # digits (reference) and EMNIST Letters (the extension), Table II
    SPLIT_SEED = 42                   # paper: seed 42 for the train/validation split; the same in every run
    VAL_FRACTION = 0.10               # paper: 6,000 of 60,000 = 10%; same fraction for EMNIST Letters
    SEEDS = [0] if QUICK else [0, 1, 2, 3, 4]   # 5 seeds per machine -> 10 runs per model in the article
    EPOCHS = 2 if QUICK else 10       # paper: 10 epochs
    LR = 1e-3                         # paper: Adam, learning rate 0.001, no weight decay, no schedule
    BATCH_SIZE = 128                  # paper: batch size 128
    EVAL_BATCH_SIZE = 1000            # evaluation only; affects speed, not results
    N_BINS = 15                       # number of confidence bins in the ECE (Eq. 4)
    QUICK_TRAIN_LIMIT = 10_000        # QUICK mode trains on a subset only
    SVM_TRAIN_LIMIT = 2_000 if QUICK else 20_000  # RBF SVM is too slow on 112k images (Sec. III-B)

    # Where trained models and result tables are written. The article's two runs are stored as
    # report/results/mac_mps and report/results/windows_cuda, each a copy of one such folder.
    RESULTS_DIR = Path("results_extension") / ("quick" if QUICK else "full")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # EXT_RUNS_DIR lets Part E reuse models trained earlier instead of training them again
    RUNS_DIR = Path(os.environ["EXT_RUNS_DIR"]) if "EXT_RUNS_DIR" in os.environ else RESULTS_DIR / "runs"
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    # EXT_DATA_DIR reuses an existing data folder (EMNIST is a 535 MB download)
    DATA_DIR = Path(os.environ.get("EXT_DATA_DIR", "data"))

    # Chars74K, the second letters dataset (Sec. III-A): the handwritten "EnglishHnd" subset, ~13 MB
    C74K_URL = "http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishHnd.tgz"
    C74K_DIR = DATA_DIR / "chars74k"
    # Candidate stroke-thinning levels k of Eq. (1). Level 0 = no thinning. Level 1 = strokes eroded by
    # one EMNIST pixel (1/128 of the frame), and so on. The level is chosen later, by ink, not by accuracy.
    C74K_STROKE_LEVELS = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    print(f"QUICK={QUICK} | datasets={DATASETS} | seeds={SEEDS} | epochs={EPOCHS} | results -> {RESULTS_DIR}"
          f" | runs -> {RUNS_DIR} | data -> {DATA_DIR}")
    return (
        BATCH_SIZE,
        C74K_DIR,
        C74K_STROKE_LEVELS,
        C74K_URL,
        DATASETS,
        DATA_DIR,
        EPOCHS,
        EVAL_BATCH_SIZE,
        LR,
        N_BINS,
        QUICK,
        QUICK_TRAIN_LIMIT,
        RESULTS_DIR,
        RUNS_DIR,
        SEEDS,
        SPLIT_SEED,
        SVM_TRAIN_LIMIT,
        VAL_FRACTION,
    )


@app.cell
def _(np, random, torch):
    # ARTICLE ▸ Sec. III-C Two Machines: "Initial weights and the batch order are drawn on the CPU from the
    #           seed, so they should agree across machines; dropout masks are drawn on the GPU, and GPU
    #           kernels are not bit-identical across platforms."
    #
    # Each training run calls this with its own seed (0–4). The seed fixes the initial weights and the
    # shuffle order, so "seed 3 on the Mac" and "seed 3 on Windows" start from the same point. This is
    # what lets Section VI compare same-seed runs across machines. The data split does not depend on
    # this seed: it always uses seed 42 (next cells), so every run sees the same images.
    def set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return (set_seed,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Data: MNIST and EMNIST Letters (article: Sec. III-A, Table II)

    EMNIST Letters keeps MNIST's 28×28 grey-scale format and NIST origin but has 26 classes, with
    upper and lower case merged (class "a" contains both "a" and "A"). It has 124,800 training images
    (112,320 train + 12,480 validation after the 10% split) and 20,800 test images, 800 per class.
    Two quirks of the files are fixed here: the images are stored **transposed**, and the labels run
    **1–26** instead of 0–25 ("its images are transposed back and the labels shifted to 0–25").
    """)
    return


@app.cell
def _(
    DATASETS,
    DATA_DIR,
    QUICK,
    QUICK_TRAIN_LIMIT,
    SPLIT_SEED,
    VAL_FRACTION,
    datasets,
    string,
    torch,
):
    # ARTICLE ▸ Table II (Datasets): the "Digits" and "Letters" columns, i.e. classes, train/validation
    #           sizes (54,000 / 6,000 and 112,320 / 12,480) and test sizes (10,000 and 20,800).
    #
    # Both datasets go through the same function, with the same preprocessing (pixels / 255, which is the
    # paper's only preprocessing) and the same split (seed 42, first 10% to validation). The only
    # dataset-specific lines are the two EMNIST fixes. So when the article reports a drop from digits to
    # letters, the drop comes from the data, not from a different pipeline.
    def _to_tensor(images):
        # Scale to [0, 1] and add a channel axis -> (N, 1, 28, 28)
        return images.float().div(255.0).unsqueeze(1)

    def load_dataset(name: str) -> dict:
        if name == "mnist":
            train_raw = datasets.MNIST(root=DATA_DIR, train=True, download=True)
            test_raw = datasets.MNIST(root=DATA_DIR, train=False, download=True)
            X_full, y_full = train_raw.data, train_raw.targets.clone()
            X_test, y_test = test_raw.data, test_raw.targets.clone()
            class_names = [str(d) for d in range(10)]
        elif name == "letters":
            train_raw = datasets.EMNIST(root=DATA_DIR, split="letters", train=True, download=True)
            test_raw = datasets.EMNIST(root=DATA_DIR, split="letters", train=False, download=True)
            # Fix 1: EMNIST stores each image transposed (rows and columns swapped). Without this fix
            # the letters appear rotated and mirrored, yet accuracy stays plausible, so the bug is easy to miss.
            X_full = train_raw.data.transpose(1, 2).contiguous()
            X_test = test_raw.data.transpose(1, 2).contiguous()
            # Fix 2: labels are 1..26 (0 is an unused class). The loss function expects 0..25.
            y_full = train_raw.targets - 1
            y_test = test_raw.targets - 1
            class_names = list(string.ascii_lowercase)   # each class = lower + upper case of one letter
        else:
            raise ValueError(f"Unknown dataset: {name}")

        X_full, X_test = _to_tensor(X_full), _to_tensor(X_test)

        # The replication's split procedure: one permutation with seed 42, the first 10% become the
        # validation set. For MNIST this reproduces the replication's 54,000 / 6,000 split exactly.
        g = torch.Generator().manual_seed(SPLIT_SEED)
        perm = torch.randperm(len(X_full), generator=g)
        n_val = int(round(VAL_FRACTION * len(X_full)))
        val_idx, train_idx = perm[:n_val], perm[n_val:]
        if QUICK:
            train_idx = train_idx[:QUICK_TRAIN_LIMIT]

        return {
            "name": name,
            "X_train": X_full[train_idx], "y_train": y_full[train_idx],
            "X_val": X_full[val_idx], "y_val": y_full[val_idx],
            "X_test": X_test, "y_test": y_test,
            "class_names": class_names, "num_classes": len(class_names),
        }

    data = {_n: load_dataset(_n) for _n in DATASETS}

    # The printout gives the numbers of Table II
    for _n, _d in data.items():
        _counts = torch.bincount(_d["y_test"], minlength=_d["num_classes"])
        print(f"{_n:8s} classes={_d['num_classes']:2d} | train {tuple(_d['X_train'].shape)} "
              f"val {tuple(_d['X_val'].shape)} test {tuple(_d['X_test'].shape)} | "
              f"labels {_d['y_train'].min().item()}..{_d['y_train'].max().item()} | "
              f"test per class min/max = {_counts.min().item()}/{_counts.max().item()}")
    # MNIST's test set is unbalanced (892..1135 per digit); EMNIST Letters' is balanced (800 per letter).
    return (data,)


@app.cell
def _(data, plt):
    # Not in the article: a visual check that the two EMNIST fixes worked. The letters should be upright
    # and readable, and each title should match its image. If not, every later number is meaningless.
    def _():
        _d = data["letters"]
        fig, axes = plt.subplots(2, 10, figsize=(14, 3.2))
        for i, ax in enumerate(axes.flat):
            ax.imshow(_d["X_train"][i, 0], cmap="gray")
            ax.set_title(_d["class_names"][_d["y_train"][i].item()])
            ax.axis("off")
        fig.suptitle("EMNIST Letters: first 20 training images (title = label after the fixes)")
        plt.tight_layout()
        return plt.show()

    _()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Models (article: Sec. II and Sec. III-B)

    `LightweightCNN` is the model of [1] (Sec. II), generalised in three ways: the number of output
    classes *K* (10 or 26), an optional batch normalization, and an optional global-average-pooling
    (GAP) head. With its default arguments it is exactly the replicated model. `MLP` (MLP-128) and
    `LeNet5` (the LeNet-5-style network) are the neural comparisons of [1].
    """)
    return


@app.cell
def _(nn):
    # ARTICLE ▸ Sec. II: Conv(8, 5×5) → MaxPool → Conv(16, 5×5) → MaxPool → Dense(128) → Dropout(0.5) →
    #           Dense(K). Sec. III-B: the variants (batch normalization [6], GAP head [7], MLP-128 and the
    #           LeNet-5-style network).
    #
    # One class covers the baseline and all of its CNN variants, so they share every line of code and
    # differ only in the arguments being tested.
    class LightweightCNN(nn.Module):
        def __init__(self, num_classes: int = 10, c1: int = 8, c2: int = 16, dense: int = 128,
                     dropout: float = 0.5, batchnorm: bool = False, head: str = "dense"):
            super().__init__()

            def conv_block(cin, cout):
                layers = [nn.Conv2d(cin, cout, kernel_size=5, stride=1)]   # no padding ("valid"), as in [1]
                if batchnorm:
                    # Batch normalization: normalises each channel's activations during training. The
                    # article reads its gain (Sec. IV-B) as "most likely speeding up optimization within
                    # the 10-epoch budget".
                    layers.append(nn.BatchNorm2d(cout))
                layers += [nn.ReLU(), nn.MaxPool2d(2)]
                return layers

            # Image size through the network: 28 -> conv 24 -> pool 12 -> conv 8 -> pool 4.
            # The final c2 feature maps of 4x4 are what Sec. V-D means by "the network's two valid
            # convolutions reduce 28×28 inputs to 4×4 maps".
            self.features = nn.Sequential(*conv_block(1, c1), *conv_block(c1, c2))

            if head == "dense":
                # The paper's head: Flatten (c2*16 values) -> Dense(128, ReLU) -> Dropout(0.5) -> Dense(K).
                # Wider c2 also widens the input of this Dense layer, which is why the article says
                # the width variants "change the head as well as the features" (Sec. III-B).
                head_layers = [nn.Flatten(), nn.Linear(c2 * 4 * 4, dense), nn.ReLU()]
                if dropout > 0:
                    head_layers.append(nn.Dropout(dropout))
                head_layers.append(nn.Linear(dense, num_classes))
            elif head == "gap":
                # GAP head (Sec. III-B): each of the c2 = 16 feature maps is averaged to a single number,
                # and these 16 numbers go straight to the K-class output, with no hidden layer. This is
                # the "16-feature bottleneck" of Sec. IV-B and the Limitations.
                head_layers = [nn.AdaptiveAvgPool2d(1), nn.Flatten()]
                if dropout > 0:
                    head_layers.append(nn.Dropout(dropout))
                head_layers.append(nn.Linear(c2, num_classes))
            else:
                raise ValueError(f"Unknown head: {head}")
            self.classifier = nn.Sequential(*head_layers)

        def forward(self, x):
            # Returns logits (raw scores z). The softmax is applied in the loss, and in Part G as p = softmax(z/T).
            return self.classifier(self.features(x))

    class MLP(nn.Module):
        # MLP-128: one hidden layer of 128 units on the raw 784 pixels, with no convolutions. It is the
        # model that "breaks" on Chars74K in Sec. V-A. Trained with exactly the same recipe as the CNN.
        def __init__(self, num_classes: int = 10, hidden: int = 128):
            super().__init__()
            self.net = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, hidden), nn.ReLU(),
                                     nn.Linear(hidden, num_classes))

        def forward(self, x):
            return self.net(x)

    class LeNet5(nn.Module):
        # The "LeNet-5-style network" of Sec. III-B. It keeps LeNet-5's layout (tanh, average pooling,
        # 6 and 16 maps, 120 and 84 units). padding=2 on the first convolution recreates the original
        # 32x32 input from a 28x28 image. Unlike the original, the second convolution (C3) is fully
        # connected and the output is a plain linear layer + softmax, not RBF units.
        def __init__(self, num_classes: int = 10):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Tanh(), nn.AvgPool2d(2),   # 28 -> 28 -> 14
                nn.Conv2d(6, 16, kernel_size=5), nn.Tanh(), nn.AvgPool2d(2),             # 14 -> 10 -> 5
            )
            self.classifier = nn.Sequential(
                nn.Flatten(), nn.Linear(16 * 5 * 5, 120), nn.Tanh(),
                nn.Linear(120, 84), nn.Tanh(), nn.Linear(84, num_classes),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    return LeNet5, LightweightCNN, MLP


@app.cell
def _(LightweightCNN, copy, nn, pd, torch):
    # ARTICLE ▸ Sec. II: "The architecture matches [1] exactly (37,610 parameters, 354,048
    #           multiply–accumulate operations, or MACs, per image)". Table VI: the Params and MACs
    #           columns for every model.
    #
    # Counts parameters and MACs layer by layer (a MAC is one multiply-and-add, the basic cost unit).
    #   - Convolution: one dot product over (input channels × 5 × 5) values per output value.
    #   - Dense layer: inputs × outputs.
    #   - Pooling, Flatten and batch normalization count as 0 MACs; at inference, batch normalization
    #     can be folded into the convolution before it (Sec. IV-C mentions this but does not measure it).
    # The assert at the bottom checks that the generalised class still reproduces the paper's model.
    def count_params(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def complexity_table(model: nn.Module, input_size=(1, 1, 28, 28)) -> pd.DataFrame:
        rows, hooks = [], []

        def make_hook(name):
            # A forward hook runs each time a layer runs, so it can record the layer's output size
            def hook(module, inp, out):
                params = sum(p.numel() for p in module.parameters())
                if isinstance(module, nn.Conv2d):
                    kernel_volume = (module.in_channels // module.groups) * module.kernel_size[0] * module.kernel_size[1]
                    macs = out[0].numel() * kernel_volume        # batch size 1
                elif isinstance(module, nn.Linear):
                    macs = module.in_features * module.out_features
                else:
                    macs = 0
                rows.append({"layer": f"{name} ({module.__class__.__name__})",
                             "output_shape": tuple(out.shape[1:]),
                             "params": params, "MACs": macs})
            return hook

        # Pass one blank image through a CPU copy, so the original model is untouched
        model_cpu = copy.deepcopy(model).cpu().eval()
        tracked = (nn.Conv2d, nn.Linear, nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d,
                   nn.BatchNorm2d, nn.Flatten)
        for name, module in model_cpu.named_modules():
            if isinstance(module, tracked):
                hooks.append(module.register_forward_hook(make_hook(name)))
        with torch.no_grad():
            model_cpu(torch.zeros(input_size))
        for h in hooks:
            h.remove()

        df = pd.DataFrame(rows)
        df.loc[len(df)] = {"layer": "TOTAL", "output_shape": "",
                           "params": df["params"].sum(), "MACs": df["MACs"].sum()}
        return df

    # 10 classes: the paper's numbers. 26 classes: only the output layer grows (128x26+26 instead of
    # 128x10+10 weights), so the letters baseline in Table VI has 39,674 parameters.
    for _k, (_params, _macs) in {10: (37_610, 354_048), 26: (39_674, 356_096)}.items():
        _m = LightweightCNN(num_classes=_k)
        _p, _mc = count_params(_m), int(complexity_table(_m).iloc[-1]["MACs"])
        print(f"{_k} classes: params {_p:,} (expected {_params:,}) | MACs {_mc:,} (expected {_macs:,})")
        assert _p == _params and _mc == _macs, "Generalised model no longer matches the paper"

    complexity_table(LightweightCNN(num_classes=26))
    return complexity_table, count_params


@app.cell
def _(LeNet5, LightweightCNN, MLP):
    # ARTICLE ▸ Sec. III-B Models: "Ten networks are trained on digits and on letters." These are the ten
    #           rows of Tables IV–IX. Each entry builds the model for K classes, so the same entry is
    #           trained on digits (K = 10) and on letters (K = 26).
    #
    # Grouped as in Sec. III-B:
    VARIANTS = {
        # The baseline: the model of [1], unchanged except for the output layer
        "Baseline 8/16":  lambda k: LightweightCNN(k),
        # Repeats of the paper's ablations (claims C4, C6; "More filters 16/32" = Width 16/32 below, C5)
        "No dropout":     lambda k: LightweightCNN(k, dropout=0.0),
        "Dense 64":       lambda k: LightweightCNN(k, dense=64),
        # Width sweep: is the model simply too narrow for 26 classes?
        "Width 4/8":      lambda k: LightweightCNN(k, c1=4, c2=8),
        "Width 16/32":    lambda k: LightweightCNN(k, c1=16, c2=32),
        "Width 32/64":    lambda k: LightweightCNN(k, c1=32, c2=64),
        # New variants at the baseline width, each probing one explanation for why letters are harder
        "BatchNorm 8/16": lambda k: LightweightCNN(k, batchnorm=True),
        # The GAP head has no dropout: dropping half of only 16 features would cripple it
        "GAP head 8/16":  lambda k: LightweightCNN(k, head="gap", dropout=0.0),
        # The neural comparisons of [1] (claim C3 for LeNet-5). The classical baselines come later (A.6).
        "MLP-128":        lambda k: MLP(k),
        "LeNet-5":        lambda k: LeNet5(k),
    }
    # Every "effect" in the article (Table V, Fig. 2) is a variant minus this model, same seed, same machine
    BASELINE = "Baseline 8/16"
    return BASELINE, VARIANTS


@app.cell
def _(
    BATCH_SIZE,
    DEVICE,
    DataLoader,
    EPOCHS,
    EVAL_BATCH_SIZE,
    LR,
    TensorDataset,
    nn,
    torch,
):
    # ARTICLE ▸ Sec. III-D: "All models use the recipe of [1] unchanged" (Adam, learning rate 0.001, batch
    #           size 128, 10 epochs, no augmentation).
    #
    # The replication's training loop, now taking the dataset as an argument. Same optimiser, loss,
    # batch size and epochs for every model and dataset, so differences in the article come from the
    # data or the architecture, never from the training recipe.
    def run_epoch(model, loader, criterion, optimizer=None):
        """One pass over `loader`. Trains if an optimizer is given, otherwise evaluates."""
        is_train = optimizer is not None
        model.train(is_train)
        total_loss, correct, n = 0.0, 0, 0
        with torch.set_grad_enabled(is_train):
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                loss = criterion(logits, yb)
                if is_train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item() * yb.size(0)
                correct += (logits.argmax(dim=1) == yb).sum().item()
                n += yb.size(0)
        return total_loss / n, correct / n

    def fit(model, d: dict, seed: int, epochs: int = EPOCHS, lr: float = LR) -> dict:
        # The seed sets the shuffle order of the batches. Together with the initial weights, this is the
        # "seed" whose effect Section VI compares with the effect of the machine.
        g = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(TensorDataset(d["X_train"], d["y_train"]),
                                  batch_size=BATCH_SIZE, shuffle=True, generator=g)
        val_loader = DataLoader(TensorDataset(d["X_val"], d["y_val"]), batch_size=EVAL_BATCH_SIZE)
        criterion = nn.CrossEntropyLoss()   # cross-entropy on logits (softmax built in)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
        # NOTE ▸ "train_acc" is logged during training, i.e. with dropout ON. Sec. IV-B (claim C7) points
        #        out that this makes training accuracy end below test accuracy and look like under-fitting.
        #        The article's dropout-off training accuracy is recomputed from the saved models by
        #        py_file/v2/reanalysis.py.
        history = {"epoch": [], "train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}
        for epoch in range(1, epochs + 1):
            tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer)
            va_loss, va_acc = run_epoch(model, val_loader, criterion)
            history["epoch"].append(epoch)
            history["train_acc"].append(tr_acc * 100)
            history["val_acc"].append(va_acc * 100)
            history["train_loss"].append(tr_loss)
            history["val_loss"].append(va_loss)
        return history

    @torch.no_grad()
    def predict_logits(model, X: torch.Tensor) -> torch.Tensor:
        # Scores for every image. model.eval() switches dropout off and makes batch normalization use its
        # stored statistics, which is how the model would be used in practice.
        model.eval()
        chunks = [model(X[i:i + EVAL_BATCH_SIZE].to(DEVICE)).cpu()
                  for i in range(0, len(X), EVAL_BATCH_SIZE)]
        return torch.cat(chunks)

    return fit, predict_logits


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Train every (dataset × model × seed) (article: Sec. III-D)

    2 datasets × 10 models × 5 seeds = 100 trainings on this machine; with the second machine, the
    article has 200 runs, 10 per model and dataset. Each run saves its training history, its
    **logits** (raw scores) on the validation and test sets, and its weights. Everything after this
    cell (accuracy, McNemar, calibration, Chars74K) is computed from these saved files, without
    training again. The same saved weights are what `py_file/v2/reanalysis.py` reloads to produce the
    extra statistics of the article.
    """)
    return


@app.cell
def _(
    DATASETS,
    DEVICE,
    RUNS_DIR,
    SEEDS,
    VARIANTS,
    complexity_table,
    count_params,
    data,
    fit,
    mo,
    predict_logits,
    set_seed,
    time,
    torch,
):
    # ARTICLE ▸ Sec. III-D: "Each configuration is trained with seeds 0–4 on each machine".
    #
    # The only expensive cell. A finished run is stored as one file and is loaded, not retrained, when
    # the notebook runs again, so an interrupted study resumes where it stopped.
    def run_path(ds: str, variant: str, seed: int):
        slug = variant.lower().replace(" ", "_").replace("/", "-")
        return RUNS_DIR / f"{ds}__{slug}__seed{seed}.pt"

    def train_or_load(ds: str, variant: str, seed: int) -> dict:
        path = run_path(ds, variant, seed)
        if path.exists():
            return torch.load(path, weights_only=True)

        d = data[ds]
        set_seed(seed)   # seed -> initial weights and shuffle order; the data split stays at seed 42
        model = VARIANTS[variant](d["num_classes"]).to(DEVICE)
        t0 = time.perf_counter()
        history = fit(model, d, seed)
        run = {
            "dataset": ds, "variant": variant, "seed": seed,
            "history": history,
            # Validation logits are used only to fit the temperature T (Eq. 3); test logits give every
            # reported metric
            "val_logits": predict_logits(model, d["X_val"]),
            "test_logits": predict_logits(model, d["X_test"]),
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "params": count_params(model),
            "macs": int(complexity_table(model).iloc[-1]["MACs"]),
            "train_time_s": time.perf_counter() - t0,
        }
        torch.save(run, path)
        print(f"trained {ds:8s} | {variant:15s} | seed {seed} | val acc {history['val_acc'][-1]:.2f}% "
              f"| {run['train_time_s']:.0f}s")
        return run

    jobs = [(ds, v, s) for ds in DATASETS for v in VARIANTS for s in SEEDS]
    runs = {}
    for _ds, _v, _s in mo.status.progress_bar(jobs, title="Training / loading runs"):
        runs[(_ds, _v, _s)] = train_or_load(_ds, _v, _s)
    print(f"{len(runs)} runs ready in {RUNS_DIR}")
    return (runs,)


@app.cell
def _(mo):
    mo.md(r"""
    # Part A: Accuracy and cost (article: Section IV-A to IV-C)

    ## A.1 Accuracy on digits and letters (Sec. IV-A "The Fall", Table IV, Fig. 1)
    """)
    return


@app.cell
def _(data, f1_score, mo, np, pd, runs):
    # ARTICLE ▸ Table IV (Test Accuracy on Digits and Letters, "Digits" and "Letters" columns) and Fig. 1.
    #           Sec. IV-A: the baseline "falls from [digits] to [letters]"; the CNN variants "span" a
    #           few tenths of a point on digits but several points on letters.
    #
    # One row per run (dataset, model, seed), then the mean ± standard deviation over the seeds.
    # NOTE ▸ This table averages the 5 seeds of one machine. Table IV averages 10 runs (5 seeds × 2
    #        machines). The per-run file saved from here (partA_per_run.csv) is what the article pools.
    def fmt_ms(mean, std, digits=2):
        # Format as "mean ± std"; std is undefined with a single seed (QUICK mode)
        return f"{mean:.{digits}f}" if np.isnan(std) else f"{mean:.{digits}f} ± {std:.{digits}f}"

    _rows = []
    for (_ds, _v, _s), _r in runs.items():
        _y = data[_ds]["y_test"].numpy()
        _pred = _r["test_logits"].argmax(1).numpy()        # predicted class = highest score
        _h = _r["history"]
        _rows.append({
            "dataset": _ds, "variant": _v, "seed": _s,
            "test_acc": 100 * (_pred == _y).mean(),
            # Macro-F1 weights every class equally. Not in the article's tables, but a check that
            # accuracy is not hiding a weak class (MNIST's test set is unbalanced).
            "macro_f1": 100 * f1_score(_y, _pred, average="macro"),
            # Last-epoch training accuracy minus validation accuracy. Training accuracy here has dropout
            # ON, so this gap is negative for models with dropout. Sec. IV-B (C7) quotes it as the
            # misleading "looks like under-fitting" number before correcting it.
            "gen_gap": _h["train_acc"][-1] - _h["val_acc"][-1],
            "params": _r["params"], "macs": _r["macs"],
        })
    per_run = pd.DataFrame(_rows)

    summary_a = (per_run.groupby(["dataset", "variant"], sort=False)
                 .agg(test_acc_mean=("test_acc", "mean"), test_acc_std=("test_acc", "std"),
                      macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
                      gen_gap_mean=("gen_gap", "mean"),
                      params=("params", "first"), macs=("macs", "first"))
                 .reset_index())

    # Display: one row per model, digits and letters side by side, as in Table IV
    _readable = summary_a.assign(
        acc=[fmt_ms(m, s) for m, s in zip(summary_a.test_acc_mean, summary_a.test_acc_std)],
        f1=[fmt_ms(m, s) for m, s in zip(summary_a.macro_f1_mean, summary_a.macro_f1_std)],
    )
    table_a = _readable.pivot(index="variant", columns="dataset", values=["acc", "f1", "params", "gen_gap_mean"])
    table_a = table_a.reindex(summary_a["variant"].unique())
    mo.output.append(table_a)
    return fmt_ms, per_run, summary_a


@app.cell
def _(BASELINE, mo, summary_a):
    # ARTICLE ▸ Table IV, "Drop" column (digits − letters, in points). Sec. IV-B, last sentence: "Even the
    #           best variant only reduces the digits-to-letters drop from [baseline] to [Width 32/64]
    #           points (x%)", which is the "gap_closed_vs_baseline_pp" column divided by the baseline gap.
    gap_table = (summary_a.pivot(index="variant", columns="dataset", values="test_acc_mean")
                 .assign(gap_pp=lambda t: t["mnist"] - t["letters"]))
    _base_gap = gap_table.loc[BASELINE, "gap_pp"]
    # Positive = this model loses fewer points than the baseline when moving to letters
    gap_table["gap_closed_vs_baseline_pp"] = _base_gap - gap_table["gap_pp"]
    mo.output.append(mo.md("**Digits → letters accuracy gap** (percentage points; lower `gap_pp` is better)"))
    mo.output.append(gap_table.sort_values("gap_pp").round(2))
    return (gap_table,)


@app.cell
def _(mo):
    mo.md(r"""
    ## A.2 Cost: parameters, MACs, size and CPU latency (Sec. IV-C "The Price of Each Point", Table VI)

    Latency depends on the architecture, not on the trained weights, so it is measured once per model
    on an untrained 26-class copy. As in Sec. III-E: "latency is the median of 1,000 single-thread CPU
    calls with batch size 1", i.e. a small device reading one character at a time.
    """)
    return


@app.cell
def _(
    QUICK,
    VARIANTS,
    copy,
    count_params,
    complexity_table,
    mo,
    np,
    os,
    pd,
    platform,
    summary_a,
    time,
    torch,
):
    # ARTICLE ▸ Table VI (Cost and Accuracy on Letters): Params, MACs and the latency columns. Each machine
    #           runs this cell on its own CPU: "Lat. Mac" comes from the Mac's run, "Lat. Win." from the
    #           Windows run. Sec. VI-C compares them ("the Windows CPU is about N× slower ... the models'
    #           latency ranking only partly holds"). Also claim C9 (deployability).
    #
    # NOTE ▸ Sec. IV-C warns that at batch size 1 in eager PyTorch these times "mostly measure framework
    #        overhead", so they are an upper bound: Width 32/64 has many times the MACs of the baseline but
    #        only a few times its latency.
    @torch.no_grad()
    def measure_latency(model, batch_size=1, n_warmup=50, n_runs=1000, threads=1):
        torch.set_num_threads(threads)
        m = copy.deepcopy(model).cpu().eval()
        x = torch.randn(batch_size, 1, 28, 28)
        for _ in range(n_warmup):   # the first calls are slow (memory allocation, caches), so discard them
            m(x)
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            m(x)
            times.append((time.perf_counter() - t0) * 1000)
        torch.set_num_threads(os.cpu_count())
        # Median = typical call (reported in Table VI); 95th percentile = realistic worst case
        return np.median(times), np.percentile(times, 95)

    _letters_acc = summary_a[summary_a.dataset == "letters"].set_index("variant")
    _rows = []
    for _v, _factory in VARIANTS.items():
        _m = _factory(26)
        _med, _p95 = measure_latency(_m, n_runs=100 if QUICK else 1000)
        _p = count_params(_m)
        _rows.append({"variant": _v, "params": _p,
                      "MACs": int(complexity_table(_m).iloc[-1]["MACs"]),
                      "size_KB": _p * 4 / 1024,           # 4 bytes per float32 parameter ("KB of weights", Sec. IV-C)
                      "latency_median_ms": _med, "latency_p95_ms": _p95,
                      "letters_acc_mean": _letters_acc.loc[_v, "test_acc_mean"]})
    cost_table = pd.DataFrame(_rows)
    mo.output.append(mo.md(f"CPU: `{platform.processor() or platform.machine()}`, 1 thread, batch 1"))
    mo.output.append(cost_table.round(3))
    return (cost_table,)


@app.cell
def _(plt, summary_a):
    # ARTICLE ▸ Fig. 3 ("Accuracy on letters against compute. The grey step is the Pareto frontier.").
    #           The article's figure is redrawn by py_file/v2/figures.py from the saved tables. This
    #           cell is the notebook's version, with digits and letters and both cost measures.
    #
    # How to read it: each point is a model, x is its cost (log scale), y its mean test accuracy, bars
    # are ± one standard deviation over seeds. The dashed line joins the "Pareto" models: those that no
    # cheaper model beats. Sec. IV-C reads it as "batch normalization improves accuracy at the same
    # compute. Beyond it, each point gets more expensive".
    def _pareto(df, x, y):
        # Walk the models from cheapest to most expensive; keep each one that beats every cheaper one
        front, best = [], -float("inf")
        for _, row in df.sort_values(x).iterrows():
            if row[y] > best:
                front.append(row)
                best = row[y]
        return front

    def _():
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, xcol, xlabel in [(axes[0], "params", "Parameters"), (axes[1], "macs", "MACs per image")]:
            for ds, marker in [("mnist", "o"), ("letters", "s")]:
                sub = summary_a[summary_a.dataset == ds]
                ax.errorbar(sub[xcol], sub["test_acc_mean"], yerr=sub["test_acc_std"].fillna(0),
                            fmt=marker, capsize=3, label=ds)
                front = _pareto(sub, xcol, "test_acc_mean")
                ax.plot([r[xcol] for r in front], [r["test_acc_mean"] for r in front], "--", alpha=0.5)
                for _, row in sub.iterrows():
                    ax.annotate(row["variant"], (row[xcol], row["test_acc_mean"]),
                                fontsize=7, xytext=(4, 3), textcoords="offset points")
            ax.set(xscale="log", xlabel=f"{xlabel} (log scale)", ylabel="Test accuracy (%)",
                   title=f"Accuracy vs. {xlabel.lower()}")
            ax.grid(alpha=0.3)
            ax.legend()
        plt.tight_layout()
        return plt.show()

    _()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## A.3 Is a difference real? McNemar's test (Sec. III-D Eq. 2, Sec. VI-B, Table XI)

    Each model is compared with the baseline trained with the **same seed**, on the same test images.
    Only the images where they disagree count: *b* = images only the variant gets right, *c* = images
    only the baseline gets right. The table reports in how many of the five seeds p < 0.05.
    """)
    return


@app.cell
def _(BASELINE, DATASETS, SEEDS, VARIANTS, data, math, mo, np, pd, runs):
    # ARTICLE ▸ Eq. (2): χ² = (|b − c| − 1)² / (b + c), "with an exact binomial test when b + c < 25".
    #           Sec. III-D: "the per-machine rule of significance in all five seeds". Table XI ("Verdicts
    #           that Depend on the Machine"): the "seeds" in its Mac and Windows columns are the
    #           "seeds_significant" column below, from each machine's own run of this cell.
    #
    # NOTE ▸ McNemar gives a yes/no verdict per seed. Section VI-B shows these verdicts are fragile
    #        (several flip between machines) while effect sizes with confidence intervals are not. The
    #        intervals of Table V and Fig. 2 are therefore computed by py_file/v2/data.py from both
    #        machines, not by this cell. "acc_diff_pp" below is the per-machine Δ of Table XI.
    def mcnemar(correct_x: np.ndarray, correct_y: np.ndarray):
        b = int(np.sum(correct_x & ~correct_y))   # X right, Y wrong
        c = int(np.sum(~correct_x & correct_y))   # X wrong, Y right
        n = b + c
        if n == 0:
            return b, c, 1.0
        if n < 25:
            # Few disagreements: exact binomial test (if the models were equal, each disagreement would be
            # a fair coin flip between b and c)
            k = min(b, c)
            p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
        else:
            # Eq. (2), with continuity correction; p-value of a chi-square with 1 degree of freedom
            chi2 = (abs(b - c) - 1) ** 2 / n
            p = math.erfc(math.sqrt(chi2 / 2))
        return b, c, p

    _rows = []
    for _ds in DATASETS:
        _y = data[_ds]["y_test"].numpy()
        for _v in VARIANTS:
            if _v == BASELINE:
                continue
            for _s in SEEDS:
                _cx = runs[(_ds, _v, _s)]["test_logits"].argmax(1).numpy() == _y
                _cy = runs[(_ds, BASELINE, _s)]["test_logits"].argmax(1).numpy() == _y
                _b, _c, _p = mcnemar(_cx, _cy)
                _rows.append({"dataset": _ds, "variant": _v, "seed": _s, "b": _b, "c": _c, "p": _p,
                              "acc_diff_pp": 100 * (_cx.mean() - _cy.mean())})
    mcnemar_per_seed = pd.DataFrame(_rows)

    # One row per (dataset, model): mean paired difference Δ and "k/5" seeds with p < 0.05
    mcnemar_table = (mcnemar_per_seed.groupby(["dataset", "variant"], sort=False)
                     .agg(acc_diff_pp=("acc_diff_pp", "mean"), b_mean=("b", "mean"), c_mean=("c", "mean"),
                          p_median=("p", "median"),
                          seeds_significant=("p", lambda p: f"{(p < 0.05).sum()}/{len(p)}"))
                     .reset_index())
    mo.output.append(mcnemar_table.round(4))
    return mcnemar, mcnemar_table


@app.cell
def _(mo):
    mo.md(r"""
    ## A.4 Training curves: are 10 epochs enough? (Sec. IV-B, claim C7)
    """)
    return


@app.cell
def _(BASELINE, DATASETS, SEEDS, VARIANTS, np, plt, runs):
    # ARTICLE ▸ Sec. IV-B, claim C7 ("Little overfitting; 10 epochs are enough"). Not a figure in the
    #           article; it shows the curves behind the sentence "10 epochs are roughly enough".
    #
    # Left: the baseline's training (dashed) and validation (solid) accuracy per epoch, on both datasets.
    # Right: validation accuracy of every model on letters. A curve still clearly rising at epoch 10 would
    # mean part of the letters gap is short training rather than model capacity.
    # NOTE ▸ The dashed training curves have dropout ON, which is why they sit below validation (see the
    #        note in the training cell). The article's dropout-off comparison is in reanalysis.py.
    def _():
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
        for ds in DATASETS:
            hs = [runs[(ds, BASELINE, s)]["history"] for s in SEEDS]
            ep = hs[0]["epoch"]
            axes[0].plot(ep, np.mean([h["train_acc"] for h in hs], 0), "o--", label=f"{ds} train")
            axes[0].plot(ep, np.mean([h["val_acc"] for h in hs], 0), "o-", label=f"{ds} val")
        axes[0].set(title=f"{BASELINE}: accuracy vs. epoch (mean over seeds)",
                    xlabel="Epoch", ylabel="Accuracy (%)")
        axes[0].grid(alpha=0.3)
        axes[0].legend()
        for v in VARIANTS:
            hs = [runs[("letters", v, s)]["history"] for s in SEEDS]
            axes[1].plot(hs[0]["epoch"], np.mean([h["val_acc"] for h in hs], 0), "o-", ms=3, label=v)
        axes[1].set(title="Letters: validation accuracy per variant", xlabel="Epoch", ylabel="Accuracy (%)")
        axes[1].grid(alpha=0.3)
        axes[1].legend(fontsize=7)
        plt.tight_layout()
        return plt.show()

    _()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## A.5 Error analysis on EMNIST (background; not reported in the article)

    The article's error analysis is on the second letters dataset (Sec. V-D, next to Fig. 6). These
    cells show the same view on EMNIST and on digits, for comparison. Expect confusions such as i↔l,
    g↔q, u↔v and h↔n on letters.
    """)
    return


@app.cell
def _(DATASETS, mo):
    # A dropdown to choose which dataset the two error-analysis cells below show
    ds_picker = mo.ui.dropdown(options=DATASETS, value="letters", label="Dataset")
    ds_picker
    return (ds_picker,)


@app.cell
def _(
    BASELINE,
    ConfusionMatrixDisplay,
    SEEDS,
    confusion_matrix,
    data,
    ds_picker,
    mo,
    pd,
    plt,
    precision_recall_fscore_support,
    runs,
):
    # Background for Sec. IV-A: accuracy says how often the baseline fails; the confusion matrix and
    # per-class precision/recall say where. One seed of the baseline only.
    _ds = ds_picker.value
    _d = data[_ds]
    _names = _d["class_names"]
    _k = _d["num_classes"]
    _y = _d["y_test"].numpy()
    _pred = runs[(_ds, BASELINE, SEEDS[0])]["test_logits"].argmax(1).numpy()
    _cm = confusion_matrix(_y, _pred, labels=range(_k))

    # Rows = true class, columns = predicted class. Counts are printed only for 10 classes.
    _fig, _ax = plt.subplots(figsize=(7, 6) if _k <= 10 else (12, 11))
    ConfusionMatrixDisplay(_cm, display_labels=_names).plot(
        ax=_ax, cmap="Blues", colorbar=False, include_values=_k <= 10)
    _ax.set_title(f"{_ds}: confusion matrix, {BASELINE}, seed {SEEDS[0]}")
    plt.tight_layout()
    plt.show()

    # Per-class precision, recall and F1 (as in the replication report's Table III), weakest first
    _p, _r, _f1, _sup = precision_recall_fscore_support(_y, _pred, labels=range(_k), zero_division=0)
    per_class = pd.DataFrame({"class": _names, "precision": _p * 100, "recall": _r * 100,
                              "f1": _f1 * 100, "support": _sup}).sort_values("f1")
    mo.output.append(mo.md("**Per-class metrics (weakest first)**"))
    mo.output.append(per_class.round(2))

    # The ten most frequent mistakes, as "true -> predicted"
    _pairs = [(_cm[i, j], i, j) for i in range(_k) for j in range(_k) if i != j and _cm[i, j] > 0]
    top_pairs = pd.DataFrame(
        [{"true": _names[i], "predicted": _names[j], "count": n,
          "% of true class": 100 * n / _cm[i].sum()} for n, i, j in sorted(_pairs, reverse=True)[:10]])
    mo.output.append(mo.md("**Top-10 confusions**"))
    mo.output.append(top_pairs.round(2))
    return (top_pairs,)


@app.cell
def _(BASELINE, SEEDS, data, ds_picker, plt, runs):
    # Background: the misclassified images themselves, for the five most frequent confusions. Looking at
    # them separates model mistakes from handwriting that is ambiguous even to a person.
    def _():
        ds = ds_picker.value
        d = data[ds]
        y = d["y_test"]
        pred = runs[(ds, BASELINE, SEEDS[0])]["test_logits"].argmax(1)
        wrong = (pred != y)
        pairs = {}
        for t, p in zip(y[wrong].tolist(), pred[wrong].tolist()):
            pairs[(t, p)] = pairs.get((t, p), 0) + 1
        top = sorted(pairs, key=pairs.get, reverse=True)[:5]
        n_cols = 8
        fig, axes = plt.subplots(len(top), n_cols, figsize=(n_cols * 1.3, len(top) * 1.5))
        for row, (t, p) in enumerate(top):
            idx = ((y == t) & (pred == p)).nonzero().flatten()[:n_cols]
            for col in range(n_cols):
                ax = axes[row, col] if len(top) > 1 else axes[col]
                ax.axis("off")
                if col < len(idx):
                    ax.imshow(d["X_test"][idx[col], 0], cmap="gray")
                if col == 0:
                    ax.set_title(f"{d['class_names'][t]}→{d['class_names'][p]}", fontsize=9, loc="left")
        fig.suptitle(f"{ds}: misclassified examples for the top-5 confusions (true→predicted)")
        plt.tight_layout()
        return plt.show()

    _()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## A.6 Classical baselines (Sec. III-B, Table IV bottom rows, claim C2)

    "Logistic regression, k-nearest neighbours (k = 3) and an RBF support vector machine on raw pixels
    are the classical baselines; the SVM uses a 20,000-image subsample." They are trained with
    scikit-learn on the same split, with each image flattened to 784 numbers. They have no random seed,
    so each machine gives one number per model; Table IV shows the mean of the two machines.
    The first run of this cell takes several minutes.
    """)
    return


@app.cell
def _(
    DATASETS,
    KNeighborsClassifier,
    LogisticRegression,
    RESULTS_DIR,
    SPLIT_SEED,
    SVC,
    SVM_TRAIN_LIMIT,
    data,
    f1_score,
    json,
    np,
    time,
):
    # ARTICLE ▸ Table IV, the three classical rows; Sec. IV-A, claim C2: the CNN's lead over classical
    #           methods "grows" from digits (over the SVM) to letters (over k-NN). Section VI, first
    #           paragraph: "k-NN and the SVM give bit-identical accuracies [on both machines]; logistic
    #           regression differs by at most ... points because its solver stops at a different
    #           iteration". The "n_iter" field saved below is where that is visible.
    #
    # NOTE ▸ These models are not evaluated on Chars74K (their fitted models are not saved), which is
    #        listed in the article's Limitations.
    _cache = RESULTS_DIR / "sklearn_baselines.json"
    sklearn_results = json.loads(_cache.read_text()) if _cache.exists() else {}

    def _flat(X):
        return X.reshape(len(X), -1).numpy()   # (N, 1, 28, 28) -> (N, 784), same /255 scaling as the CNN

    _models = {
        # 784 features x 26 classes needs more than the default 100 iterations to converge. A model that
        # stops early would make the CNN look better than it is, so up to 1000 are allowed.
        "Logistic regression": lambda: LogisticRegression(max_iter=1000),
        "KNN (k=3)": lambda: KNeighborsClassifier(n_neighbors=3, n_jobs=-1),
        "SVM (RBF)": lambda: SVC(kernel="rbf", gamma="scale"),
    }
    for _ds in DATASETS:
        _d = data[_ds]
        _Xtr, _ytr = _flat(_d["X_train"]), _d["y_train"].numpy()
        _Xte, _yte = _flat(_d["X_test"]), _d["y_test"].numpy()
        for _name, _make in _models.items():
            _key = f"{_ds}|{_name}"
            if _key in sklearn_results:
                continue
            _X, _yy = _Xtr, _ytr
            if _name.startswith("SVM") and len(_Xtr) > SVM_TRAIN_LIMIT:
                # The 20,000-image subsample, drawn with seed 42 so both machines use the same images
                _idx = np.random.default_rng(SPLIT_SEED).choice(len(_Xtr), SVM_TRAIN_LIMIT, replace=False)
                _X, _yy = _Xtr[_idx], _ytr[_idx]
            _t0 = time.perf_counter()
            _clf = _make().fit(_X, _yy)
            _pred = _clf.predict(_Xte)
            sklearn_results[_key] = {
                "dataset": _ds, "variant": _name,
                "test_acc": 100 * float((_pred == _yte).mean()),
                "macro_f1": 100 * float(f1_score(_yte, _pred, average="macro")),
                "train_samples": int(len(_X)), "time_s": time.perf_counter() - _t0,
                # Optimiser iterations used. For logistic regression, a value equal to max_iter means it
                # stopped before converging.
                "n_iter": int(np.max(_clf.n_iter_)) if hasattr(_clf, "n_iter_") else None,
            }
            _cache.write_text(json.dumps(sklearn_results, indent=2))   # saved after each model
            print(f"{_ds:8s} | {_name:20s} | acc {sklearn_results[_key]['test_acc']:.2f}% "
                  f"| {sklearn_results[_key]['time_s']:.0f}s")
    return (sklearn_results,)


@app.cell
def _(BASELINE, fmt_ms, mo, pd, sklearn_results, summary_a):
    # ARTICLE ▸ Table IV: the baseline, MLP-128, the LeNet-5-style network and the classical methods side
    #           by side on digits and letters. (The name "table7" refers to Table 7 of the base paper [1],
    #           whose values were simulated; these are measured.)
    _neural = summary_a[summary_a.variant.isin([BASELINE, "MLP-128", "LeNet-5"])].copy()
    _neural["acc"] = [fmt_ms(m, s) for m, s in zip(_neural.test_acc_mean, _neural.test_acc_std)]
    _neural["macro_f1"] = [fmt_ms(m, s) for m, s in zip(_neural.macro_f1_mean, _neural.macro_f1_std)]
    _classic = pd.DataFrame(list(sklearn_results.values()))
    _classic["acc"] = _classic["test_acc"].map(lambda a: f"{a:.2f}")   # no seeds, so no ± spread
    _classic["macro_f1"] = _classic["macro_f1"].map(lambda a: f"{a:.2f}")
    table7 = (pd.concat([_neural[["dataset", "variant", "acc", "macro_f1"]],
                         _classic[["dataset", "variant", "acc", "macro_f1"]]])
              .pivot(index="variant", columns="dataset", values=["acc", "macro_f1"]))
    mo.output.append(mo.md("**Table 7 (measured): baselines on digits and letters**"))
    mo.output.append(table7)
    return (table7,)


@app.cell
def _(mo):
    mo.md(r"""
    # Part G: Reliability (article: Sec. III-E and Sec. IV-D "Can the Confidence be Trusted?")

    *Confidence* is the highest softmax probability of a prediction. A model is *calibrated* when,
    of all the predictions it makes with about 80% confidence, about 80% are correct. Everything here
    is computed from the saved logits. The temperature *T* is always fitted on the **validation** set
    and evaluated on the **test** set. This part checks claim C9, which [1] does not examine.
    """)
    return


@app.cell
def _(N_BINS, nn, np, torch):
    # ARTICLE ▸ Sec. III-E Reliability Metrics: Eq. (3) temperature scaling, the negative log-likelihood
    #           (NLL), the Brier score, Eq. (4) the expected calibration error (ECE) over 15 bins, and
    #           selective prediction: AURC and "the largest coverage that keeps 99% accuracy".
    #
    # The definitions are written once here and reused for EMNIST (this part) and Chars74K (Part E), so
    # the two datasets are always measured the same way.
    def probs_from_logits(logits: torch.Tensor, T: float = 1.0) -> np.ndarray:
        # Eq. (3): p = softmax(z / T). T > 1 makes the model less confident, T < 1 more confident.
        # Dividing every score by the same T never changes which class is highest, so accuracy is unchanged.
        return torch.softmax(logits.float() / T, dim=1).numpy()

    def fit_temperature(val_logits: torch.Tensor, val_labels: torch.Tensor) -> float:
        # Find the one number T that minimises the NLL on the validation set. The search is over log T,
        # which keeps T positive. With a single parameter there is almost nothing to overfit.
        log_t = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200, line_search_fn="strong_wolfe")
        z, y = val_logits.float(), val_labels.long()

        def closure():
            opt.zero_grad()
            loss = nn.functional.cross_entropy(z / log_t.exp(), y)
            loss.backward()
            return loss

        opt.step(closure)
        return float(log_t.exp().item())

    def calibration_metrics(probs: np.ndarray, labels: np.ndarray, n_bins: int = N_BINS) -> dict:
        n = len(labels)
        conf = probs.max(1)
        correct = probs.argmax(1) == labels
        # NLL: mean of -log(probability given to the true class). Confident mistakes cost the most.
        nll = float(-np.log(np.clip(probs[np.arange(n), labels], 1e-12, None)).mean())
        # Brier score: squared distance between the probability vector and the true one-hot label (0..2)
        onehot = np.zeros_like(probs)
        onehot[np.arange(n), labels] = 1.0
        brier = float(((probs - onehot) ** 2).sum(1).mean())
        # Eq. (4): split predictions into 15 equal-width confidence bins B_m; in each, compare accuracy
        # with mean confidence; ECE = the average gap weighted by |B_m|/n. MCE = the worst single bin.
        edges = np.linspace(0, 1, n_bins + 1)
        bin_acc, bin_conf, bin_count = np.full(n_bins, np.nan), np.full(n_bins, np.nan), np.zeros(n_bins, int)
        for b in range(n_bins):
            mask = (conf > edges[b]) & (conf <= edges[b + 1]) if b > 0 else (conf <= edges[1])
            bin_count[b] = mask.sum()
            if bin_count[b]:
                bin_acc[b], bin_conf[b] = correct[mask].mean(), conf[mask].mean()
        gaps = np.abs(bin_acc - bin_conf)
        filled = bin_count > 0
        return {"nll": nll, "brier": brier,
                "ece": float((bin_count[filled] / n * gaps[filled]).sum()),
                "mce": float(gaps[filled].max()),
                "bin_acc": bin_acc, "bin_conf": bin_conf, "bin_count": bin_count, "edges": edges}

    def risk_coverage(probs: np.ndarray, labels: np.ndarray) -> dict:
        # Selective prediction [10]: the model answers only its most confident inputs and hands the rest
        # to a person. Coverage = fraction answered; risk = error rate among the answered ones.
        conf = probs.max(1)
        correct = probs.argmax(1) == labels
        order = np.argsort(-conf, kind="stable")      # most confident first
        k = np.arange(1, len(labels) + 1)
        risk = np.cumsum(~correct[order]) / k          # risk when answering the k most confident images
        coverage = k / len(labels)

        def acc_at(c):           # accuracy when answering the most confident fraction c
            return 100 * (1 - risk[max(int(np.ceil(c * len(labels))) - 1, 0)])

        def coverage_at(target):  # largest coverage whose accuracy is still >= target
            ok = np.nonzero(risk <= 1 - target)[0]
            return 100 * coverage[ok[-1]] if len(ok) else 0.0

        return {"coverage": coverage, "risk": risk,
                "aurc": float(risk.mean()),   # AURC: area under the risk-coverage curve, lower is better
                "acc@90cov": acc_at(0.90), "acc@95cov": acc_at(0.95), "acc@99cov": acc_at(0.99),
                # "Cov." of Table VII. It picks the cut-off using the test labels themselves, so it can
                # only compare models; the deployable version is discussed in Sec. V-E.
                "cov@99acc": coverage_at(0.99)}

    return calibration_metrics, fit_temperature, probs_from_logits, risk_coverage


@app.cell
def _(
    calibration_metrics,
    data,
    fit_temperature,
    fmt_ms,
    mo,
    pd,
    probs_from_logits,
    risk_coverage,
    runs,
):
    # ARTICLE ▸ Table VII (Reliability on Letters): the T, "ECE raw" and "ECE + T" columns, and the
    #           sentences of Sec. IV-D: "every CNN except the GAP head has an ECE of at most ...% on
    #           letters", and temperature scaling "helps the over-confident models (Width 32/64) and
    #           slightly worsens the already well-calibrated baseline and BatchNorm".
    #
    # For every run: fit T on validation, then measure calibration on test before (T = 1) and after.
    # NOTE ▸ As for accuracy, the article averages the 10 runs of both machines (partG_per_run.csv).
    _rows = []
    for (_ds, _v, _s), _r in runs.items():
        _d = data[_ds]
        _yte = _d["y_test"].numpy()
        _T = fit_temperature(_r["val_logits"], _d["y_val"])   # validation only: the test set is never used to fit
        _before = calibration_metrics(probs_from_logits(_r["test_logits"]), _yte)
        _after = calibration_metrics(probs_from_logits(_r["test_logits"], _T), _yte)
        # Selective prediction uses the raw confidences; T does not change which inputs are most confident
        _rc = risk_coverage(probs_from_logits(_r["test_logits"]), _yte)
        _rows.append({"dataset": _ds, "variant": _v, "seed": _s, "T": _T,
                      **{f"{m}_before": _before[m] for m in ("nll", "brier", "ece", "mce")},
                      **{f"{m}_after": _after[m] for m in ("nll", "brier", "ece", "mce")},
                      **{m: _rc[m] for m in ("aurc", "acc@90cov", "acc@95cov", "acc@99cov", "cov@99acc")}})
    calib_per_run = pd.DataFrame(_rows)

    _metric_cols = [c for c in calib_per_run.columns if c not in ("dataset", "variant", "seed")]
    _g = calib_per_run.groupby(["dataset", "variant"], sort=False)[_metric_cols]
    summary_g = _g.mean().add_suffix("_mean").join(_g.std().add_suffix("_std")).reset_index()

    # Display: T, and ECE (in %), NLL and Brier before -> after temperature scaling
    _show = summary_g[["dataset", "variant"]].copy()
    _show["T"] = [fmt_ms(m, s) for m, s in zip(summary_g["T_mean"], summary_g["T_std"])]
    for _m, _scale in [("ece", 100), ("nll", 1), ("brier", 1)]:
        for _when in ("before", "after"):
            _show[f"{_m}_{_when}"] = [fmt_ms(_scale * m, _scale * s, 3) for m, s in
                                      zip(summary_g[f"{_m}_{_when}_mean"], summary_g[f"{_m}_{_when}_std"])]
    mo.output.append(mo.md("**Calibration before → after temperature scaling** (ECE in %, lower is better; "
                           "T > 1 means the model was overconfident)"))
    mo.output.append(_show)
    return calib_per_run, summary_g


@app.cell
def _(BASELINE, DATASETS, SEEDS, calibration_metrics, data, fit_temperature, plt, probs_from_logits, runs):
    # ARTICLE ▸ Sec. IV-D, "The confidence is honest". Not a figure in the article; this is the picture
    #           behind the ECE numbers of Table VII, for one seed of the baseline.
    #
    # How to read a reliability diagram: each bar is one of the 15 confidence bins; its height is the
    # accuracy in that bin. A perfectly calibrated model has bars on the dashed diagonal. The red part
    # is the gap between confidence and accuracy: red above a bar means over-confident.
    def _():
        fig, axes = plt.subplots(len(DATASETS), 2, figsize=(11, 5 * len(DATASETS)), squeeze=False)
        for row, ds in enumerate(DATASETS):
            r, d = runs[(ds, BASELINE, SEEDS[0])], data[ds]
            T = fit_temperature(r["val_logits"], d["y_val"])
            for col, (label, temp) in enumerate([("before", 1.0), (f"after (T={T:.2f})", T)]):
                m = calibration_metrics(probs_from_logits(r["test_logits"], temp), d["y_test"].numpy())
                ax = axes[row, col]
                width = m["edges"][1] - m["edges"][0]
                centers = m["edges"][:-1] + width / 2
                ax.bar(centers, m["bin_acc"], width=width, edgecolor="black", alpha=0.8, label="accuracy")
                ax.bar(centers, m["bin_conf"] - m["bin_acc"], bottom=m["bin_acc"], width=width,
                       color="red", alpha=0.3, edgecolor="red", label="gap to confidence")
                ax.plot([0, 1], [0, 1], "k--", label="perfect calibration")
                ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Confidence", ylabel="Accuracy",
                       title=f"{ds}, {label}: ECE = {100 * m['ece']:.2f}%")
                ax.legend(loc="upper left", fontsize=8)
        fig.suptitle(f"Reliability diagrams, {BASELINE}, seed {SEEDS[0]} ({len(m['edges']) - 1} bins)")
        plt.tight_layout()
        return plt.show()

    _()
    return


@app.cell
def _(BASELINE, DATASETS, SEEDS, data, fmt_ms, mo, plt, probs_from_logits, risk_coverage, runs, summary_g):
    # ARTICLE ▸ Table VII, "AURC (%)" and "Cov. (%)" columns. Sec. IV-D: "At 99% accuracy the baseline can
    #           answer [x]% of digits but only [y]% of letters, so about [z]% of letters must go to a
    #           human." Fig. 7b shows the same coverage for digits, EMNIST and Chars74K.
    #
    # The curve: moving right, the model answers more of the test set (coverage) and makes more errors
    # among what it answers (risk). A lower curve is better; AURC is the area under it.
    def _():
        fig, ax = plt.subplots(figsize=(8, 5))
        for ds in DATASETS:
            r = runs[(ds, BASELINE, SEEDS[0])]
            rc = risk_coverage(probs_from_logits(r["test_logits"]), data[ds]["y_test"].numpy())
            ax.plot(rc["coverage"] * 100, rc["risk"] * 100, label=f"{ds} (AURC = {100 * rc['aurc']:.2f}%)")
        ax.set(xlabel="Coverage (% of test images answered)", ylabel="Risk (% errors among answered)",
               title=f"Risk–coverage curves, {BASELINE}, seed {SEEDS[0]}")
        ax.grid(alpha=0.3)
        ax.legend()
        plt.tight_layout()
        return plt.show()

    _()

    # "cov@99acc" is the "Cov." column of Table VII
    selective_table = summary_g[["dataset", "variant"]].copy()
    for _m in ("acc@90cov", "acc@95cov", "acc@99cov", "cov@99acc"):
        selective_table[_m] = [fmt_ms(m, s) for m, s in zip(summary_g[f"{_m}_mean"], summary_g[f"{_m}_std"])]
    selective_table["AURC_%"] = [fmt_ms(100 * m, 100 * s, 3) for m, s in
                                 zip(summary_g["aurc_mean"], summary_g["aurc_std"])]
    mo.output.append(mo.md("**Selective prediction**: accuracy at 90/95/99% coverage, and coverage that keeps "
                           "≥ 99% accuracy"))
    mo.output.append(selective_table)
    return (selective_table,)


@app.cell
def _(DATASETS, plt, summary_a, summary_g):
    # ARTICLE ▸ Sec. IV-D: which models are over-confident before temperature scaling (e.g. Width 32/64)
    #           and which are already well calibrated. Not a figure in the article.
    #
    # Each point is a model: x = parameters (log scale), y = ECE before scaling, colour = accuracy.
    # Points high up are poorly calibrated, so this shows whether bigger or more accurate models pay
    # for it in calibration.
    def _():
        merged = summary_g.merge(summary_a[["dataset", "variant", "params", "test_acc_mean"]],
                                 on=["dataset", "variant"])
        fig, axes = plt.subplots(1, len(DATASETS), figsize=(7 * len(DATASETS), 4.5), squeeze=False)
        for ax, ds in zip(axes[0], DATASETS):
            sub = merged[merged.dataset == ds]
            sc = ax.scatter(sub["params"], 100 * sub["ece_before_mean"], c=sub["test_acc_mean"], cmap="viridis")
            for _, row in sub.iterrows():
                ax.annotate(row["variant"], (row["params"], 100 * row["ece_before_mean"]),
                            fontsize=7, xytext=(4, 3), textcoords="offset points")
            ax.set(xscale="log", xlabel="Parameters (log scale)", ylabel="ECE before scaling (%)",
                   title=f"{ds}: calibration vs. model size")
            ax.grid(alpha=0.3)
            fig.colorbar(sc, ax=ax, label="Test accuracy (%)")
        plt.tight_layout()
        return plt.show()

    _()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## G3: Does the model notice inputs from the wrong dataset? (Sec. IV-D, Table VIII)

    "OOD detection [11] shows the letters model digits and the digits model letters." A useful
    confidence score should be lower on these out-of-distribution (OOD) images.
    **AUROC** = the probability that a random image of the model's own kind gets higher confidence
    than a random foreign one (0.5 = chance, 1 = perfect).
    **FPR@95** = the share of foreign images still accepted when the threshold accepts 95% of the
    model's own images.
    """)
    return


@app.cell
def _(
    BASELINE,
    DATASETS,
    DEVICE,
    SEEDS,
    VARIANTS,
    data,
    mo,
    np,
    pd,
    plt,
    predict_logits,
    probs_from_logits,
    roc_auc_score,
    runs,
):
    # ARTICLE ▸ Table VIII (Out-of-Distribution Detection with Maximum Softmax Probability): one row per
    #           model and machine, from each machine's run of this cell. Sec. IV-D: the letters model
    #           "accepts about [x]% of [digits] as letters at 95% true-positive rate, since many digits look
    #           like letters (0/O, 1/l, 5/S)". Sec. VI-C repeats the figure for each machine.
    #
    # Uses the baseline trained with the first seed. The second table printed below shows what the model
    # "thinks" each foreign class is (e.g. the digit 0 read as the letter o): the source of the 0/O,
    # 1/l, 5/S examples.
    if len(DATASETS) == 2:
        _rows, _assign = [], []
        _fig, _axes = plt.subplots(1, 2, figsize=(13, 4))
        for _ax, (_in, _out) in zip(_axes, [("mnist", "letters"), ("letters", "mnist")]):
            _d_in, _d_out = data[_in], data[_out]
            # Rebuild the trained baseline from its saved weights (no retraining)
            _m = VARIANTS[BASELINE](_d_in["num_classes"])
            _m.load_state_dict(runs[(_in, BASELINE, SEEDS[0])]["state_dict"])
            _m = _m.to(DEVICE)
            _conf_in = probs_from_logits(predict_logits(_m, _d_in["X_test"])).max(1)
            _probs_out = probs_from_logits(predict_logits(_m, _d_out["X_test"]))
            _conf_out = _probs_out.max(1)
            _auroc = roc_auc_score(np.r_[np.ones(len(_conf_in)), np.zeros(len(_conf_out))],
                                   np.r_[_conf_in, _conf_out])
            _thr = np.quantile(_conf_in, 0.05)            # threshold that accepts 95% of the model's own images
            _fpr = float((_conf_out >= _thr).mean())      # share of foreign images accepted at that threshold
            _rows.append({"model trained on": _in, "OOD inputs": _out, "AUROC": _auroc,
                          "FPR@95TPR_%": 100 * _fpr,
                          "mean conf in-dist": _conf_in.mean(), "mean conf OOD": _conf_out.mean()})
            # Histogram: how confident the model is on its own data (blue) and on foreign data (orange)
            _ax.hist(_conf_in, bins=50, alpha=0.6, density=True, label=f"in-distribution ({_in})")
            _ax.hist(_conf_out, bins=50, alpha=0.6, density=True, label=f"OOD ({_out})")
            _ax.axvline(_thr, color="k", ls="--", label="95% TPR threshold")
            _ax.set(title=f"{_in} model: AUROC = {_auroc:.3f}", xlabel="Confidence (max softmax)", ylabel="Density")
            _ax.legend(fontsize=8)
            # For each foreign class, the class the model most often assigns to it
            _pred_out = _probs_out.argmax(1)
            for _c, _cname in enumerate(_d_out["class_names"]):
                _mask = _d_out["y_test"].numpy() == _c
                _vals, _cnt = np.unique(_pred_out[_mask], return_counts=True)
                _assign.append({"model": _in, "OOD class": _cname,
                                "most predicted as": _d_in["class_names"][_vals[_cnt.argmax()]],
                                "% of that class": 100 * _cnt.max() / _mask.sum()})
        plt.tight_layout()
        plt.show()
        ood_table = pd.DataFrame(_rows)
        mo.output.append(ood_table.round(3))
        mo.output.append(mo.md("**How OOD classes are labelled** (digits seen by the letters model and vice versa)"))
        mo.output.append(pd.DataFrame(_assign).round(1))
    else:
        ood_table = pd.DataFrame()
    return (ood_table,)


@app.cell
def _(mo):
    mo.md(r"""
    # Part E: A second letters dataset, Chars74K (article: Section V, Table IX, Figs. 4–7)

    "Everything above was learned on EMNIST. Were those lessons about letters, or about EMNIST?"
    EMNIST comes from the same NIST forms as MNIST. Chars74K [5] does not: 55 volunteers drew each
    upper- and lowercase letter on a tablet. That gives 2,860 letter images (digits are dropped),
    with cases merged into the same 26 classes as EMNIST Letters (Table II, "Letters (new writers)").

    **Protocol (test only), as in Sec. III-A and III-E.**
    - **No retraining.** "All 100 letter models meet the 2,860 Chars74K letters as they are": every
      EMNIST Letters run is reloaded from its saved weights. (100 = 5 seeds × 10 models × 2 machines;
      one run of this notebook gives 50.)
    - **Temperature from EMNIST.** "On Chars74K, the temperature fitted on EMNIST is reused, not
      refitted." This tests whether calibration survives the new data.
    - **EMNIST's conversion.** Inversion, cropping, centring with a 2-pixel border at 128-pixel scale,
      Gaussian blur σ = 1, bicubic resizing to 28×28, scaling to [0, 1].
    - **Stroke matching, Eq. (1).** Chars74K strokes are almost twice as thick, so they are thinned
      at the level *k* whose mean ink is closest to EMNIST's training images. No Chars74K labels are used.

    The first run downloads ~13 MB from the University of Surrey and caches the preprocessed images in
    `data/chars74k/`.
    """)
    return


@app.cell
def _(
    C74K_DIR,
    C74K_STROKE_LEVELS,
    C74K_URL,
    Image,
    ImageFilter,
    download_and_extract_archive,
    mo,
    ndimage,
    np,
    torch,
):
    # ARTICLE ▸ Sec. III-A: "Chars74K images follow the EMNIST conversion [3]: inversion, cropping,
    #           centring with a 2-pixel border at 128-pixel scale, Gaussian blur with σ = 1, bicubic
    #           resizing to 28×28 and scaling to [0, 1]... they are thinned by erosion, applied at the
    #           128-pixel scale before blurring."
    #
    # Turns each raw 1200×900 Chars74K scan into an EMNIST-like 28×28 image, once for every candidate
    # stroke level. Each step below follows one item of that sentence.
    # NOTE ▸ The article's writer-level bootstrap treats the second number in each file name
    #        (imgKKK-NNN.png, NNN = 1..55) as the writer. That is done in py_file/v2/reanalysis.py.
    #        This cell only needs the class and the case.
    _img_root = C74K_DIR / "English" / "Hnd" / "Img"
    _cache = C74K_DIR / "chars74k_letters_28x28.pt"

    def _preprocess(path, levels):
        ink = (255 - np.asarray(Image.open(path).convert("L"))) > 127   # inversion: white ink on black, like EMNIST
        ys, xs = np.nonzero(ink)
        ink = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]          # cropping to the letter
        h, w = ink.shape
        side = max(h, w)
        pad = round(2 * side / 128)                                       # 2-pixel border at 128-pixel scale
        frame = side + 2 * pad
        square = np.zeros((frame, frame), bool)                           # centring, aspect ratio kept
        top, left = pad + (side - h) // 2, pad + (side - w) // 2
        square[top:top + h, left:left + w] = ink
        # Erosion: each ink pixel's distance to the nearest background pixel. Keeping only pixels farther
        # than r from the edge thins every stroke by r. One distance map serves all levels.
        dist = ndimage.distance_transform_edt(square)
        out = []
        for level in levels:
            img = (Image.fromarray((dist > level * frame / 128).astype(np.uint8) * 255)
                   .filter(ImageFilter.GaussianBlur(radius=frame / 128))  # Gaussian blur, σ = 1 at 128-pixel scale
                   .resize((28, 28), Image.BICUBIC))                      # bicubic resizing to 28x28
            a = np.asarray(img, np.float32)
            out.append(a / max(a.max(), 1.0))                             # scaling to [0, 1] (full range)
        return np.stack(out)

    _c = torch.load(_cache, weights_only=True) if _cache.exists() else None
    if _c is None or _c["levels"] != list(C74K_STROKE_LEVELS):
        if not _img_root.exists():
            download_and_extract_archive(C74K_URL, download_root=str(C74K_DIR))
        # Folder Sample011..036 = "A".."Z", Sample037..062 = "a".."z" (Sample001..010 are digits, skipped).
        # Upper and lower case of a letter get the same class: "Cases are merged as in EMNIST".
        _items = []
        for _k in range(11, 63):
            _cls, _case = (_k - 11, "upper") if _k <= 36 else (_k - 37, "lower")
            for _f in sorted((_img_root / f"Sample{_k:03d}").glob("*.png")):
                _items.append((_f, _cls, _case))
        _X = np.stack([_preprocess(_f, C74K_STROKE_LEVELS)
                       for _f, _, _ in mo.status.progress_bar(_items, title="Preprocessing Chars74K")])
        _c = {"levels": list(C74K_STROKE_LEVELS),
              # Stored as 8-bit integers to keep the cache small; shape (levels, N, 28, 28)
              "X": torch.from_numpy(np.round(_X.transpose(1, 0, 2, 3) * 255).astype(np.uint8)),
              "y": torch.tensor([_cls for _, _cls, _ in _items]),
              "is_upper": torch.tensor([_case == "upper" for _, _, _case in _items])}
        torch.save(_c, _cache)

    # The case of each image is kept: Sec. V-B and Fig. 4b split every result into upper- and lowercase
    c74k = {
        "X_by_level": {_l: _c["X"][_i].float().div(255.0).unsqueeze(1) for _i, _l in enumerate(_c["levels"])},
        "y": _c["y"],
        "case": np.where(_c["is_upper"].numpy(), "upper", "lower"),
    }
    print(f"Chars74K letters: {len(c74k['y'])} images, {len(torch.unique(c74k['y']))} classes, "
          f"per class {torch.bincount(c74k['y']).min().item()}..{torch.bincount(c74k['y']).max().item()}")
    return (c74k,)


@app.cell
def _(C74K_STROKE_LEVELS, c74k, data, mo, pd, plt):
    # ARTICLE ▸ Eq. (1): k* = argmin_k |ink(k) − ink_EMNIST|. Sec. III-A: "Their strokes are almost twice as
    #           thick (mean ink [level 0] against [EMNIST])... This uses no Chars74K labels and selects
    #           level [k*]". Section VI: "both machines select the same stroke-thinning level with the
    #           same mean ink to six decimals".
    #
    # "Mean ink" = the average pixel value of the images, i.e. how much of the frame the strokes cover.
    # The table printed below lists it for EMNIST's training images and for Chars74K at every level.
    # Level 0 is the unthinned "ink_raw" of the article. The chosen level is the closest match.
    # NOTE ▸ Sec. V-B and Fig. 4a then check that the choice does not drive the results, by
    #        re-evaluating all models at levels 2–6. That sweep is in py_file/v2/reanalysis.py.
    _target = data["letters"]["X_train"].mean().item()
    stroke_table = pd.DataFrame(
        [{"source": "EMNIST Letters (train)", "stroke level": None, "mean ink": _target,
          "% pixels > 0.5": 100 * (data["letters"]["X_train"] > 0.5).float().mean().item()}]
        + [{"source": "Chars74K", "stroke level": _l, "mean ink": _X.mean().item(),
            "% pixels > 0.5": 100 * (_X > 0.5).float().mean().item()}
           for _l, _X in c74k["X_by_level"].items()])
    # Eq. (1)
    C74K_LEVEL = min(C74K_STROKE_LEVELS, key=lambda _l: abs(c74k["X_by_level"][_l].mean().item() - _target))
    # The Chars74K images used by every later cell
    X_ext = c74k["X_by_level"][C74K_LEVEL]

    def _():
        # Visual check: one EMNIST letter per class (top) against Chars74K upper and lower case
        d = data["letters"]
        fig, axes = plt.subplots(3, 26, figsize=(20, 2.8))
        rows = [("EMNIST", d["X_test"], d["y_test"].numpy(), None),
                ("C74K upper", X_ext, c74k["y"].numpy(), "upper"),
                ("C74K lower", X_ext, c74k["y"].numpy(), "lower")]
        for r, (name, X, y, case) in enumerate(rows):
            for c in range(26):
                mask = (y == c) if case is None else (y == c) & (c74k["case"] == case)
                axes[r, c].imshow(X[mask.nonzero()[0][0], 0], cmap="gray")
                axes[r, c].set_xticks([])
                axes[r, c].set_yticks([])
                if r == 0:
                    axes[r, c].set_title(d["class_names"][c], fontsize=9)
            axes[r, 0].set_ylabel(name, fontsize=8)
        fig.suptitle(f"EMNIST Letters vs. Chars74K after preprocessing (stroke level {C74K_LEVEL})")
        plt.tight_layout()
        return plt.show()

    _()
    mo.output.append(mo.md(f"**Stroke matching:** EMNIST mean ink = {_target:.4f}; chosen Chars74K level = "
                           f"**{C74K_LEVEL}**. If the images above don't look like EMNIST letters, fix this "
                           f"before reading any Part E number."))
    mo.output.append(stroke_table.round(4))
    return C74K_LEVEL, X_ext, stroke_table


@app.cell
def _(
    DEVICE,
    VARIANTS,
    X_ext,
    c74k,
    calibration_metrics,
    data,
    f1_score,
    fit_temperature,
    np,
    pd,
    predict_logits,
    probs_from_logits,
    risk_coverage,
    runs,
):
    # ARTICLE ▸ The data behind all of Section V: Table IX (EMNIST, Chars74K, Drop), Sec. V-A ("Every CNN
    #           loses between ... points"; "MLP-128 ... breaks"), Fig. 4b (drop on upper- and lowercase,
    #           columns c74k_upper_acc / c74k_lower_acc), Sec. V-C (spread of the baseline's runs on
    #           Chars74K vs EMNIST) and Fig. 7a (ECE before and after the EMNIST temperature).
    #
    # Each trained EMNIST Letters run is rebuilt from its weights and scored on Chars74K with the same
    # metric functions as Parts A and G, so every Chars74K number has an EMNIST twin.
    # NOTE ▸ The saved per-run file (partE_per_run.csv) is pooled over both machines by data.py. The
    #        95% CI of each Chars74K accuracy, quoted in Sec. V-A, comes from a bootstrap over seeds and
    #        the 55 writers, computed in data.py from reanalysis.py's per-image results.
    _d = data["letters"]
    _y = c74k["y"].numpy()
    _upper = c74k["case"] == "upper"
    ext_logits = {}
    _rows = []
    for (_ds, _v, _s), _r in runs.items():
        if _ds != "letters":
            continue
        _m = VARIANTS[_v](_d["num_classes"])
        _m.load_state_dict(_r["state_dict"])
        _z = predict_logits(_m.to(DEVICE), X_ext)            # dropout off, as for EMNIST test
        ext_logits[(_v, _s)] = _z
        _ok = _z.argmax(1).numpy() == _y
        _T = fit_temperature(_r["val_logits"], _d["y_val"])  # T from EMNIST validation, exactly as in Part G
        _p1 = probs_from_logits(_z)
        _before = calibration_metrics(_p1, _y)
        _after = calibration_metrics(probs_from_logits(_z, _T), _y)
        _rc = risk_coverage(_p1, _y)
        _emnist_acc = 100 * (_r["test_logits"].argmax(1).numpy() == _d["y_test"].numpy()).mean()
        _rows.append({
            "variant": _v, "seed": _s,
            "emnist_acc": _emnist_acc, "c74k_acc": 100 * _ok.mean(),
            "drop_pp": _emnist_acc - 100 * _ok.mean(),       # the "Drop" column of Table IX
            "c74k_upper_acc": 100 * _ok[_upper].mean(), "c74k_lower_acc": 100 * _ok[~_upper].mean(),
            "c74k_macro_f1": 100 * f1_score(_y, _z.argmax(1).numpy(), average="macro"),
            # Mean confidence: Sec. V-E compares it with accuracy ("mean confidence exceeds its accuracy by
            # ... points"), a check on calibration that does not depend on the 15 bins
            "mean_conf": float(_p1.max(1).mean()), "T": _T,
            **{f"{_k}_before": _before[_k] for _k in ("nll", "brier", "ece", "mce")},
            **{f"{_k}_after": _after[_k] for _k in ("nll", "brier", "ece", "mce")},
            **{_k: _rc[_k] for _k in ("aurc", "acc@90cov", "acc@95cov", "acc@99cov", "cov@99acc")},
        })
    ext_per_run = pd.DataFrame(_rows)
    _cols = [_c for _c in ext_per_run.columns if _c not in ("variant", "seed")]
    _g = ext_per_run.groupby("variant", sort=False)[_cols]
    summary_e = _g.mean().add_suffix("_mean").join(_g.std().add_suffix("_std")).reset_index()
    summary_e = summary_e.sort_values("emnist_acc_mean", ascending=False, ignore_index=True)
    np.testing.assert_equal(len(ext_logits), len(ext_per_run))
    return ext_logits, ext_per_run, summary_e


@app.cell
def _(BASELINE, fmt_ms, mo, pd, summary_e):
    # ARTICLE ▸ Table IX (The Two Letters Datasets Compared: EMNIST, Chars74K, Drop) and Sec. V-C "The
    #           Ranking Tells the Same Story": "Spearman ρ = ..., from one adjacent swap (the baseline and
    #           the LeNet-5-style network)". Compare the "rank EMNIST" and "rank Chars74K" columns to
    #           see the swap. Fig. 5 plots the same two accuracies against each other.
    #
    # NOTE ▸ The "Eff. rob." column of Table IX and the probit line of Fig. 5 are not computed here. They
    #        come from data.py, which fits Chars74K accuracy against EMNIST accuracy on a probit scale
    #        over the CNNs. A model's effective robustness is its distance above (+) or below (−) that line.
    _t = summary_e[["variant"]].copy()
    for _col, _name in [("emnist_acc", "EMNIST acc"), ("c74k_acc", "Chars74K acc"), ("drop_pp", "drop (pp)"),
                        ("c74k_upper_acc", "upper"), ("c74k_lower_acc", "lower"),
                        ("c74k_macro_f1", "Chars74K macro-F1")]:
        _t[_name] = [fmt_ms(m, s) for m, s in zip(summary_e[f"{_col}_mean"], summary_e[f"{_col}_std"])]
    _base = summary_e.set_index("variant").loc[BASELINE]
    _t["Δ vs baseline EMNIST"] = (summary_e["emnist_acc_mean"] - _base["emnist_acc_mean"]).round(2).values
    _t["Δ vs baseline Chars74K"] = (summary_e["c74k_acc_mean"] - _base["c74k_acc_mean"]).round(2).values
    _t["rank EMNIST"] = summary_e["emnist_acc_mean"].rank(ascending=False).astype(int).values
    _t["rank Chars74K"] = summary_e["c74k_acc_mean"].rank(ascending=False).astype(int).values

    # Rank agreement over the 10 models: 1 = identical order on both datasets
    rank_agreement = pd.DataFrame([{
        "spearman": summary_e["emnist_acc_mean"].corr(summary_e["c74k_acc_mean"], method="spearman"),
        "kendall": summary_e["emnist_acc_mean"].corr(summary_e["c74k_acc_mean"], method="kendall"),
        "n_variants": len(summary_e)}])
    mo.output.append(mo.md("**E1. Accuracy on EMNIST test vs. Chars74K** (mean ± std over seeds, sorted by "
                           "EMNIST accuracy; drop = EMNIST − Chars74K)"))
    mo.output.append(_t)
    mo.output.append(mo.md(f"**Ranking agreement EMNIST → Chars74K:** Spearman ρ = "
                           f"{rank_agreement.spearman[0]:.2f}, Kendall τ = {rank_agreement.kendall[0]:.2f} "
                           f"(1 = same order; computed on 10 variant means, so read it as indicative)."))
    return (rank_agreement,)


@app.cell
def _(BASELINE, SEEDS, VARIANTS, c74k, ext_logits, mcnemar, mo, pd):
    # ARTICLE ▸ Table XI, last two rows: the LeNet-5-style network and Width 16/32 on Chars74K, whose
    #           number of significant seeds differs a lot between the machines. Sec. VI-B: "one machine
    #           alone would have reported a robustness weakness ... the other finds ...".
    #
    # The same McNemar test as A.3 (Eq. 2), on the 2,860 Chars74K images.
    # NOTE ▸ The Chars74K column of Table V and Fig. 2 uses the bootstrap intervals of data.py instead,
    #        because with only 2,860 images from 55 writers, one seed's verdict is noisy.
    _y = c74k["y"].numpy()
    _rows = []
    for _v in VARIANTS:
        if _v == BASELINE:
            continue
        for _s in SEEDS:
            _cx = ext_logits[(_v, _s)].argmax(1).numpy() == _y
            _cy = ext_logits[(BASELINE, _s)].argmax(1).numpy() == _y
            _b, _c, _p = mcnemar(_cx, _cy)
            _rows.append({"variant": _v, "seed": _s, "b": _b, "c": _c, "p": _p,
                          "acc_diff_pp": 100 * (_cx.mean() - _cy.mean())})
    mcnemar_ext = (pd.DataFrame(_rows).groupby("variant", sort=False)
                   .agg(acc_diff_pp=("acc_diff_pp", "mean"), b_mean=("b", "mean"), c_mean=("c", "mean"),
                        p_median=("p", "median"),
                        seeds_significant=("p", lambda p: f"{(p < 0.05).sum()}/{len(p)}"))
                   .reset_index())
    mo.output.append(mo.md(f"**E2. McNemar vs. {BASELINE} on Chars74K** (b = variant right & baseline wrong)"))
    mo.output.append(mcnemar_ext.round(4))
    return (mcnemar_ext,)


@app.cell
def _(plt, summary_e):
    # ARTICLE ▸ Sec. V-A "Accuracy Bends but Does Not Break". The notebook's view of the EMNIST → Chars74K
    #           part of Fig. 1 (the article's Fig. 1 and Fig. 5 are drawn by py_file/v2/figures.py).
    #
    # Each row is a model: circle = EMNIST accuracy, square = Chars74K accuracy, the grey line joins them.
    # Long lines are large drops (MLP-128). If the order of the squares matched the order of the
    # circles exactly, the ranking would be unchanged.
    def _():
        s = summary_e.iloc[::-1]
        fig, ax = plt.subplots(figsize=(9, 5))
        yy = range(len(s))
        ax.hlines(yy, s["c74k_acc_mean"], s["emnist_acc_mean"], color="grey", alpha=0.5)
        ax.errorbar(s["emnist_acc_mean"], yy, xerr=s["emnist_acc_std"].fillna(0), fmt="o", label="EMNIST test")
        ax.errorbar(s["c74k_acc_mean"], yy, xerr=s["c74k_acc_std"].fillna(0), fmt="s", label="Chars74K (external)")
        ax.set_yticks(list(yy), s["variant"])
        ax.set(xlabel="Test accuracy (%)", title="Letters models: EMNIST test vs. external Chars74K")
        ax.grid(axis="x", alpha=0.3)
        ax.legend()
        plt.tight_layout()
        return plt.show()

    _()
    return


@app.cell
def _(fmt_ms, mo, summary_e):
    # ARTICLE ▸ Sec. V-E "Trust Is What Does Not Travel" and Fig. 7: the baseline's ECE rises from EMNIST to
    #           Chars74K ("ece_before" here vs Table VII); "The EMNIST temperature still lowers the ECE
    #           for [most] of the ten models; it raises it only for [those] whose temperature [is] below 1"
    #           (compare ece_before with ece_after and look at T); and the reference coverage at 99% on
    #           Chars74K ("cov@99acc", Fig. 7b).
    #
    # Mean confidence above accuracy = over-confident, the typical behaviour under a change of data
    # source reported in [19].
    # NOTE ▸ The "decisive test" of Sec. V-E, a 99% threshold fixed on EMNIST validation and applied
    #        unchanged to Chars74K, is computed in py_file/v2/reanalysis.py. "cov@99acc" here chooses the
    #        cut-off with the Chars74K labels, so it is only the "for reference" number of that paragraph.
    _t = summary_e[["variant"]].copy()
    _t["T (EMNIST val)"] = [fmt_ms(m, s) for m, s in zip(summary_e["T_mean"], summary_e["T_std"])]
    _t["mean conf"] = [fmt_ms(100 * m, 100 * s) for m, s in zip(summary_e["mean_conf_mean"], summary_e["mean_conf_std"])]
    _t["accuracy"] = [fmt_ms(m, s) for m, s in zip(summary_e["c74k_acc_mean"], summary_e["c74k_acc_std"])]
    for _m, _scale, _dig in [("ece", 100, 2), ("nll", 1, 3)]:
        for _w in ("before", "after"):
            _t[f"{_m}_{_w}"] = [fmt_ms(_scale * m, _scale * s, _dig) for m, s in
                                zip(summary_e[f"{_m}_{_w}_mean"], summary_e[f"{_m}_{_w}_std"])]
    for _m in ("acc@90cov", "cov@99acc"):
        _t[_m] = [fmt_ms(m, s) for m, s in zip(summary_e[f"{_m}_mean"], summary_e[f"{_m}_std"])]
    _t["AURC_%"] = [fmt_ms(100 * m, 100 * s, 3) for m, s in zip(summary_e["aurc_mean"], summary_e["aurc_std"])]
    mo.output.append(mo.md("**E3. Calibration and selective prediction on Chars74K** (ECE in %; T fitted on "
                           "EMNIST validation, not refitted; mean conf > accuracy means overconfident)"))
    mo.output.append(_t)
    return


@app.cell
def _(
    BASELINE,
    SEEDS,
    c74k,
    calibration_metrics,
    data,
    ext_logits,
    fit_temperature,
    plt,
    probs_from_logits,
    runs,
):
    # ARTICLE ▸ Sec. V-E, as a picture, for one seed of the baseline. Not a figure in the article.
    #
    # Left and middle: reliability diagrams on Chars74K before and after the EMNIST temperature (read
    # them like the Part G diagrams: red above the bars = over-confident). Right: the baseline's
    # confidence on EMNIST test and on Chars74K. On new writers it stays high while accuracy falls.
    def _():
        r, d = runs[("letters", BASELINE, SEEDS[0])], data["letters"]
        z, y = ext_logits[(BASELINE, SEEDS[0])], c74k["y"].numpy()
        T = fit_temperature(r["val_logits"], d["y_val"])
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
        for ax, (label, temp) in zip(axes[:2], [("before", 1.0), (f"after EMNIST T={T:.2f}", T)]):
            m = calibration_metrics(probs_from_logits(z, temp), y)
            width = m["edges"][1] - m["edges"][0]
            centers = m["edges"][:-1] + width / 2
            ax.bar(centers, m["bin_acc"], width=width, edgecolor="black", alpha=0.8, label="accuracy")
            ax.bar(centers, m["bin_conf"] - m["bin_acc"], bottom=m["bin_acc"], width=width,
                   color="red", alpha=0.3, edgecolor="red", label="gap to confidence")
            ax.plot([0, 1], [0, 1], "k--", label="perfect calibration")
            ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Confidence", ylabel="Accuracy",
                   title=f"Chars74K, {label}: ECE = {100 * m['ece']:.2f}%")
            ax.legend(loc="upper left", fontsize=8)
        axes[2].hist(probs_from_logits(r["test_logits"]).max(1), bins=50, alpha=0.6, density=True,
                     label="EMNIST test")
        axes[2].hist(probs_from_logits(z).max(1), bins=50, alpha=0.6, density=True, label="Chars74K")
        axes[2].set(xlabel="Confidence (max softmax)", ylabel="Density", title="Confidence distribution")
        axes[2].legend(fontsize=8)
        fig.suptitle(f"{BASELINE}, seed {SEEDS[0]}: reliability on the external test set")
        plt.tight_layout()
        return plt.show()

    _()
    return


@app.cell
def _(BASELINE, SEEDS, c74k, data, ext_logits, mo, np, pd):
    # ARTICLE ▸ Fig. 6 (accuracy of the baseline on each Chars74K letter, uppercase and lowercase) and
    #           Sec. V-D "Where New Writers Hurt": "The hardest lowercase letters are ..., and uppercase
    #           I is read as l in [x]% of cases ... The other frequent mistakes ... differ only in a short
    #           hook or tail." Also Sec. V-B: "Uppercase letters do not drop at all... The whole drop is
    #           lowercase."
    #
    # Pooled over the seeds of the baseline (the article pools the 10 runs of both machines).
    # The first table is Fig. 6 in numbers: accuracy per letter, separately for its upper- and
    # lowercase form. The second lists the most frequent mistakes. "% of that letter+case" is the share
    # of that letter's 55 images × seeds that went to the wrong class, e.g. the I→l percentage.
    _names = data["letters"]["class_names"]
    _y = c74k["y"].numpy()
    _case = c74k["case"]
    _preds = np.stack([ext_logits[(BASELINE, _s)].argmax(1).numpy() for _s in SEEDS])   # (seeds, images)
    _ok = _preds == _y
    ext_class_table = pd.DataFrame([{
        "letter": _names[_c],
        "upper_acc": 100 * _ok[:, (_y == _c) & (_case == "upper")].mean(),
        "lower_acc": 100 * _ok[:, (_y == _c) & (_case == "lower")].mean(),
    } for _c in range(len(_names))])
    ext_class_table["lower − upper"] = ext_class_table["lower_acc"] - ext_class_table["upper_acc"]

    _true = np.repeat(_y[None], len(SEEDS), 0)[~_ok]
    _pred = _preds[~_ok]
    _casew = np.repeat(_case[None], len(SEEDS), 0)[~_ok]
    ext_confusions = (pd.DataFrame({"true": [_names[i] for i in _true], "case": _casew,
                                    "predicted": [_names[i] for i in _pred]})
                      .value_counts().rename("count").reset_index().head(15))
    ext_confusions["% of that letter+case"] = (100 * ext_confusions["count"] / (55 * len(SEEDS))).round(1)
    mo.output.append(mo.md(f"**E4. Per-letter accuracy on Chars74K, {BASELINE}** (pooled over seeds; sorted "
                           f"by the worse case)"))
    mo.output.append(ext_class_table.assign(_w=ext_class_table[["upper_acc", "lower_acc"]].min(1))
                     .sort_values("_w").drop(columns="_w").round(1))
    mo.output.append(mo.md("**Top confusions on Chars74K** (true letter and case → predicted class)"))
    mo.output.append(ext_confusions)
    return ext_class_table, ext_confusions


@app.cell
def _(
    C74K_LEVEL,
    RESULTS_DIR,
    c74k,
    ext_class_table,
    ext_confusions,
    ext_per_run,
    json,
    mcnemar_ext,
    rank_agreement,
    stroke_table,
    summary_e,
):
    # ARTICLE ▸ Inputs to Section V. py_file/v2/data.py reads these files from each machine's results folder:
    #   partE_per_run.csv          Table IX, Fig. 4b, Fig. 5, Fig. 7 (pooled over both machines)
    #   partE_mcnemar.csv          Table XI (Chars74K rows)
    #   partE_per_letter.csv       Fig. 6
    #   partE_confusions.csv       Sec. V-D (top confusions, I→l)
    #   partE_stroke_matching.csv  Sec. III-A (mean ink), Section VI (same level on both machines)
    #   partE_config.json          the chosen stroke level k* and the number of images
    _tables = {"partE_per_run": ext_per_run, "partE_summary": summary_e, "partE_rank_agreement": rank_agreement,
               "partE_mcnemar": mcnemar_ext, "partE_per_letter": ext_class_table,
               "partE_confusions": ext_confusions, "partE_stroke_matching": stroke_table}
    for _name, _df in _tables.items():
        _df.to_csv(RESULTS_DIR / f"{_name}.csv", index=False)
    (RESULTS_DIR / "partE_config.json").write_text(json.dumps({
        "dataset": "Chars74K EnglishHnd (letters only, case merged)", "n_images": int(len(c74k["y"])),
        "stroke_level": C74K_LEVEL, "temperature": "fitted on EMNIST validation (Part G), not refitted",
    }, indent=2))
    print(f"Saved {len(_tables)} Part E tables + partE_config.json to {RESULTS_DIR}/")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Save results (input to every table of the article, and to Section VI)

    Each machine's results folder was copied to `report/results/mac_mps` and
    `report/results/windows_cuda`. Section VI ("Does the Story Survive a Second Machine?") compares the
    two folders file by file: same-seed differences, ranking, effect sizes and flipped verdicts.
    """)
    return


@app.cell
def _(
    DEVICE,
    EPOCHS,
    QUICK,
    RESULTS_DIR,
    SEEDS,
    calib_per_run,
    cost_table,
    gap_table,
    json,
    mcnemar_table,
    np,
    ood_table,
    per_run,
    platform,
    selective_table,
    sklearn,
    summary_a,
    summary_g,
    table7,
    torch,
    torchvision,
):
    # ARTICLE ▸ The files the article is built from (read by py_file/v2/data.py):
    #   partA_per_run.csv    Tables IV–V, Figs. 1–2, Table X (seed vs machine)
    #   partA_cost.csv       Table VI, Fig. 3 (parameters and MACs from the Mac; latency from each machine)
    #   partA_mcnemar.csv    Table XI (digits and EMNIST rows)
    #   partG_per_run.csv    Table VII, Fig. 7
    #   partG_ood.csv        Table VIII
    #   environment.json     Table III
    # Section VI also notes that an earlier execution on the same Mac reproduced these tables byte for
    # byte, which is checked by comparing two copies of these CSV files.
    _tables = {
        "partA_per_run": per_run, "partA_summary": summary_a, "partA_gap": gap_table.reset_index(),
        "partA_cost": cost_table, "partA_mcnemar": mcnemar_table, "partA_table7": table7,
        "partG_per_run": calib_per_run, "partG_summary": summary_g, "partG_selective": selective_table,
        "partG_ood": ood_table,
    }
    for _name, _df in _tables.items():
        _df.to_csv(RESULTS_DIR / f"{_name}.csv", index=_name == "partA_table7")
    (RESULTS_DIR / "environment.json").write_text(json.dumps({
        "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
        "torchvision": torchvision.__version__, "sklearn": sklearn.__version__,
        "platform": platform.platform(), "cpu": platform.processor(), "device": str(DEVICE),
        "quick": QUICK, "seeds": SEEDS, "epochs": EPOCHS,
    }, indent=2))
    print(f"Saved {len(_tables)} tables + environment.json to {RESULTS_DIR}/")
    return


if __name__ == "__main__":
    app.run()
