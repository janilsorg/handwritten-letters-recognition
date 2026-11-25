import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    # Reading guide. This notebook is the experiment behind the replication report
    # ("Replicação Experimental de uma CNN Leve...", relatorio_replicacao_ieee.docx). Its results are
    # also summarised in Section II of the extension article (03_article_ieee_v2).
    mo.md(r"""
    # Replication of Mohammed (2026) on MNIST: annotated copy

    This is the notebook behind the **replication report**. Its single-seed numbers are also the
    starting point of Section II ("The Base Model and its Claims") of the extension article. The code
    is the same as in `../01_replication_mnist.py`. Only the comments and these markdown cells are new.

    **How to read the comments.** Every cell starts with an `ARTICLE ▸` line naming the section,
    table or figure it produces. Lines marked `NOTE ▸` point out where the report reads more into a
    number than the cell prints.

    | Notebook cell | Replication report | Extension article |
    |---|---|---|
    | Environment | *Ambiente Experimental* | — |
    | Data split, class counts | *Metodologia › Dados* | Table II (Digits column) |
    | Model + parameter/MAC count | *Arquitetura*, Table I | Section II (37,610 params, 354,048 MACs) |
    | Training (seed 42) | *Treinamento*, Fig. 1 | Section II (C7 discussion, Sec. IV-B) |
    | Test accuracy | Table II | Section II (single-seed reference) |
    | Confusion matrix | Fig. 2, *Análise de Erros* | — |
    | Per-class P/R/F1 | Table III, Fig. 3 | — |
    | Size and latency | Table II | Table VI (re-measured there) |
    | Ablation | Table IV, Fig. 4, *Inconsistências* | Section II (misprinted counts) |

    The whole study uses one seed (42). Section *Limitações* of the report explains why differences of
    0.1–0.2 points should not be over-read here. The extension notebook repeats everything with five seeds.
    """)
    return


@app.cell
def _():
    # ARTICLE ▸ Report, "Ambiente Experimental": Python, PyTorch, torchvision, scikit-learn versions and
    #           the device used for training (the report's run: Apple GPU through MPS).
    #
    # Imports for the whole notebook, then the device choice: an NVIDIA GPU (cuda) if present, otherwise
    # the Apple GPU (mps), otherwise the CPU. The printout is what the report copies into its
    # environment paragraph, so that another reader can tell whether their setup matches.
    import copy
    import os
    import platform
    import random
    import time

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import sklearn
    import torch
    import torch.nn as nn
    import torchvision
    from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                                 confusion_matrix, precision_recall_fscore_support)
    from torch.utils.data import DataLoader, TensorDataset
    from torchvision import datasets

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

    # Figures and the saved model are written here (see the training-curve, confusion-matrix and size cells)
    os.makedirs("results", exist_ok=True)
    return (
        ConfusionMatrixDisplay,
        DEVICE,
        DataLoader,
        TensorDataset,
        classification_report,
        confusion_matrix,
        copy,
        datasets,
        nn,
        np,
        os,
        pd,
        plt,
        precision_recall_fscore_support,
        random,
        time,
        torch,
    )


@app.cell
def _(np, random, torch):
    # ARTICLE ▸ Report, "Treinamento": "seed 42 was also applied to weight initialisation and batch
    #           shuffling, so that the experiments can be repeated."
    #
    # The paper only says seed 42 is used for the train/validation split. Here the same seed also fixes
    # every other random choice (initial weights, shuffle order), so re-running the notebook on the same
    # machine gives the same model. The two cudnn flags make NVIDIA GPUs deterministic too. They have no
    # effect on MPS, which is why the report's limitations note that MPS training may not be bit-exact.
    SPLIT_SEED = 42

    def set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    set_seed(SPLIT_SEED)
    return SPLIT_SEED, set_seed


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Data (report: *Metodologia › Dados*)

    60,000 training and 10,000 test images of 28×28 grey-scale digits. The only preprocessing is
    scaling pixels to [0, 1]; there is no augmentation.
    """)
    return


@app.cell
def _(datasets):
    # ARTICLE ▸ Report, "Dados": the standard MNIST train (60,000) and test (10,000) sets.
    # torchvision downloads them into ./data on the first run.
    train_raw = datasets.MNIST(root="data", train=True, download=True)
    test_raw = datasets.MNIST(root="data", train=False, download=True)
    return test_raw, train_raw


@app.cell
def _(plt, train_raw):
    # A visual sanity check, not used in the report: the first five training digits with their labels.
    # If the title does not match the picture, the data are being read wrongly.
    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    for i, ax in enumerate(axes):
        img, label = train_raw[i]      # a PIL image and its integer label
        ax.imshow(img, cmap="gray")
        ax.set_title(label)
        ax.axis("off")
    plt.show()

    # The same images as a raw tensor: 5 images of 28 x 28 pixels, values 0..255
    print(train_raw.data[:5].shape)
    print(train_raw.targets[:5])
    return


@app.cell
def _(test_raw, train_raw):
    # ARTICLE ▸ Report, "Dados": "pixels were normalised to [0, 1]".
    #
    # Divide by 255 so pixels lie in [0, 1], and add a channel axis: PyTorch expects images as
    # (number of images, channels, height, width) = (N, 1, 28, 28).
    X_full = train_raw.data.float().div(255.0).unsqueeze(1)
    y_full = train_raw.targets.clone()
    X_test = test_raw.data.float().div(255.0).unsqueeze(1)
    y_test = test_raw.targets.clone()
    return X_full, X_test, y_full, y_test


@app.cell
def _(SPLIT_SEED, X_full, X_test, torch, y_full, y_test):
    # ARTICLE ▸ Report, "Dados": 6,000 training images set aside for validation with seed 42, leaving
    #           54,000 for training; and the observation that the test set is NOT balanced (892 images
    #           of digit 5 up to 1,135 of digit 1), contrary to the paper's Table 4.
    #           Extension article, Table II, "Digits" column (54,000 / 6,000 / 10,000).
    #
    # One random permutation of the 60,000 training images with seed 42: the first 6,000 go to
    # validation, the rest to training. The test set is never touched until the final evaluation.
    g = torch.Generator().manual_seed(SPLIT_SEED)
    perm = torch.randperm(len(X_full), generator=g)
    val_idx, train_idx = perm[:6000], perm[6000:]

    X_train, y_train = X_full[train_idx], y_full[train_idx]
    X_val, y_val = X_full[val_idx], y_full[val_idx]

    print("Train:", tuple(X_train.shape), "Val:", tuple(X_val.shape), "Test:", tuple(X_test.shape))
    print("Pixel range:", X_train.min().item(), "-", X_train.max().item())

    # Count of test images per digit. This line is the source of the "892 to 1,135" statement.
    print("Test samples per class:", torch.bincount(y_test).tolist())
    return X_train, X_val, y_train, y_val


@app.cell
def _(
    DataLoader,
    TensorDataset,
    X_test,
    X_train,
    X_val,
    torch,
    y_test,
    y_train,
    y_val,
):
    # ARTICLE ▸ Report, "Treinamento": batches of 128 images.
    #
    # A DataLoader hands the images to the model in batches. Only the training loader shuffles, and its
    # shuffle order comes from the seed, so two runs with the same seed see the batches in the same order.
    # The evaluation batch size (1000) only affects speed, not results.
    BATCH_SIZE = 128
    EVAL_BATCH_SIZE = 1000

    def make_loaders(seed: int):
        g = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(TensorDataset(X_train, y_train),
                                  batch_size=BATCH_SIZE, shuffle=True, generator=g)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=EVAL_BATCH_SIZE)
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=EVAL_BATCH_SIZE)
        return train_loader, val_loader, test_loader

    return (make_loaders,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Architecture (report: *Arquitetura*, Table I)

    Two 5×5 convolutions with 8 and 16 filters (no padding), each followed by ReLU and 2×2
    max-pooling, then a 128-unit dense layer with dropout 0.5 and a 10-class output.
    """)
    return


@app.cell
def _(nn):
    # ARTICLE ▸ Report, "Arquitetura" and Table I (layer, output, parameters, MACs).
    #           Extension article, Section II: Conv(8, 5×5) → MaxPool → Conv(16, 5×5) → MaxPool →
    #           Dense(128) → Dropout(0.5) → Dense(K).
    #
    # The arguments c1, c2, dense and dropout exist so that the ablation cell at the end can build the
    # paper's variants (16/32 filters, 64 dense units, no dropout) from the same class. With the default
    # values this is exactly the paper's model. The comments on the right follow the image size through
    # the network: 28×28 in, 4×4 feature maps out, which is the "Saída" column of Table I.
    class LightweightCNN(nn.Module):
        def __init__(self, c1: int = 8, c2: int = 16, dense: int = 128, dropout: float = 0.5):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, c1, kernel_size=5, stride=1),   # 28 -> 24
                nn.ReLU(),
                nn.MaxPool2d(2),                             # 24 -> 12
                nn.Conv2d(c1, c2, kernel_size=5, stride=1),  # 12 -> 8
                nn.ReLU(),
                nn.MaxPool2d(2),                             # 8 -> 4
            )
            # Flatten turns the c2 maps of 4x4 into one vector of c2*16 numbers (256 for the baseline)
            head = [nn.Flatten(), nn.Linear(c2 * 4 * 4, dense), nn.ReLU()]
            if dropout > 0:
                head.append(nn.Dropout(dropout))
            head.append(nn.Linear(dense, 10))
            self.classifier = nn.Sequential(*head)

        def forward(self, x):
            # Returns raw scores (logits). The softmax of the paper's output layer is applied inside the
            # loss function during training, and is not needed to pick the predicted class.
            return self.classifier(self.features(x))

    model = LightweightCNN()
    print(model)
    return LightweightCNN, model


@app.cell
def _(copy, model, nn, pd, torch):
    # ARTICLE ▸ Report, Table I (per-layer parameters and MACs, total 37,610 and 354,048) and the
    #           "Parâmetros" / "MACs por imagem" rows of Table II.
    #           Extension article, Section II: "The architecture matches [1] exactly (37,610 parameters,
    #           354,048 multiply–accumulate operations, or MACs, per image)".
    #
    # Counts parameters and multiply-accumulate operations (MACs) layer by layer. A MAC is one
    # "multiply two numbers and add to a running sum", the basic cost unit of a neural network.
    #   - Convolution: each output value is a dot product over (input channels × 5 × 5) inputs.
    #   - Dense layer: inputs × outputs.
    #   - Pooling, ReLU and Flatten: counted as zero.
    # The two asserts stop the notebook if the model differs from the paper by even one parameter.
    def count_params(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def complexity_table(model: nn.Module, input_size=(1, 1, 28, 28)) -> pd.DataFrame:
        rows, hooks = [], []

        def make_hook(name):
            # A "forward hook" is called each time a layer runs, so it can record the layer's output size
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

        # Pass one blank image through a CPU copy of the model, so the original is left untouched
        model_cpu = copy.deepcopy(model).cpu().eval()
        for name, module in model_cpu.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.MaxPool2d, nn.Flatten)):
                hooks.append(module.register_forward_hook(make_hook(name)))
        with torch.no_grad():
            model_cpu(torch.zeros(input_size))
        for h in hooks:
            h.remove()

        df = pd.DataFrame(rows)
        df.loc[len(df)] = {
            "layer": "TOTAL",
            "output_shape": "",
            "params": df["params"].sum(),
            "MACs": df["MACs"].sum()
        }
        return df

    cx = complexity_table(model)
    cx

    n_params = count_params(model)
    total_macs = int(cx.iloc[-1]["MACs"])
    print(f"Parameters: {n_params:,}  (paper: 37,610)")
    print(f"MACs/image: {total_macs:,}  (paper: 354,048)")
    assert n_params == 37_610, "Architecture does not match the paper"
    assert total_macs == 354_048, "MAC count does not match the paper"
    return complexity_table, count_params, cx


@app.cell
def _(cx):
    # ARTICLE ▸ Report, Table I: this displayed table is copied there row by row.
    cx
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Training (report: *Treinamento*, *Curvas de Aprendizado*)

    Adam with learning rate 0.001, cross-entropy loss, batch size 128 and 10 epochs. No weight decay,
    no learning-rate schedule and no early stopping, as in the paper.
    """)
    return


@app.cell
def _(DEVICE, SPLIT_SEED, make_loaders, nn, pd, time, torch):
    # ARTICLE ▸ Report, "Treinamento" (the recipe) and "Métricas" (accuracy and loss).
    #
    # Two helpers used by every later training cell:
    #   fit()      trains for 10 epochs and records train/validation accuracy and loss after each epoch;
    #              these records become the learning curves of Fig. 1.
    #   evaluate() runs the trained model on a dataset and returns loss, accuracy and predictions,
    #              which feed Table II, the confusion matrix and Table III.
    #
    # NOTE ▸ Training accuracy is measured while the model trains, i.e. with dropout switched ON, while
    #        validation is measured with dropout OFF (model.train(False)). This is why the measured
    #        curves show validation ABOVE training, the opposite of the paper's simulated curves. The
    #        report explains it under "Curvas de Aprendizado", and the extension article (Sec. IV-B,
    #        claim C7) recomputes training accuracy with dropout off to settle the question.
    EPOCHS = 10
    LR = 1e-3

    def run_epoch(model, loader, criterion, optimizer=None):
        """One pass over `loader`. Trains if an optimizer is given, otherwise evaluates."""
        is_train = optimizer is not None
        model.train(is_train)          # train mode = dropout active; eval mode = dropout off
        total_loss, correct, n = 0.0, 0, 0
        with torch.set_grad_enabled(is_train):
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                loss = criterion(logits, yb)
                if is_train:
                    # The three standard steps: clear old gradients, back-propagate, update weights
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item() * yb.size(0)
                correct += (logits.argmax(dim=1) == yb).sum().item()
                n += yb.size(0)
        return total_loss / n, correct / n


    def fit(model, seed=SPLIT_SEED, epochs=EPOCHS, lr=LR, verbose=True) -> pd.DataFrame:
        train_loader, val_loader, _ = make_loaders(seed)
        criterion = nn.CrossEntropyLoss()      # cross-entropy, with the softmax built in
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
        rows = []
        for epoch in range(1, epochs + 1):
            t0 = time.perf_counter()
            tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer)
            va_loss, va_acc = run_epoch(model, val_loader, criterion)
            dt = time.perf_counter() - t0
            rows.append({"epoch": epoch,
                         "train_acc": tr_acc * 100, "val_acc": va_acc * 100,
                         "train_loss": tr_loss, "val_loss": va_loss,
                         "epoch_time_s": dt})
            if verbose:
                print(f"Epoch {epoch:2d}/{epochs} | "
                      f"train acc {tr_acc*100:6.2f}% loss {tr_loss:.4f} | "
                      f"val acc {va_acc*100:6.2f}% loss {va_loss:.4f} | {dt:.1f}s")
        return pd.DataFrame(rows)


    @torch.no_grad()
    def evaluate(model, loader):
        """Returns (loss, accuracy %, predictions, true labels)."""
        model.eval()
        criterion = nn.CrossEntropyLoss(reduction="sum")
        total_loss, preds, labels = 0.0, [], []
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            total_loss += criterion(logits, yb).item()
            preds.append(logits.argmax(dim=1).cpu())      # predicted class = highest score
            labels.append(yb.cpu())
        preds, labels = torch.cat(preds).numpy(), torch.cat(labels).numpy()
        return total_loss / len(labels), (preds == labels).mean() * 100, preds, labels

    return evaluate, fit


@app.cell
def _(DEVICE, LightweightCNN, SPLIT_SEED, fit, mo, pd, set_seed):
    # ARTICLE ▸ Report, Fig. 1 (paper's simulated curves on the left, these measured curves on the right)
    #           and Table II rows "Acurácia de validação (época 10)" and "Tempo médio por época (MPS)".
    #
    # Trains the baseline once with seed 42 and puts the measured history next to the paper's Table 3,
    # typed in by hand below. The paper's values are "simulated typical outcomes", not measurements, so
    # this table is the first real comparison. Watch epoch 1: in the report's run validation is about
    # 10 points ahead of training (95.13% vs 85.23%); in the paper training is ahead throughout.
    set_seed(SPLIT_SEED)
    model2 = LightweightCNN().to(DEVICE)
    history = fit(model2, seed=SPLIT_SEED)

    # The paper's Table 3 (simulated), for comparison
    paper_table3 = pd.DataFrame({
        "epoch": range(1, 11),
        "paper_train_acc": [88.7, 94.0, 96.5, 97.6, 98.2, 98.6, 98.9, 99.0, 99.2, 99.3],
        "paper_val_acc":   [85.0, 93.5, 95.5, 96.8, 97.5, 98.0, 98.1, 98.2, 98.3, 98.4],
        "paper_train_loss": [0.3695, 0.2, 0.12, 0.08, 0.06, 0.04, 0.03, 0.03, 0.025, 0.02],
        "paper_val_loss":   [0.4458, 0.21, 0.15, 0.1, 0.08, 0.07, 0.06, 0.058, 0.055, 0.05],
    })
    table3 = history.merge(paper_table3, on="epoch")
    mo.output.append(table3.round(4))

    print(f"Mean time per epoch: {history['epoch_time_s'].mean():.2f}s on {DEVICE}")
    return history, model2


@app.cell
def _(history, plt):
    # ARTICLE ▸ Report, Fig. 1, right panel: measured accuracy (and loss) per epoch.
    #           Saved as results/training_curves.png.
    #
    # Left: accuracy; right: loss. Two things the report reads from this plot: validation stays above
    # training in every epoch (explained by dropout, see the training cell), and validation accuracy is
    # still rising at epoch 10, so a few more epochs might help (report, "Curvas de Aprendizado").
    def _():
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(history["epoch"], history["train_acc"], "o-", label="Train")
        axes[0].plot(history["epoch"], history["val_acc"], "o-", label="Validation")
        axes[0].set(title="Accuracy vs. epochs", xlabel="Epoch", ylabel="Accuracy (%)")
        axes[0].legend(); axes[0].grid(alpha=0.3)

        axes[1].plot(history["epoch"], history["train_loss"], "o-", label="Train")
        axes[1].plot(history["epoch"], history["val_loss"], "o-", label="Validation")
        axes[1].set(title="Loss vs. epochs", xlabel="Epoch", ylabel="Cross-entropy loss")
        axes[1].legend(); axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig("results/training_curves.png", dpi=150)
        return plt.show()


    _()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Results on the test set (report: *Resultados*)
    """)
    return


@app.cell
def _(SPLIT_SEED, evaluate, make_loaders, model2):
    # ARTICLE ▸ Report, Table II: test accuracy (report's run: 98.87%), test loss (0.0349) and
    #           "Erros no teste" (113 / 10,000), against the paper's simulated ~98.7% and ~0.05.
    #           Extension article, Section II: "C1 holds where it was made".
    #
    # The first and only time the test set is used for the baseline. 0.17 points above the paper is
    # 17 images; with a single seed that is within run-to-run noise, so the report reads it as
    # "confirms the paper", not "beats it".
    _, _, test_loader = make_loaders(SPLIT_SEED)
    test_loss, test_acc, y_pred, y_true = evaluate(model2, test_loader)
    print(f"Test accuracy: {test_acc:.2f}%  (paper: ~98.7%)")
    print(f"Test loss    : {test_loss:.4f}  (paper: ~0.05)")
    print(f"Errors       : {(y_pred != y_true).sum()} / {len(y_true)}")
    return test_loader, y_pred, y_true


@app.cell
def _(ConfusionMatrixDisplay, confusion_matrix, mo, pd, plt, y_pred, y_true):
    # ARTICLE ▸ Report, Fig. 2 (confusion matrix) and "Análise de Erros": the largest error cell is
    #           14 fives predicted as 3, but only 1 three predicted as 5 (strongly asymmetric); 7→2 shows
    #           the same pattern (9 vs 2); 4↔9 is the only roughly symmetric pair (7 vs 5).
    #           Saved as results/confusion_matrix.png.
    #
    # Rows are the true digit, columns the predicted digit; the diagonal holds the correct answers.
    # NOTE ▸ Fig. 3 of the report (false negatives and false positives per digit) is read off this
    #        matrix: a digit's false negatives are its row total minus the diagonal cell, its false
    #        positives its column total minus the diagonal cell. In a 10-class problem every error is
    #        one of each, so both totals equal the number of errors (113).
    cm = confusion_matrix(y_true, y_pred, labels=range(10))
    cm_df = pd.DataFrame(cm, index=[f"true {i}" for i in range(10)],
                         columns=[f"pred {i}" for i in range(10)])
    cm_df["support"] = cm.sum(axis=1)
    mo.output.append(cm_df)

    fig2, ax2 = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=range(10)).plot(ax=ax2, cmap="Blues", colorbar=False)
    ax2.set_title("Confusion matrix on the MNIST test set (counts)")
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png", dpi=150)
    plt.show()

    # The five largest off-diagonal cells as (count, true digit, predicted digit).
    # The paper mentions 5 vs 3 and 9 vs 4; this list is where 7→2 was found as well.
    off_diag = [(cm[i, j], i, j) for i in range(10) for j in range(10) if i != j and cm[i, j] > 0]
    print("Top confusions (count, true, predicted):", sorted(off_diag, reverse=True)[:5])
    return


@app.cell
def _(
    classification_report,
    mo,
    pd,
    precision_recall_fscore_support,
    y_pred,
    y_true,
):
    # ARTICLE ▸ Report, Table III (per-class metrics) and equations (1)–(3) of "Métricas":
    #           precision P = TP/(TP+FP), recall R = TP/(TP+FN), F1 = 2PR/(P+R).
    #           In the report's run, digit 5 has the lowest recall (97.65%) and digit 3 the lowest
    #           precision (97.38%), the two ends of the 5→3 confusion.
    #
    # The table also has two averages: "macro" weights every digit equally; "weighted" weights each
    # digit by its number of test images. They differ slightly because the test set is unbalanced.
    print(classification_report(y_true, y_pred, digits=4))

    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred, labels=range(10))
    table5 = pd.DataFrame({"digit": range(10), "precision": p * 100, "recall": r * 100,
                           "f1": f1 * 100, "support": s})
    for avg in ["macro", "weighted"]:
        ap, ar, af, _ = precision_recall_fscore_support(y_true, y_pred, average=avg)
        table5.loc[len(table5)] = [f"{avg} avg", ap * 100, ar * 100, af * 100, len(y_true)]
    mo.output.append(table5.round(2))
    return


@app.cell
def _(copy, count_params, model2, np, os, time, torch):
    # ARTICLE ▸ Report, Table II rows "Tamanho dos pesos (float32)" (≈0.143 MB), "Latência em CPU
    #           (mediana / p95)" (0.069 / 0.076 ms in the report's run) and "Vazão em CPU (lote de 256)".
    #           The latency protocol (1 thread, batch of one image, 50 warm-up calls, 1,000 timed calls,
    #           median and 95th percentile) is described in "Ambiente Experimental".
    #           The extension article re-measures latency for every variant on both machines (Table VI).
    #
    # The paper calls the model "lightweight" but never times it; this cell does.
    #   - Size: each parameter is a 32-bit float = 4 bytes. The saved file is slightly larger because it
    #     also stores layer names.
    #   - Latency: time to classify one image on one CPU thread, like a small embedded device reading one
    #     character at a time. Warm-up calls are discarded because the first calls are always slower.
    #     The median is the typical call; the 95th percentile is a realistic worst case.
    #   - Throughput: images per second when 256 images are processed together.
    theoretical_mb = count_params(model2) * 4 / 1024**2
    torch.save(model2.state_dict(), "results/lightweight_cnn.pt")
    file_mb = os.path.getsize("results/lightweight_cnn.pt") / 1024**2
    print(f"Weights (float32): {theoretical_mb:.3f} MB  |  saved state_dict: {file_mb:.3f} MB")

    @torch.no_grad()
    def measure_latency(model, device="cpu", batch_size=1, n_warmup=50, n_runs=1000, threads=1):
        torch.set_num_threads(threads)
        m = copy.deepcopy(model).to(device).eval()
        x = torch.randn(batch_size, 1, 28, 28, device=device)   # random input: speed doesn't depend on content
        for _ in range(n_warmup):
            m(x)
        times = []
        for _ in range(n_runs):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            m(x)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
        torch.set_num_threads(os.cpu_count())
        return np.median(times), np.percentile(times, 95)

    med, p95 = measure_latency(model2, device="cpu", batch_size=1)
    print(f"CPU latency, batch=1, 1 thread: median {med:.3f} ms, p95 {p95:.3f} ms")
    med_b, _ = measure_latency(model2, device="cpu", batch_size=256, n_runs=200)
    print(f"CPU throughput, batch=256: {256 / (med_b / 1000):,.0f} images/s")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Ablation study (report: *Estudo de Ablação*, Table IV, Fig. 4)

    The paper's three variants, each changing one thing, trained with the same recipe and seed.
    """)
    return


@app.cell
def _(
    DEVICE,
    LightweightCNN,
    SPLIT_SEED,
    complexity_table,
    count_params,
    evaluate,
    fit,
    mo,
    pd,
    set_seed,
    test_loader,
):
    # ARTICLE ▸ Report, Table IV (paper vs. measured parameters, validation and test accuracy) and
    #           Fig. 4 (test accuracy against parameters). Also "Inconsistências no Artigo Original":
    #           the paper lists 37,482 parameters for "No Dropout" (dropout has no parameters, so it must
    #           be 37,610) and 77,130 for "16/32 filters" (Flatten gives 512 values, not 256, so 80,202).
    #           Extension article, Section II repeats these two misprints. Its Section IV-B then re-runs
    #           the same ablations with five seeds on two machines and on letters (claims C4–C6).
    #
    # Each variant gets the same seed as the baseline, so the only difference between them is the
    # architecture change. `paper_table6` holds the paper's simulated values, typed in by hand.
    # In the report's run, 16/32 filters is the only clear gain: the error rate falls from 1.13% to
    # 0.69%, about 39% fewer errors, for 2.1x the parameters.
    ablation_variants = {
        "Baseline (Dropout=0.5, 8/16 filters)": dict(c1=8, c2=16, dense=128, dropout=0.5),
        "No Dropout":                           dict(c1=8, c2=16, dense=128, dropout=0.0),
        "More Filters (16/32)":                 dict(c1=16, c2=32, dense=128, dropout=0.5),
        "Smaller Dense (64)":                   dict(c1=8, c2=16, dense=64, dropout=0.5),
    }
    paper_table6 = {  # (params, val acc, test acc) as printed in the paper (simulated)
        "Baseline (Dropout=0.5, 8/16 filters)": (37610, 98.4, 98.7),
        "No Dropout":                           (37482, 98.1, 98.4),
        "More Filters (16/32)":                 (77130, 98.7, 98.9),
        "Smaller Dense (64)":                   (20522, 98.2, 98.5),
    }

    ablation_rows = []
    for name, cfg in ablation_variants.items():
        print(f"\n=== {name} ===")
        set_seed(SPLIT_SEED)
        m = LightweightCNN(**cfg).to(DEVICE)
        h = fit(m, seed=SPLIT_SEED, verbose=False)
        _, t_acc, _, _ = evaluate(m, test_loader)
        macs = int(complexity_table(m).iloc[-1]["MACs"])
        pp, pv, pt = paper_table6[name]
        ablation_rows.append({"variant": name, "params": count_params(m), "paper_params": pp,
                              "MACs": macs, "val_acc@10": h["val_acc"].iloc[-1], "paper_val_acc": pv,
                              "test_acc": t_acc, "paper_test_acc": pt})
        print(f"val {h['val_acc'].iloc[-1]:.2f}% | test {t_acc:.2f}% | params {count_params(m):,}")

    # Compare "params" with "paper_params": the two rows where they differ are the misprints
    table6 = pd.DataFrame(ablation_rows)
    mo.output.append(table6.round(2))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
