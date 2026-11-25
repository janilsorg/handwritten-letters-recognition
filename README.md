# Beyond Digits: A Lightweight CNN Baseline on Two Handwritten Letter Datasets and Two Machines

Code for the paper by **Janilson Mariano de Andrade** (CIn/UFPE).

The study takes the 37,610-parameter digit CNN proposed by Mohammed (2026) and tests whether its claims hold when moving from MNIST digits to **EMNIST Letters** (26 classes), to a second independent dataset (**Chars74K**, new writers), and to a second machine (Mac/MPS vs. Windows/CUDA).

## Main findings

- The baseline drops from **98.95%** (digits) to **92.55%** (EMNIST Letters) and **87.40%** (Chars74K), while its lead over classical methods grows.
- Letters overturn the original design conclusions: filter width and dense-layer size matter, removing dropout gives a small gain, and batch normalization is the cheapest improvement (+0.67 points at the same compute).
- The drop on Chars74K is almost entirely on **lowercase** letters; models stay on a single probit line (slope 1.03), so better models stay better.
- **Confidence does not transfer**: a threshold set on EMNIST for 99% accuracy delivers 97.4% on new writers.
- Across machines all 27 effects keep their sign and model rankings are identical, but significance verdicts near the threshold flip. Effect sizes with confidence intervals reproduce; yes/no verdicts do not.

## Setup

Requires Python 3.12 (recommended; some dependencies have no prebuilt wheels for very new Python versions and will ask for a C++ compiler on Windows).

```bash
git clone https://github.com/janilsorg/handwritten-letters-recognition.git
cd handwritten-letters-recognition

python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Or with [uv](https://docs.astral.sh/uv/), without creating a venv manually:

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
```

## Running the notebooks

All notebooks are [marimo](https://marimo.io) notebooks (plain `.py` files).

Open a notebook in the editor:

```bash
python -m marimo edit 02-extension-v2.py
```

Run it as a read-only app:

```bash
python -m marimo run 02-extension-v2.py
```

Run it as a plain script (no UI):

```bash
python 02-extension-v2.py
```

### Quick mode

Set `EXT_QUICK=1` for a faster, reduced run (useful for testing the setup before the full experiment).

```bash
# Linux/macOS
EXT_QUICK=1 python -m marimo edit 02-extension-v2.py
```

```powershell
# Windows PowerShell
$env:EXT_QUICK = "1"; python -m marimo edit 02-extension-v2.py
```

Unset it for the full run: `Remove-Item Env:EXT_QUICK` (PowerShell) or `unset EXT_QUICK` (bash).

### Notebooks

| File | Description |
|------|-------------|
| `01-*.py` | Baseline reproduction on MNIST digits |
| `02-extension-v2.py` | Extension to EMNIST Letters: variants, cost, calibration, selective prediction, OOD |
| `03-*.py` | Chars74K comparison and cross-machine analysis |

> Adjust the names and descriptions above to match the files in this repository.

## Reproducibility notes

- Ten architectures, seeds 0-4, on each of two machines (200 runs total).
- Both machines used the same code, seeds and data:

| | Mac | Windows |
|---|---|---|
| GPU | Apple GPU (MPS) | RTX 4060 Ti (CUDA 12.6) |
| Python | 3.14.6 | 3.12.10 |
| PyTorch | 2.14.1 | 2.10.0 |

- GPU kernels are not bit-identical across platforms, so small differences between machines are expected. On a single machine, runs are deterministic.

## Datasets

- [MNIST](http://yann.lecun.com/exdb/mnist/)
- [EMNIST](https://www.nist.gov/itl/products-and-services/emnist-dataset) (Letters split)
- [Chars74K](https://info-ee.surrey.ac.uk/CVSSP/demos/chars74k/) (handwritten letters, used only for testing)

## Citation

```bibtex
@misc{andrade2026beyonddigits,
  author = {Janilson Mariano de Andrade},
  title  = {Beyond Digits: A Lightweight CNN Baseline on Two Handwritten Letter Datasets and Two Machines},
  year   = {2026},
  note   = {Centro de Inform\'atica, Universidade Federal de Pernambuco}
}
```

## License

Add your license here (e.g. MIT).
