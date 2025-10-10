# FedGIE — Gradient-Free Federated Learning via Layer-wise Least Squares

**FedGIE** is a gradient-free federated learning framework. Each layer update is solved as a **least-squares** problem with a **Moore–Penrose pseudoinverse**, avoiding backpropagation and black-box gradient estimation. A top-down **feedback projection** plus a ReLU **diagonal Jacobian correction** stabilizes update directions under strong Non-IID data. The repository includes both **MLP** and **CNN** reference models and supports **MNIST**, **Fashion-MNIST**, and **CIFAR-10**.

> All `.py` sources are intentionally **comment-free** as requested.

---

## ✨ Features

- Closed-form per-layer updates (weights & bias via least squares with pseudoinverse).
- Top-down feedback projection to supervise lower layers without gradients.
- Activation-aware correction (diagonal Jacobian for ReLU).
- CNN support using `unfold/fold` to linearize convolutions for closed-form solutions.
- Federated training loop with broadcast + parameter averaging.
- Configurable Non-IID partitions via Dirichlet sampling.

---

## 🧱 Repository Layout

```
fedgie-multi/
  README.md
  requirements.txt
  train.py
  fedgie/
    __init__.py
    utils.py
    server.py
    client.py
    data/
      __init__.py
      partition.py
    models/
      __init__.py
      mlp.py
      cnn.py
```
- `train.py`: entrypoint (CLI, initialization, training, evaluation)
- `fedgie/server.py`: global model, broadcast, aggregation, evaluation
- `fedgie/client.py`: client-side closed-form local updates (Linear + Conv2d)
- `fedgie/models/`: MLP and CNN reference models
- `fedgie/data/partition.py`: datasets and Dirichlet Non-IID partitioning

---

## 🔧 Installation

**Requirements**
- Python ≥ 3.9
- PyTorch and TorchVision (CPU or CUDA builds)

```bash
python -m venv .venv
. .venv/bin/activate                # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
If you need GPU acceleration, install a CUDA-matching PyTorch wheel per the official PyTorch instructions, then install `torchvision`.

---

## 🚀 Quick Start

### MLP + MNIST
```bash
python train.py --dataset mnist --model mlp --clients 20 --rounds 100 --batch 32 --alpha 0.6
```

### CNN + MNIST
```bash
python train.py --dataset mnist --model cnn --clients 20 --rounds 100 --batch 32 --alpha 0.6
```

### CNN + CIFAR-10
```bash
python train.py --dataset cifar10 --model cnn --clients 20 --rounds 100 --batch 32 --alpha 0.6
```
TorchVision will auto-download datasets into `./data/`.

---

## ⚙️ Command-Line Arguments

| Argument     | Type   | Default | Description                                                                     |
|--------------|--------|---------|---------------------------------------------------------------------------------|
| `--dataset`  | str    | `mnist` | One of: `mnist`, `fashion_mnist`, `cifar10`.                                   |
| `--model`    | str    | `mlp`   | One of: `mlp`, `cnn`.                                                           |
| `--clients`  | int    | `20`    | Number of clients.                                                              |
| `--rounds`   | int    | `100`   | Number of federated rounds.                                                     |
| `--batch`    | int    | `32`    | Local batch size per client update.                                             |
| `--alpha`    | float  | `0.6`   | Dirichlet Non-IID strength (smaller = more skewed).                            |
| `--seed`     | int    | `42`    | Random seed.                                                                    |
| `--device`   | str    | `auto`  | `cpu`, `cuda`, or `auto` (use GPU if available).                                |

---

## 🧠 Method Overview

**Goal.** Avoid unstable black-box gradient estimation in federated settings by replacing backprop with structured, per-layer least-squares updates.

**Per-round, per-client outline:**
1. Run a single forward pass and cache each layer’s input `h` and pre-activation `z`.
2. At the top layer, define a target matrix `F` (e.g., one-hot labels, spatially broadcast for CNN).
3. Solve a bias-augmented linear regression in closed form:
   - Build `Ĥ = [1; Hᵀ]`, compute `Ŵ = F · pinv(Ĥ)`.
   - Extract `W = Ŵ[:,1:]`, `b = Ŵ[:,0]`.
4. Compute a top-down feedback signal for the previous layer by pseudo-inverting the updated mapping and apply ReLU diagonal Jacobian (element-wise mask on positive pre-activations).
5. Repeat for all layers down to the input.
6. Return local weights to the server; the server averages parameters to form the new global model.

**CNN specifics.** For `Conv2d`, use `torch.nn.functional.unfold` to produce local receptive-field matrices, solve the linear system in closed form, then use `fold` to project the feedback back to feature maps.

---

## 📊 Datasets & Partitioning

- Datasets: MNIST, Fashion-MNIST, CIFAR-10 (auto-downloaded to `./data/`).
- Non-IID Split: Dirichlet(α) over label distributions into `--clients` partitions.
  - Lower `alpha` → stronger heterogeneity.

---

## 🔎 Reproducibility

- Use `--seed` to fix randomness.
- The script prints test accuracy each round:
  ```
  round=1 acc=0.8123
  round=2 acc=0.8410
  ...
  ```
- Tip: redirect logs for analysis:
  ```bash
  python train.py ... | tee run.log
  ```

---

## 🧩 Extending the Project

**Add a new model**
- Create a file under `fedgie/models/` (e.g., `resnet.py`) exposing:
  - `layers`: list of modules to be updated in order (e.g., `Linear`/`Conv2d`).
  - `activations`: list of activation names aligned with `layers` (e.g., `["relu","relu","none"]`).
  - `forward(x)` and `forward_cache(x)` returning `(h_list, z_list)`.

**Add a new dataset**
- Extend `get_dataset` in `fedgie/data/partition.py` to return `(train, test, num_classes, in_dim_or_none)`.

---

## ⚠️ Known Limitations

- Memory/compute: `torch.linalg.pinv` may be heavy for large layers; reduce `--batch` or model width if needed.
- Pooling/strides: The CNN example focuses on a minimal consistent setup. When adding pooling or different strides/dilations, ensure `unfold/fold` parameters exactly match the convolution configuration.
- Aggregation: Default is uniform parameter averaging; you may replace it with data-size weighted averaging.

---

## 📦 Requirements

`requirements.txt` contains:
```
torch
torchvision
```
For GPU builds, install CUDA-compatible wheels as per PyTorch’s official guide.

---

## ❓ FAQ

**Q: Why no backprop or optimizer?**  
A: Each layer update is a closed-form least-squares solution, so no gradient steps are needed.

**Q: How is the classification target formed?**  
A: We use one-hot labels (or their spatially broadcast version for CNN), then propagate top-down with pseudoinverse and activation-aware correction.

**Q: Does it support GPU?**  
A: Yes. Set `--device cuda` or leave `--device auto` to use GPU if available.

---

## 📜 License & Citation

- License: Add a `LICENSE` file of your choice (e.g., MIT) at the repository root.
- Citation: If this repository is useful in your research or product, please cite it. Example:
```bibtex
@misc{fedgie2025,
  title  = {FedGIE: Gradient-Free Federated Learning via Layer-wise Least Squares},
  author = {Your Name and Coauthors},
  year   = {2025},
  note   = {Code available at: https://github.com/your-org/your-repo}
}
```

---

## 📬 Contact

For enhancements (e.g., residual blocks, batch normalization, sparsity, asynchronous aggregation) or issues, please open a GitHub Issue.
