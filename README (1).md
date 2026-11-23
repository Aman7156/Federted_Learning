# Federated Learning Across Very Different Medical Images

**Can hospitals that hold completely different kinds of medical images train one shared model, without ever sharing a single image, and still do as well as if they had pooled all their data?**

This repository contains the experiments for my bachelor's thesis at the University of Messina (Apr – Oct 2026, advisor Prof. Salvatore Distefano). It compares centralized training, local-only training and six federated learning algorithms on five MedMNIST datasets that have almost nothing visually in common: chest X-rays, breast ultrasound, skin photographs, tissue slides and retina photographs.

---

## The idea in one paragraph

Most federated learning research assumes the clients hold the *same kind* of data with different statistics, such as X-rays from different hospitals. Here each client holds a *different imaging modality*, which is the hard case. I measured how different the clients really are with **linear CKA**, grouped them into four heterogeneity levels from weak to strong, and at every level compared centralized training against every federated method on the same healthy-versus-diseased task.

---

## Clients

Each dataset is one client. Its images never leave it.

| Client | Modality | Native classes | Train images |
|---|---|---|---|
| PneumoniaMNIST | Chest X-ray | 2 | 2,000 |
| BreastMNIST | Breast ultrasound | 2 | 546 |
| DermaMNIST | Dermatoscopy (skin) | 7 | 2,000 |
| PathMNIST | Histopathology (tissue) | 9 | 2,000 |
| RetinaMNIST | Fundus (retina) | 5 | 1,080 |

## Methods compared

| Type | Methods |
|---|---|
| Baselines | Centralized (all data pooled), Local-only (each client alone) |
| Weight-sharing federated | FedAvg, FedProx, FedAdam |
| Other federated | Federated Distillation (FedDF), FedMix, FedMoE (mixture of experts) |

**Setup:** ResNet-18 pretrained on ImageNet, 96x96 images, 8 rounds x 4 local epochs (32 epochs total, matched to the centralized baseline), batch size 64. BatchNorm layers stay local to each client (FedBN), and distillation uses the MedMNIST *validation* split as its public set, so no test data leaks into training.

---

## Heterogeneity levels

Heterogeneity is measured as the mean pairwise linear CKA between the clients' representations. Lower CKA means the datasets are more different.

| Level | Clients | Mean CKA | Heterogeneity |
|---|---|---|---|
| L1 (weak) | Derma, Path | 0.329 | 0.671 |
| L2 | + Retina | 0.271 | 0.729 |
| L3 | + Breast | 0.211 | 0.789 |
| L4 (strong) | all five | 0.164 | 0.836 |

## Results

Mean accuracy across clients, healthy vs diseased, one seed:

| Method | L1 (weak) | L2 | L3 | L4 (strong) |
|---|---|---|---|---|
| Centralized | 0.836 | 0.769 | 0.804 | 0.807 |
| Local-only | 0.840 | 0.786 | 0.814 | 0.815 |
| FedAvg | 0.803 | 0.782 | 0.777 | 0.831 |
| FedProx | 0.836 | 0.785 | 0.774 | 0.826 |
| FedAdam | 0.832 | 0.804 | **0.816** | 0.824 |
| FedMix | 0.805 | 0.785 | 0.787 | **0.835** |
| FedMoE (3 experts) | **0.833** | **0.813** | 0.795 | 0.834 |
| Federated Distillation | 0.828 | 0.661 | 0.618 | 0.601 |

### What I found

1. **Weight-sharing methods kept up with centralized training at every level.** Even at the strongest heterogeneity (all five modalities), FedAvg, FedMix and FedMoE reached about 83% mean accuracy, matching or slightly beating the centralized model (80.7%), without any client sharing an image.
2. **Federated Distillation fell apart as the data became more different.** It was competitive at the weakest level (82.8%) but dropped to 66.1%, 61.8% and then 60.1% as more modalities were added. Averaging predictions on a small public set carries too little shared signal when the clients see completely different kinds of images.
3. **Distillation is cheaper to communicate:** about half the traffic of the weight-sharing methods (for example 1.8 GB against 3.6 GB at L4), but in this setting the accuracy cost is far too high.

**Caveat:** these numbers come from a single seed, so differences of 1 to 3 points between the weight-sharing methods are within normal run-to-run variation. The clear pattern is the gap between weight sharing and distillation as heterogeneity grows.

---

## Notebooks

The repository also contains the earlier experiments that led to the final design. The main ones are:

| Notebook | What it does |
|---|---|
| `run_hetero (3).ipynb` | **Main result:** the four heterogeneity levels, centralized vs every federated method (outputs included) |
| `federated_medical_v4.ipynb` | Full thesis campaign: label granularity, fair epoch matching, mixture-of-experts sweep, bootstrapping a new client, privacy noise, scaling with number of clients |
| `run_missing.ipynb`, `run_remaining.ipynb` | Fill-in runs for configurations the main campaign missed |
| `federated_medical_multiclass_*.ipynb`, `federated_transfer_learning_*.ipynb` | Earlier versions of the medical experiments (binary and multiclass) |
| `hfl_*.ipynb`, `six_federated_algorithms_cifar10.ipynb`, `two_client_cifar_stl_algo_comparison.ipynb` | First experiments on MNIST and CIFAR-10 with the Flower framework, used to validate the algorithms before moving to medical data |

---

## How to run

The notebooks were run on Google Colab and Kaggle with a GPU.

1. Open `run_hetero (3).ipynb` in Colab or Kaggle and choose a GPU runtime.
2. Run all cells. The first cell installs the dependencies:
   ```bash
   pip install torch torchvision medmnist scikit-learn
   ```
3. The MedMNIST datasets download automatically. Results are saved to `results.json` as each experiment finishes, so a re-run skips anything already done. The full run takes about 2 to 3 hours.

The Flower (CIFAR-10 and MNIST) notebooks also need `pip install flwr matplotlib numpy`.

---

## Tech stack

Python, PyTorch, torchvision (ResNet-18), MedMNIST, scikit-learn, Flower (flwr), NumPy, Matplotlib

## References

The algorithms are established methods: FedAvg (McMahan et al., 2017), FedProx (Li et al., 2020), FedAdam / FedOpt (Reddi et al., 2021), Federated Distillation / FedDF (Jeong et al., 2018; Lin et al., 2020), FedBN (Li et al., 2021), FedMix (Yoon et al., 2021), mixture of experts (Shazeer et al., 2017), CKA (Kornblith et al., 2019). Datasets: MedMNIST (Yang et al., 2023).

## Author

**Aman Kumar Singh**  
M.Sc. Data Science and Engineering, Politecnico di Torino  
[LinkedIn](https://linkedin.com/in/aman-kumar-singh-4a632a1a0) · [GitHub](https://github.com/Aman7156)
