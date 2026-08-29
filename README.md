# CAN-DETECTION

**A lightweight recurrence-plot CNN for multi-class intrusion detection on the automotive CAN bus.**

This repository contains the code developed for my MSc thesis, *"Evaluating Lightweight Recurrence Plot CNNs for Multi-Class Intrusion Detection in Controller Area Networks"* (Criss Zachariah Parayathukattil, 25006195).

In simple terms: CAN bus communication does not provide built-in security mechanisms such as encryption or authentication. This project explores a lightweight CNN-based approach that converts CAN traffic into recurrence-plot images and uses them to distinguish normal traffic from four types of attacks.

---

## What's actually going on here

Modern vehicles contain many Electronic Control Units (ECUs) responsible for functions such as the engine, braking, and steering. These ECUs communicate with each other through the CAN bus. CAN was originally designed with reliability and real-time communication in mind rather than security, so it does not provide features such as encryption or sender authentication.

The approach used in this project starts with a sequence of CAN arbitration IDs collected over a fixed window. The sequence is converted into a two-dimensional recurrence plot, where repeated CAN IDs produce corresponding patterns in the image. These images are then provided to a small CNN for classification.

The model is deliberately kept lightweight. It uses two convolutional layers followed by a `Flatten → Dense(32)` head:

```
Conv(64) → Conv(16) → Flatten → Dense(32)
```

The dropout rate is selected using a Hyperband search rather than being chosen manually. This architecture follows the specification given in the project proposal.

---

## How well does it actually work?

The final model was trained and evaluated using all four HCRL Car-Hacking attack files:

- DoS
- Fuzzy
- Gear-spoofing
- RPM-spoofing

The data were divided using a group-aware `GroupShuffleSplit`. This prevents overlapping windows from being placed in both the training and test sets.

**Final results:**

| Metric | Value |
|---|---|
| Test accuracy | 97.50% |
| Training accuracy | 98.41% |
| Macro precision | 0.9744 |
| Macro recall | 0.9760 |
| Macro F1 | 0.9750 |
| Macro AUC (one-vs-rest) | 0.9991 |
| Macro FPR | 0.0062 |

These results are strong on the benchmark test set. However, the frequency-sweep experiment gives a more important qualification to these numbers.

When the attack injection frequency was reduced, detection performance fell substantially, reaching approximately **0–8%** across the usable periods tested. **RPM-spoofing detection remained at 0% throughout the sweep.**

This behaviour is related to the input representation used by the model. The recurrence plot is constructed from CAN arbitration-ID equality, while Gear- and RPM-spoofing attacks can modify data-byte values without changing the arbitration ID. As a result, changes caused by these attacks may not be represented in the recurrence plot itself.

The frequency-sweep results and their implications are discussed in **Section 4.7** of the thesis.

---

## What's in this repository

**`final_multiclass.ipynb`**
An earlier version of the multi-class pipeline using a global-average-pooling classifier head. It achieved 96.87% test accuracy, but differed from the proposal in several respects, including the model architecture, hyperparameter search, evaluation procedure, and out-of-vocabulary handling. It is retained to make the development history of the project traceable.

**`final.ipynb`**
The final pipeline used for the thesis results. It contains the preprocessing, recurrence-plot generation, Hyperband dropout search, model training, evaluation, and low-frequency injection sweep.

**`app_multiclass_new_UPDATED.py`**
A Streamlit dashboard developed to demonstrate how the trained model could be presented in an operational setting. It contains four pages covering the main model information, monitoring, evaluation results, and recurrence-plot visualisation.

**`lightweight_can_cnn_multiclass_new1.pth`**
The trained model checkpoint produced by the final pipeline. The checkpoint also contains the saved `CAN_ID_MAP`, allowing CAN IDs to be indexed consistently during inference.

> Earlier pilot and binary-only experiments are not included in the repository. They are described as part of the development history in the thesis, but they are not used for the final reported results.

---

## Getting the dataset

The project uses the **HCRL Car-Hacking Dataset**, published by the Hacking and Countermeasure Research Lab at Korea University. The dataset is available for academic use, but it is not redistributed in this repository.

You will need the following four files:

- `DoS_dataset.csv`
- `Fuzzy_dataset.csv`
- `gear_dataset.csv`
- `RPM_dataset.csv`

Place them in the project root before running the notebook.

Each row represents a CAN frame and contains information such as the timestamp, CAN ID, DLC, data bytes, and attack flag.

---

## Setup

```bash
pip install torch scikit-learn pandas numpy streamlit
```

Python 3.9 or later is recommended.

---

## Running the final pipeline

1. Obtain the four HCRL dataset files.
2. Place them in the project root.
3. Open `final.ipynb`.
4. Run the notebook from beginning to end.

The notebook performs the preprocessing, recurrence-plot generation, Hyperband search, model training, evaluation, and low-frequency injection sweep. The trained checkpoint is saved at the end of the process.

The experiments use:

```
SEED = 42
```

The seed is applied to NumPy, Python's `random` module, and PyTorch. However, the Hyperband search can still show some variation between runs, particularly in the dropout value selected. This behaviour is discussed as a limitation in **Section 5.6** of the thesis.

---

## Running the dashboard

After obtaining the trained checkpoint, place:

```
lightweight_can_cnn_multiclass_new1.pth
```

in the project directory and run:

```bash
streamlit run app_multiclass_new_UPDATED.py
```

It is recommended to start the application from a fresh Streamlit process. An already-running process may retain stale cached results after source-code or model changes.

---

## Citing the dataset

If you use the HCRL Car-Hacking Dataset, please cite the original publication:

> Song, H.M., Woo, J. and Kim, H.K. (2020) In-vehicle network intrusion detection using deep convolutional neural network. *Vehicular Communications*, 21, p.100198.

---

## Who made this

**Criss Zachariah Parayathukattil** (25006195) — MSc Cyber Security
