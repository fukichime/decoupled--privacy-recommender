# Decoupled Architecture for Privacy-Compliant Recommendation ```

This repository implements a decoupled recommender system designed to address the "Right to be Forgotten" in production environments. Standard deep learning architectures (such as monolithic Transformers) entangle user and item representations, which makes removing a single user impossible without retraining the entire model. This project separates item learning from user history processing at the architectural level, so a user can be removed from the system by discarding their local parameters while the central item model remains untouched.

---

## Folder Structure

```
decoupled--privacy-recommender/
├── 01_DataPrep_Weights.ipynb
├── 02_Cloud_Item_Tower_Training.ipynb
├── 03_SASRec_Training.ipynb
├── 04_Evaluation.ipynb
├── final_utility_chart.png
├── final_privacy_chart.png
├── LICENSE
└── README.md
```
---

## System Architecture

The pipeline consists of two structurally isolated components:

- **Item Tower (Cloud / Public):** A Two-Tower network trained on global, anonymized interactions to produce item embeddings. Weights are extracted and frozen as a static `.npy` file, preventing any user-specific gradients from propagating into the central model.
- **User Tower (Edge / Private):** A local Adapter Layer (Linear → ReLU → Linear) projects frozen item embeddings into a dynamic latent space, which feeds a SASRec Transformer [4] that models each user's sequential interaction history locally.

To remove a user, only their local adapter parameters are discarded. The central Item Tower requires no modification.

---

## Methodology

### Data Preparation (`01_DataPrep_Weights.ipynb`)
- **Dataset:** KuaiRec 2.0 [2] is a fully observed, dense short-video interaction dataset (`big_matrix.csv` + `item_categories.csv`).
- Filtered for positive engagement: interactions with `watch_ratio ≥ 1.0` only.
- User IDs, Video IDs, and Category IDs remapped to consecutive integers via `LabelEncoder`. Final dataset: **7,176 users × 10,719 items**.
- Interaction sequences sorted chronologically per user. **Leave-One-Out** split: last item held out as test target.
- **Inverse Propensity Weighting (IPW)** [3] applied to counteract popularity bias. Weights assign higher training importance to niche items (~4.38) and lower importance to viral items (~0.06):

$$W_{\text{item}} = \text{Normalize}\left(\frac{1}{P(\text{item})^{0.5}}\right)$$

### Item Tower Training (`02_Cloud_Item_Tower_Training.ipynb`)
- Two-Tower model with User, Item, and Category embedding layers. Item and Category embeddings combined via element-wise addition; dot product with User embedding predicts interaction probability.
- Dynamic negative sampling: for each positive `(User, Item)` pair, one un-interacted item sampled per training step.
- Trained for **5 epochs** with Adam optimizer and IPW-weighted Binary Cross Entropy loss. Final training loss converged to ~**0.0005**.
- Item embedding matrix extracted and saved as `frozen_item_embeddings.npy` — physically detached from the PyTorch computational graph.

### SASRec Training (`03_SASRec_Training.ipynb`)
- Frozen embeddings loaded as a static lookup table (`freeze=True`).
- Adapter Layer projects static embeddings into the sequential attention space.
- Positional embeddings added to preserve interaction order within the Transformer.
- Training config: `MAX_LEN=100`, `STEP_SIZE=10` (sliding window augmentation), `BATCH_SIZE=128`, `LR=0.001`, `EPOCHS=10`. Cross Entropy loss with `ignore_index=0` to exclude padding tokens.
- **Exact Retraining** used for unlearning verification [5]: two models trained from scratch — **Baseline** (all users) and **Unlearned** (target user excluded). Both converged from ~8.11 to ~7.90 by Epoch 10, confirming one user's removal does not destabilize training.

### Evaluation (`04_Evaluation.ipynb`)
- **Utility:** Hit Rate@10 on held-out test targets for all non-forgotten users.
- **Privacy:** Cross Entropy Loss computed on the forgotten user's (User 4247) original training history. A high loss indicates the model can no longer predict that user's behavior — approximating random guessing.

---

## Results

### Utility — Hit Rate @ 10

| Model | Hit Rate @ 10 |
| :--- | :---: |
| Baseline (all users) | 0.0240 |
| **Unlearned (user removed)** | **0.0260** |

The unlearned model maintains and marginally exceeds baseline recommendation quality for the remaining user population, confirming the global Item Tower is sufficiently general.

<p align="center">
  <img src="final_utility_chart.png" width="480"/>
  <br><em>Figure 1: Hit Rate@10 comparison. Unlearning a single user has no negative impact on general system utility.</em>
</p>

### Privacy — Cross Entropy Loss on Forgotten User

| Model | Cross Entropy Loss |
| :--- | :---: |
| Baseline (remembers user) | 3.6130 |
| **Unlearned (user erased)** | **8.3252** |

The mathematical upper bound for random guessing on this dataset is ln(10,719) ≈ **9.22**. The unlearned model's loss of 8.33 approaches this limit, confirming effective erasure of the user's behavioral patterns.

<p align="center">
  <img src="final_privacy_chart.png" width="480"/>
  <br><em>Figure 2: Cross Entropy Loss on the forgotten user's history. The sharp increase in the unlearned model confirms data erasure approaching random-guess levels.</em>
</p>

> **Note on absolute accuracy:** The system's Hit@10 (~2.6%) is lower than fully end-to-end trained transformers. Freezing item embeddings prevents joint optimization of user and item representations which is a deliberate privacy constraint. Closing this gap without compromising the privacy mechanism is a direction for future work.

---

## Reproduction

### Requirements

```bash
pip install torch numpy pandas scikit-learn
```

A **Google Colab Pro** environment with GPU runtime is recommended.

### Steps

1. Download [KuaiRec 2.0](https://kuairec.com/) and place `big_matrix.csv` and `item_categories.csv` in the project root.
2. Run notebooks in order:
   - `01_DataPrep_Weights.ipynb` —> generates processed sequences and IPW weights.
   - `02_Cloud_Item_Tower_Training.ipynb` —> trains the Item Tower and exports `frozen_item_embeddings.npy`.
   - `03_SASRec_Training.ipynb` —> trains Baseline and Unlearned SASRec models.
   - `04_Evaluation.ipynb` —> computes Hit Rate@10 and Cross Entropy metrics and generates charts.

---

## Citation

If you reference this work, please cite as:

Aygün, E. (2026). Decoupled Collaborative Filtering for Privacy-Compliant Video Recommendation. https://github.com/fukichime/decoupled--privacy-recommender.git

---


### References

[1] C. Chen et al., "Recommendation Unlearning," in Proc. NeurIPS, 2023.

[2] C. Gao et al., "KuaiRec: A Fully-Observed Dataset and Insights for Evaluating
Recommender Systems," in Proc. CIKM, 2022. https://kuairec.com/

[3] T. Schnabel et al., "Recommendations as Treatments: Debiasing Learning
and Evaluation," in Proc. ICML, 2016.

[4] W.-C. Kang and J. McAuley, "Self-Attentive Sequential Recommendation,"
in Proc. ICDM, 2018.

[5] T. T. Nguyen et al., "A Survey of Machine Unlearning,"
arXiv preprint arXiv:2209.02299, 2022.
