# Improving Feature Representations for Few-Shot Classification of Neglected Tropical Skin Diseases on Underrepresented Skin Tones
## Overview

This repository contains the implementation and experiments for a Master's research project focused on improving visual feature representations for few-shot recognition of Neglected Tropical Diseases (NTDs) on dark skin tones. The study investigates how sequential domain adaptation and supervised contrastive learning improve representation quality under:
1) Domain shift (ImageNet → Dermatology)
2) Data scarcity (limited labeled NTD samples)
3) Underrepresentation of darker skin tones
The framework emphasizes data-efficient learning under realistic clinical constraints.
## Methodology Overview
<img width="1132" height="755" alt="image" src="https://github.com/user-attachments/assets/e8934663-af63-4b86-b66e-23ebbce47cb4" />


### Stage 1 – Feature Extractor Evaluation
- Models initialized with ImageNet weights
- Frozen backbone + linear classifier
- Compared using Accuracy and F1-score
- Top models selected (DenseNet161, ViT-B/16)
### Stage 2 – Domain Adaptation
- Partial fine-tuning of higher layers
- Focused on Fitzpatrick IV–VI skin tones
- Mitigates domain shift
### Stage 3 – Supervised Contrastive Learning
- Refines embedding structure
- Increases inter-class margins
- Improves intra-class compactness
### Stage 4 – Few-Shot Evaluation
Fixed N-way K-shot setups:
- 8-way 5-shot
- 5-way 5-shot
- 3-way 5-shot
- 3-way 2-shot
- 3-way 1-shot
NB:
Nearest centroid classification in feature space
Same support sets across stages for controlled comparison
## Evaluation Metrics
1) Accuracy
2) Weighted F1-score
3) Macro F1-score
4) Nearest Centroid Cosine Distance
5) Mean Pairwise Cosine Distance
## Dataset Sources
1) Fitzpatrick17k (Groh et al., 2021)
2) SD-198
3) Curated NTD subsets

Note: Access permissions were obtained where required.
