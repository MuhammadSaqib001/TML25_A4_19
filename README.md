# Model Explainability (TML Assignment 4)

This repository contains a comprehensive analysis of deep learning model interpretability using multiple techniques including **CLIP-dissect**, **Grad-CAM**, **ScoreCAM**, **AblationCAM**, and **LIME**. It showcases results for image classification models trained on different datasets and evaluated on diverse image samples.

---

## 📁 Repository Structure

```markdown
├── CLIP-dissect/ # Neuron dissection analysis using CLIP
├── Grad-CAM/ # Grad-CAM, ScoreCAM, and AblationCAM visualizations
├── LIME/ # LIME-based explanations with superpixel outlines
├── images/ # Supporting input/output images
├── explain_params.pkl # Saved LIME/CLIP explanation data
├── pickle_submit.py # Script to load/run models and explanation logic
├── Report A4 TML.pdf # Detailed PDF report with methods, results, and findings
└── README.md # Project summary and usage instructions
```

---

## ✅ Tasks Covered

### Task 1: Network Dissection with CLIP
- **Goal**: Identify semantically interpretable neurons in the final layers of ResNet18 trained on **ImageNet** and **Places365**.
- **Tool**: [CLIP-dissect](https://github.com/Trustworthy-ML-Lab/CLIP-dissect)
- **Highlights**:
  - Places365 shows higher diversity in learned concepts.
  - ImageNet model neurons detect natural features (fur, feathers), while Places365 focuses on man-made structures (windows, furniture).
  - Only 38% concept overlap between the two models.

### Task 2: Grad-CAM, AblationCAM & ScoreCAM
- **Goal**: Visualize salient regions for predictions using ResNet50.
- **Tool**: [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)
- **CAMs Implemented**: Grad-CAM, Score-CAM, AblationCAM
- **Findings**:
  - Strong object-focused attention for clean images (e.g., tiger shark, flamingo).
  - ScoreCAM provided the sharpest, most accurate heatmaps.

### Task 3: Local Explanations with LIME
- **Goal**: Use LIME to provide localized, human-interpretable explanations.
- **Output**: Superpixel-based yellow outlines of most influential regions.
- **Observation**:
  - Excellent segmentation for distinct objects (e.g., goldfish, flamingo).
  - Performance degrades slightly in cluttered backgrounds.

### Task 4: LIME vs CAM Comparison
- **Insight**:
  - High agreement for simple images.
  - Disagreement in complex scenes reveals model ambiguity or dataset bias.
  - LIME is model-agnostic; CAMs are class-specific and model-aware.

---

## Report

For a complete description of experiments, methodologies, and comparative findings, refer to [`Report A4 TML.pdf`](./Report%20A4%20TML.pdf).

---

Install dependencies:
```bash
pip install torch torchvision matplotlib scikit-learn lime git+https://github.com/openai/CLIP.git
pip install git+https://github.com/jacobgil/pytorch-grad-cam.git
```

## Acknowledgements
1. CLIP-dissect: Trustworthy-ML-Lab
2. Grad-CAM Tools: Jacob Gildenblat
3. LIME: Ribeiro et al., 2016







