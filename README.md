# 🐕 Dog Breed Identification using Transfer Learning

![Python](https://img.shields.io/badge/Python-3.11.9-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.12.0-red?style=for-the-badge&logo=pytorch)
![Model](https://img.shields.io/badge/Model-ResNet18-orange?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-green?style=for-the-badge)
![Difficulty](https://img.shields.io/badge/Difficulty-Basic-yellow?style=for-the-badge)

> Identify dog breeds from images using **ResNet18 Transfer Learning** with two-phase fine-tuning across 120 breeds.

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| Name | Stanford Dogs Dataset |
| Total Images | ~20,580 |
| Dog Breeds | 120 |
| Image Size | 224×224 RGB |
| Train Split | 80% |
| Val Split | 20% |

---

## 🏗️ Model Architecture

```
Input (3 × 224 × 224)
        ↓
┌─────────────────────────────────────┐
│  ResNet18 Backbone (Pretrained)     │
│  Weights: ImageNet                  │
│  Layers: Conv → BN → ReLU × 8      │
│  Output: 512-dim feature vector     │
└─────────────────────────────────────┘
        ↓
┌─────────────────────────────────────┐
│  Custom Classification Head         │
│  Dropout(0.5)                       │
│  Linear(512 → 512) → ReLU          │
│  Dropout(0.3)                       │
│  Linear(512 → 120)                  │
└─────────────────────────────────────┘
        ↓
Output (120 Dog Breeds)
```

---

## ⚙️ Training Strategy

### 🔒 Phase 1 — Feature Extraction (5 epochs)
- ResNet18 backbone **frozen**
- Only FC head trained
- Learning Rate: `0.001`

### 🔓 Phase 2 — Fine Tuning (10 epochs)
- Full network **unfrozen**
- Differential learning rates applied
- Backbone LR: `0.0001` | Head LR: `0.001`

| Parameter | Value |
|-----------|-------|
| Total Epochs | 15 |
| Batch Size | 32 |
| Optimizer | Adam |
| Phase 1 Scheduler | StepLR |
| Phase 2 Scheduler | CosineAnnealingLR |
| Loss Function | CrossEntropy + Label Smoothing |

---

## 📈 Results

| Metric | Score |
|--------|-------|
| Best Validation Accuracy | ~75-80% |
| Number of Classes | 120 breeds |

### Training Curves
![Training Curves](training_curves.png)

### Sample Predictions
![Sample Predictions](sample_predictions.png)

---

## ⚠️ Model Weights

> The trained model file `best_dog_breed_model.pth` exceeds GitHub's 100MB file size limit and therefore could not be uploaded to this repository.
> To reproduce the weights, clone this repo and run `dog_breed_identification.py` — the best model will be saved automatically during training.

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/5682003/dog-breed-identification.git
cd dog-breed-identification
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Training
```bash
python dog_breed_identification.py
```

### 4. Run on Google Colab (Recommended)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

---

## 📁 Project Structure

```
dog-breed-identification/
├── dog_breed_identification.py   # Main training script
├── requirements.txt              # Dependencies
├── README.md                     # Project documentation
├── training_curves.png           # Loss & accuracy curves
└── sample_predictions.png        # Sample predictions grid
```

---

## 🎯 Learning Outcomes

- ✅ Transfer Learning with pretrained ResNet18
- ✅ Feature extraction & fine-tuning strategies
- ✅ Differential learning rates
- ✅ Multi-class classification (120 breeds)
- ✅ Data augmentation for small datasets

---

## 📚 References

- [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [PyTorch Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [Reference Implementation](https://github.com/yashrane/Dog-Breed-Identification)
