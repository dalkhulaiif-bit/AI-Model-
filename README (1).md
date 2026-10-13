# Facial Emotion Recognition Using Deep Learning

**SDAIA Academy | Computer Vision Project**

## Project Overview

This project develops a deep-learning system to recognize facial expressions from images and, as a future extension, live video streams. It explores a custom Convolutional Neural Network (CNN) and transfer-learning architectures, **ResNet18** and **EfficientNet-B1**, to classify facial expressions into seven categories. The project is developed in **Google Colab** using **Python** and **PyTorch**.

> **Project status:** Image classification and model comparison are the documented core workflow. Live-stream inference, optimization, and deployment are included as proposed extensions unless implemented and validated separately.

## 1. Problem Description

Recognizing facial expressions from images is a challenging computer vision task because of differences in lighting, head pose, occlusion, image quality, and variation between people. This project aims to:

- Build a model that predicts one of seven facial-expression categories from an input image.
- Compare a CNN trained from scratch with pretrained architectures.
- Evaluate performance on data not used for training.
- Investigate how the best-performing model could support real-time predictions.

**Note:** A facial-expression prediction is not a reliable determination of a person's internal emotional state. Outputs should be interpreted as visual-expression classifications, not psychological assessments.

## 2. Dataset & Model Used

### Dataset

The project uses a seven-class facial-expression image dataset organized into training and test splits (consistent with the common FER2013 setup). The dataset source/license should be added before publication.

| Split | Number of images | Notes |
| --- | ---: | --- |
| Original training split | 28,709 | Split into training and validation subsets |
| Training subset | 22,967 | Used for model training |
| Validation subset | 5,742 | Used for tuning and model selection |
| Test split | 7,178 | Held out for final evaluation |

**Expression classes:** `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`.

### Models

| Model | Approach | Input dimensions |
| --- | --- | --- |
| Custom CNN | Trained for seven-class facial-expression classification | Confirm in notebook |
| ResNet18 | Pretrained backbone adapted for seven classes | 224 × 224 (from project loader) |
| EfficientNet-B1 | Pretrained backbone adapted for seven classes | 240 × 240 (from project loader) |

For pretrained ImageNet models, image normalization uses mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]` when configured as recorded in the project. Each model's inference preprocessing must match its training preprocessing.

## 3. Workflow / Architecture

```text
Facial-expression dataset
          |
          v
Data loading and label mapping
          |
          v
Train / validation / test separation
          |
          v
Model-specific preprocessing
(resizing, tensor conversion, normalization)
          |
          v
+-------------+-------------+-----------------+
| Custom CNN  | ResNet18    | EfficientNet-B1 |
+-------------+-------------+-----------------+
          |         |              |
          +---------+--------------+
                    |
                    v
          Training and validation
                    |
                    v
             Test evaluation
                    |
                    v
         Model performance comparison
                    |
                    v
        Best model for inference
                    |
                    v
    [Planned] Live video / deployment
```

Key stages:

1. Load and inspect the labeled facial-expression images.
2. Create separate training, validation, and test datasets.
3. Apply preprocessing appropriate to each model.
4. Train or fine-tune the CNN, ResNet18, and EfficientNet-B1 models.
5. Track training and validation metrics.
6. Evaluate trained models on the same held-out test examples.
7. Compare prediction quality and, where measured, inference speed.
8. Optionally integrate the selected model with a webcam or video-streaming pipeline.

## 4. Results & Evaluation

All three models were trained and evaluated on the FER2013 facial emotion recognition dataset, using **7,178 test images** across seven emotion categories.

### Model Performance Comparison

| Model | Test Accuracy | Macro F1-score | Inference Latency | Notes |
| --- | --- | --- | --- | --- |
| Custom CNN | 56.45% | 0.4500 | Not measured | Trained from scratch |
| ResNet18 | 46.31% | 0.3811 | Not measured | Pretrained model with frozen backbone |
| **EfficientNet-B1** | **58.82%** | **0.5395** | Not measured | Pretrained model with fine-tuning |

### Key Findings

- **EfficientNet-B1 achieved the highest test accuracy (58.82%)** and Macro F1-score (0.5395).
- Custom CNN achieved 56.45% accuracy, outperforming ResNet18.
- ResNet18 achieved the lowest accuracy (46.31%) with feature extraction only.
- EfficientNet-B1 improved test accuracy by **2.37 percentage points** compared with Custom CNN.
- The `happy` category achieved strong classification performance, while `fear` and `disgust` remained challenging.
- Inference latency has not yet been measured and will be added after benchmarking.

### Conclusion

EfficientNet-B1 with fine-tuning demonstrated the best overall classification performance among the three evaluated models. These results suggest that fine-tuning a pretrained architecture can improve facial emotion recognition performance on FER2013 compared with the training approaches used for the other two models.

## 5. Technologies Used

- **Python** — project programming language.
- **Google Colab / Jupyter Notebook** — development and experimentation.
- **PyTorch** and **torchvision** — neural networks, datasets, transforms, and inference.
- **CUDA-enabled GPU** — accelerated training and inference when available.
- **NumPy**, **Matplotlib** — numerical operations and visualizations (if used by notebook).
- **scikit-learn** — classification metrics and confusion matrix (if used by notebook).
- **OpenCV** — proposed webcam/video-stream capture for real-time inference.

## 6. How to Run the Project

### Prerequisites

- A Google account with access to Google Colab, or a local Python environment.
- Access to the facial-expression dataset used for training.
- The project notebook(s) and, for inference, the trained model checkpoint(s).
- Optional: a GPU runtime to accelerate training.

### Google Colab steps

1. Open the project `.ipynb` notebook in [Google Colab](https://colab.research.google.com/).
2. If using a GPU, select **Runtime → Change runtime type → GPU**.
3. Mount Google Drive if your dataset or saved checkpoints are located there.
4. Update dataset and checkpoint paths to match your Colab or Drive directories.
5. Run the dependency/import cells and dataset loading/preprocessing cells in order.
6. Train the models if checkpoints do not already exist; otherwise load the saved model weights.
7. Run validation and test evaluation cells to produce metrics and graphs.
8. For inference, upload an image and run the prediction cells using the matching model preprocessing.

**Optional package installation**, depending on what the notebook imports:

```bash
pip install torch torchvision numpy matplotlib scikit-learn opencv-python
```

> Package versions, exact notebook name, dataset path, and checkpoint filenames must be taken from your actual environment. A restart of the Colab runtime clears in-memory models and variables; rerun setup cells or reload saved checkpoints afterward.

### Live streaming and deployment

Real-time webcam inference requires additional integration beyond a notebook image classifier: video capture, face detection/cropping, consistent preprocessing, model inference, and prediction overlay. Browser webcam access in Colab may require a JavaScript bridge; a local OpenCV application is often more straightforward. Do not assume live-streaming functionality is complete until tested.

## 7. Future Improvements

- Add and test real-time webcam/video-stream prediction.
- Reduce inference latency using optimized preprocessing, batching when appropriate, and hardware-aware inference.
- Evaluate model export via **ONNX** or another deployment-compatible format.
- Build a small interactive interface (for example, **Streamlit** or **Gradio**).
- Evaluate performance under varied lighting, camera angles, occlusions, and demographic groups.
- Analyze per-class confusion and address class imbalance with suitable training and evaluation methods.
- Add reproducible configuration, random seeds, pinned dependencies, checkpoint versioning, and automated tests.
- Include privacy safeguards, especially consent, data minimization, and retention rules for any camera-based deployment.

## 8. SDAIA Academy

This project is presented as part of learning and practical application at **SDAIA Academy**, focusing on deep learning, computer vision, transfer learning, and model evaluation. It demonstrates an end-to-end machine-learning workflow—from dataset preparation and architecture selection to testing and potential real-world integration.

**Project affiliation:** SDAIA Academy — educational project. This README does not imply formal approval, certification, or endorsement by SDAIA.

---

### Before submission

Confirm the **CNN input dimensions** and the dataset's download source/license; benchmark **inference latency**; and specify which live-streaming or deployment features have actually been implemented.

### Useful references

- [SDAIA official website](https://sdaia.gov.sa/)
- [PyTorch documentation](https://docs.pytorch.org/docs/stable/index.html)
- [torchvision model documentation](https://docs.pytorch.org/vision/stable/models.html)
- [Google Colab](https://colab.research.google.com/)
- [scikit-learn classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)
-[one of the prerequisites of SDAIA ACADEMY]