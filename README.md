

# Pedestrian Attention Recognition with Deep Learning

A CNN-based computer vision system that classifies pedestrians as **attentive** or **distracted** from images.  
Built in **PyTorch** (trained on **Google Colab**) with a focus on **high recall** for safety-critical use.

**Highlights**
- **Test (held-out):** 87.0% accuracy · 86.46% recall  
- **Unseen split (20% of dataset, never touched during dev):** 86.56% accuracy · 85.63% recall  
- **Throughout:** reduced worst-case training time **4+ hours → ~50 minutes** via input resizing (32×32), pipeline optimizations, and architecture tweaks  
- Balanced ~**25k** training images sampled from **Gaze360** to mitigate class skew

## 🔗 Project Resources
  
📓 [Notebook](notebook/final_training.ipynb)

---

## Goals
- Detect pedestrian attentiveness in (near) real time  
- Prioritize **recall** of distracted pedestrians (minimize false negatives)  
- Build a robust preprocessing + training pipeline resilient to **class imbalance**

---

## Dataset & Labeling
**Source:** [Gaze360](https://gaze360.csail.mit.edu/) (rich head/gaze annotations across varied subjects, poses, and environments) 

**Labeling (angle-based):**
- Extract 3D gaze vectors and compare to the camera forward axis
- Compute angular difference (degrees)
- **Attentive (0)** if ≤ **20°**; **Distracted (1)** otherwise

**Preprocessing:**
- Grayscale conversion; **32×32** resolution
- Class-balanced sampling (downsample majority)
- CSV index of paths, angles, and labels

**Samples:**

![Sample Image](./images/SamplePreProcessedData.png "Sample Image Data and Labels")
---

## Model
**Architecture**
- 3× Conv → ReLU → (MaxPool after conv1 & conv2)  
- **Skip connection** (stability) + **two auxiliary outputs** (deeper supervision)  
- Dropout(0.5) before the classifier  
- Input: **32×32** grayscale

**Training**
- Loss: `BCEWithLogitsLoss` (+ aux losses)  
- Optimizer: SGD (momentum 0.9)  
- Typical batch size: 64; LR: 1e-3  
- Environment: preprocessing in local VS Code; training in **Google Colab**

**Baseline (for context)**
- RBF-SVM on simple head-direction features (left/right) achieved ~99% on its easier task—useful sanity check, but not comparable to full-image attentiveness.

---

## Results
**Held-out Test**
- Accuracy **0.870** · Recall **0.8646**

**Unseen Split (20% never used in tuning)**
- Accuracy **0.8656** · Recall **0.8563**

**Why this model?**
- We selected the final architecture because it **recalls distracted pedestrians better** than alternatives with slightly higher accuracy—aligning with safety priorities.

**What I got out of the experience**
- Developed my skills in python, its wide selection of libraries (PyTorch, NumPy, Pandas, MatPlot Lib).
- Learned about model architecture, data preprocessing, fine tuning and presentation of ideas.
- Machine learning architectures are a significant component of Robotics as it provides tools to understand concepts such as computer vision.
