# 👁️ Computer Vision Fundamentals (Pure NumPy)

## 📌 Project Overview
As part of my AIML engineering coursework at NIE Mysuru, I am building an intuition for Computer Vision from the ground up. Before relying on heavy libraries like OpenCV or TensorFlow, I built this script to prove that image manipulation is fundamentally just linear algebra applied to 3D matrices.

## ⚙️ The Math Behind the Code
This script generates a gradient image entirely from scratch (using `np.linspace`) and applies transformations using pure mathematical operations:
* **Grayscale:** Achieved via a mathematical dot product, applying human perception weights `[0.299, 0.587, 0.114]` to the RGB channels.
* **Brightness:** Achieved via scalar multiplication and NumPy clipping (e.g., multiplying the matrix by 1.5).
* **Cropping:** Achieved via standard NumPy 3D array slicing.

## 📊 Visual Output
*(Upload your Matplotlib grid screenshot to GitHub and drag it here)*

## 🚀 How to Run Locally
1. Clone this repository.
2. Install dependencies: `pip install numpy matplotlib`
3. Run the script: `python numpy_image_filters.py`
