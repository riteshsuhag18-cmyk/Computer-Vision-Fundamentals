import numpy as np
import matplotlib.pyplot as plt

def create_and_transform_image():
    """Generates a gradient image and applies Numpy-based transformations."""
    
    # ==========================================
    # 1. CREATE GRADIENT IMAGE FROM SCRATCH
    # ==========================================
    height, width = 100, 150
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Red increases left to right
    image[:, :, 0] = np.linspace(0, 255, width, dtype=np.uint8)
    
    # Green increases top to bottom (reshaped for broadcasting)
    image[:, :, 1] = np.linspace(0, 255, height, dtype=np.uint8).reshape(-1, 1)
    
    # Blue remains constant
    image[:, :, 2] = 100

    print("--- Image Specifications ---")
    print(f"Shape: {image.shape}")
    print(f"Height: {image.shape[0]} pixels")
    print(f"Width: {image.shape[1]} pixels")
    print(f"Channels: {image.shape[2]} (R,G,B)")
    print(f"Total Pixels: {image.size:,}\n")

    # ==========================================
    # 2. IMAGE OPERATIONS USING NUMPY MATH
    # ==========================================
    
    # Grayscale: Dot product with human perception weights
    weights = np.array([0.299, 0.587, 0.114])
    grayscale = np.dot(image.astype(float), weights).astype(np.uint8)

    # Brightness adjustment: Scalar multiplication & Clipping
    brighter = np.clip(image.astype(int) * 1.5, 0, 255).astype(np.uint8)
    darker = (image * 0.5).astype(np.uint8)

    # Cropping: Array slicing
    cropped = image[20:80, 30:120]

    # ==========================================
    # 3. VISUALIZE WITH MATPLOTLIB
    # ==========================================
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original")

    axes[0, 1].imshow(grayscale, cmap='gray')
    axes[0, 1].set_title("Grayscale (Dot Product)")

    axes[0, 2].imshow(brighter)
    axes[0, 2].set_title("Brighter (×1.5)")

    axes[1, 0].imshow(darker)
    axes[1, 0].set_title("Darker (×0.5)")

    axes[1, 1].imshow(cropped)
    axes[1, 1].set_title(f"Cropped {cropped.shape[:2]}")

    axes[1, 2].imshow(image[:, :, 0], cmap='Reds')
    axes[1, 2].set_title("Red Channel Extraction")

    # Remove axes for a cleaner look
    for ax in axes.flat:
        ax.axis('off')

    plt.suptitle('Image Manipulation - Powered Purely by Numpy Math', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    create_and_transform_image()