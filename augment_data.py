import os
import cv2
import numpy as np
from tqdm import tqdm

# Paths to dataset
images_dir = "data/images/train"  # Original training images
augmented_dir = "data/images/train_augmented"  # Directory to save augmented images

# Create output directory if it doesn't exist
os.makedirs(augmented_dir, exist_ok=True)

# Define augmentation functions
def darken_image(image, factor_range=(0.2, 0.7)):
    """Darken the image by multiplying pixel values by a random factor."""
    factor = np.random.uniform(*factor_range)  # Factor between 0.2 and 0.7
    return (image * factor).astype(np.uint8)

def random_occlusion(image, max_holes=8, max_size=(100, 100), fill_type='black'):
    """Apply larger random occlusions to the image."""
    h, w, _ = image.shape
    for _ in range(np.random.randint(1, max_holes + 1)):
        hole_h, hole_w = np.random.randint(50, max_size[0]), np.random.randint(50, max_size[1])
        start_x = np.random.randint(0, max(1, w - hole_w))
        start_y = np.random.randint(0, max(1, h - hole_h))

        if fill_type == 'random':
            image[start_y:start_y + hole_h, start_x:start_x + hole_w] = np.random.randint(0, 255, (hole_h, hole_w, 3), dtype=np.uint8)
        elif fill_type == 'black':
            image[start_y:start_y + hole_h, start_x:start_x + hole_w] = 0
        elif fill_type == 'gray':
            image[start_y:start_y + hole_h, start_x:start_x + hole_w] = 127
        elif fill_type == 'white':
            image[start_y:start_y + hole_h, start_x:start_x + hole_w] = 255
    return image

def apply_augmentations(image):
    """Apply all augmentations."""
    if np.random.rand() > 0.5:
        image = darken_image(image)  # Darken the image
    if np.random.rand() > 0.5:
        image = random_occlusion(image, max_holes=5, max_size=(200, 200))  # Larger occlusions
    return image

# Apply augmentations to all images in the training dataset
print("Applying augmentations to training images...")
for img_name in tqdm(os.listdir(images_dir), desc="Augmenting images"):
    img_path = os.path.join(images_dir, img_name)
    img = cv2.imread(img_path)

    if img is None:
        print(f"Error reading {img_name}, skipping.")
        continue

    # Apply augmentations
    augmented_img = apply_augmentations(img)

    # Save the augmented image
    augmented_path = os.path.join(augmented_dir, f"aug_{img_name}")
    cv2.imwrite(augmented_path, augmented_img)

print("Augmentation completed!")
