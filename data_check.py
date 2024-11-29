import os
import cv2

def check_augmented_consistency(images_dir, augmented_dir):
    """Verify augmented images match the original training images."""
    print("Checking augmented dataset consistency...")

    # Get original and augmented filenames
    original_images = set(
        os.path.splitext(f)[0].strip() for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg'))
    )
    augmented_images = set(
        os.path.splitext(f)[0].replace('aug_', '').strip() for f in os.listdir(augmented_dir) if f.lower().endswith(('.jpg', '.jpeg'))
    )

    # Check for missing and extra images
    missing_augmented = original_images - augmented_images
    extra_augmented = augmented_images - original_images

    # Report findings
    if missing_augmented:
        print(f"Missing augmented images: {missing_augmented}")
    else:
        print("No missing augmented images.")

    if extra_augmented:
        print(f"Extra augmented images: {extra_augmented}")
    else:
        print("No extra augmented images.")

    if not missing_augmented and not extra_augmented:
        print("Augmented dataset is consistent and corruption-free!")
    else:
        print("Inconsistencies detected in the augmented dataset.")

# Paths to original and augmented directories
train_images_dir = "data/images/train"
augmented_images_dir = "data/images/train_augmented"

# Run augmented dataset consistency check
check_augmented_consistency(train_images_dir, augmented_images_dir)