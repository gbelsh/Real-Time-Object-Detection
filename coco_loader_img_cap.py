import os
import requests
from random import sample
from pycocotools.coco import COCO
from tqdm import tqdm

# Paths to COCO annotation files
train_annotations_file = 'annotations/instances_train2017.json'
train_captions_file = 'annotations/captions_train2017.json'
val_annotations_file = 'annotations/instances_val2017.json'
val_captions_file = 'annotations/captions_val2017.json'
test_annotations_file = 'annotations/image_info_test2017.json'  # Test set metadata

# Output directories for images and labels
output_image_dir = 'data/images'
output_label_dir = 'data/labels'

# Check if directories exist; create them if not
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(output_image_dir, split), exist_ok=True)
    if split != 'test':  # Labels directory only for train/val
        os.makedirs(os.path.join(output_label_dir, split), exist_ok=True)

def download_human_images_and_labels(coco, captions_coco, output_image_dir, output_label_dir, split, num_images):
    # Get the category ID for "person" (only applicable for train/val)
    person_cat_id = coco.getCatIds(catNms=['person'])[0] if captions_coco else None

    # Get all image IDs (filter for humans if captions_coco is provided)
    img_ids = coco.getImgIds(catIds=[person_cat_id]) if captions_coco else coco.getImgIds()

    # Randomly sample the required number of images
    selected_img_ids = sample(img_ids, num_images)

    # Extract captions for the selected images (only for train/val)
    captions_by_image_id = {}
    if captions_coco:
        for annotation in captions_coco.dataset['annotations']:
            image_id = annotation['image_id']
            if image_id not in captions_by_image_id:
                captions_by_image_id[image_id] = []
            captions_by_image_id[image_id].append(annotation['caption'])

    # Process each selected image
    for img_id in tqdm(selected_img_ids, desc=f"Downloading {split} images"):
        # Load image metadata
        img_info = coco.loadImgs(img_id)[0]
        img_url = img_info['coco_url']
        img_name = img_info['file_name']

        # Download the image
        response = requests.get(img_url)
        if response.status_code == 200:
            # Save the image to the output directory
            dest_img_path = os.path.join(output_image_dir, split, img_name)
            with open(dest_img_path, 'wb') as f:
                f.write(response.content)
        else:
            print(f"Failed to download image: {img_url}")
            continue

        # Save captions to a .txt file (only for train/val)
        if captions_coco:
            label_name = os.path.splitext(img_name)[0] + '.txt'
            label_path = os.path.join(output_label_dir, split, label_name)
            with open(label_path, 'w') as f:
                captions = captions_by_image_id.get(img_id, ["No captions available"])
                f.write("\n".join(captions))

# Initialize COCO API for training, validation, and testing
train_coco = COCO(train_annotations_file)
train_captions_coco = COCO(train_captions_file)
val_coco = COCO(val_annotations_file)
val_captions_coco = COCO(val_captions_file)
test_coco = COCO(test_annotations_file)

# Download and process training images and captions containing humans
print("Processing training data...")
download_human_images_and_labels(
    train_coco, 
    train_captions_coco, 
    output_image_dir, 
    output_label_dir, 
    'train', 
    num_images=500
)

# Download and process validation images and captions containing humans
print("Processing validation data...")
download_human_images_and_labels(
    val_coco, 
    val_captions_coco, 
    output_image_dir, 
    output_label_dir, 
    'val', 
    num_images=100
)

# Download and process test images (no captions available)
print("Processing test data...")
download_human_images_and_labels(
    test_coco, 
    None,  # No captions for test set
    output_image_dir, 
    output_label_dir, 
    'test', 
    num_images=100
)

print("Image downloading and processing completed successfully!")
