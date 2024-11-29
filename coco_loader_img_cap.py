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

# Output directories for images and labels
output_image_dir = 'data/images'
output_label_dir = 'data/labels'

# Check if they exist, if not make em
for split in ['train', 'val']:
    os.makedirs(os.path.join(output_image_dir, split), exist_ok=True)
    os.makedirs(os.path.join(output_label_dir, split), exist_ok=True)

def download_human_images_and_labels(coco, captions_coco, output_image_dir, output_label_dir, split, num_images):
    # Get only people from COCO
    person_cat_id = coco.getCatIds(catNms=['person'])[0]

    # Gather Image ID's of those people so that they can be matched to captions
    img_ids = coco.getImgIds(catIds=[person_cat_id])

    # Randomly sample the dataset, according to pretermined dataset size
    selected_img_ids = sample(img_ids, num_images)
    
    # Extract captions for the selected images
    captions_by_image_id = {}
    for annotation in captions_coco.dataset['annotations']:
        image_id = annotation['image_id']
        if image_id not in captions_by_image_id:
            captions_by_image_id[image_id] = []
        captions_by_image_id[image_id].append(annotation['caption'])
    
    # Process each selected image
    for img_id in tqdm(selected_img_ids, desc=f"Downloading {split} human images"):
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
        
        # Save the corresponding captions to a .txt file
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(output_label_dir, split, label_name)
        with open(label_path, 'w') as f:
            captions = captions_by_image_id.get(img_id, ["No captions available"])
            f.write("\n".join(captions))

# Initialize COCO API for training and validation
train_coco = COCO(train_annotations_file)
train_captions_coco = COCO(train_captions_file)
val_coco = COCO(val_annotations_file)
val_captions_coco = COCO(val_captions_file)

# Download and process 500 training images and captions containing humans
download_human_images_and_labels(
    train_coco, 
    train_captions_coco, 
    output_image_dir, 
    output_label_dir, 
    'train', 
    num_images=500
)

# Download and process 100 validation images and captions containing humans
download_human_images_and_labels(
    val_coco, 
    val_captions_coco, 
    output_image_dir, 
    output_label_dir, 
    'val', 
    num_images=100
)

print("Human image and caption successfully completed")
