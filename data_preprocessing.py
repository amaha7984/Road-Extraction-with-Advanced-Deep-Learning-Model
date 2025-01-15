import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

def load_dataset(image_dir, mask_dir, size=512, max_images=None):
    """Loads and processes the dataset of images and masks."""
    image_dataset, mask_dataset = [], []
    
    image_subdirs = sorted([d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))])
    mask_subdirs = sorted([d for d in os.listdir(mask_dir) if os.path.isdir(os.path.join(mask_dir, d))])

    # Optionally limit the dataset size
    if max_images:
        image_subdirs = image_subdirs[:max_images]
        mask_subdirs = mask_subdirs[:max_images]

    for subdir in image_subdirs:
        image_files = sorted(os.listdir(os.path.join(image_dir, subdir)))
        mask_files = sorted(os.listdir(os.path.join(mask_dir, subdir)))

        for img_file, mask_file in zip(image_files, mask_files):
            img_path = os.path.join(image_dir, subdir, img_file)
            mask_path = os.path.join(mask_dir, subdir, mask_file)

            image = Image.open(img_path).convert('RGB').resize((size, size))
            mask = Image.open(mask_path).convert('L').resize((size, size))

            image_dataset.append(np.array(image) / 255.0)
            mask_dataset.append(np.array(mask) / 255.0)

    return np.array(image_dataset), np.expand_dims(np.array(mask_dataset), axis=-1)

def split_dataset(image_dataset, mask_dataset, val_size=0.1, test_size=0.2, random_state=0):
    """Splits the dataset into train, validation, and test sets."""
    # First splitting into train+val and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        image_dataset, mask_dataset, test_size=test_size, random_state=random_state
    )
    # Then splitting train+val into train and val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size / (1 - test_size), random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
