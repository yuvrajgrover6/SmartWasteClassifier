import os
import shutil
import random
from tqdm import tqdm

def split_dataset(source_dir, output_dir, val_split=0.2):
    classes = os.listdir(source_dir)
    for class_name in classes:
        class_path = os.path.join(source_dir, class_name)
        images = os.listdir(class_path)
        random.shuffle(images)

        val_size = int(len(images) * val_split)
        val_images = images[:val_size]
        train_images = images[val_size:]

        for split, split_images in zip(["train", "val"], [train_images, val_images]):
            split_dir = os.path.join(output_dir, split, class_name)
            os.makedirs(split_dir, exist_ok=True)
            for img in tqdm(split_images, desc=f"Copying {split}/{class_name}"):
                src = os.path.join(class_path, img)
                dst = os.path.join(split_dir, img)
                shutil.copy2(src, dst)

if __name__ == "__main__":
    src = "./dataset/garbage_classification"
    out = "./dataset/split_garbage"
    split_dataset(src, out)
