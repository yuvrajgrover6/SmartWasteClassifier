import os
from PIL import Image
from tqdm import tqdm

# Configuration
CLASS_NAMES = [
    "battery", "biological", "brown-glass", "cardboard", "clothes",
    "green-glass", "metal", "paper", "plastic", "shoes",
    "trash", "white-glass"
]


CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}

SOURCE_DIR = "dataset/split_garbage"
TARGET_DIR = "dataset"

# Output structure
IMAGE_DIRS = {
    "train": os.path.join(TARGET_DIR, "images", "train"),
    "val": os.path.join(TARGET_DIR, "images", "val"),
}
LABEL_DIRS = {
    "train": os.path.join(TARGET_DIR, "labels", "train"),
    "val": os.path.join(TARGET_DIR, "labels", "val"),
}

# Create target folders
for folder in list(IMAGE_DIRS.values()) + list(LABEL_DIRS.values()):
    os.makedirs(folder, exist_ok=True)


def convert_and_copy(split):
    print(f"🔄 Converting {split} data...")
    split_dir = os.path.join(SOURCE_DIR, split)

    for class_name in os.listdir(split_dir):
        class_path = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        label_id = CLASS_TO_ID.get(class_name)
        if label_id is None:
            print(f"⚠️ Unknown class: {class_name}")
            continue

        for img_name in tqdm(os.listdir(class_path), desc=f"{split}/{class_name}"):
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(class_path, img_name)
            image = Image.open(img_path)
            w, h = image.size

            # Central bounding box: full image
            x_center = 0.5
            y_center = 0.5
            width = 1.0
            height = 1.0

            # Save image
            out_img_path = os.path.join(IMAGE_DIRS[split], img_name)
            if image.mode != "RGB":
                image = image.convert("RGB")
            image.save(out_img_path, "JPEG")


            # Save label
            label_path = os.path.join(LABEL_DIRS[split], os.path.splitext(img_name)[0] + ".txt")
            with open(label_path, "w") as f:
                f.write(f"{label_id} {x_center} {y_center} {width} {height}\n")


# Run conversion
convert_and_copy("train")
convert_and_copy("val")
print("✅ Conversion complete! YOLOv8 format ready.")
