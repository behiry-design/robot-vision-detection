import cv2
import albumentations as A
import os
import glob

OUTPUT_DIR = "/home/behiry/robot_dataset/augmented_manual2"
os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/labels", exist_ok=True)

DATASETS = [
    {"images_dir": "/home/behiry/robot_dataset/sorted_frames/plastic_bottle", "class_id": 0, "num_augs": 8, "prefix": "bottle_vid_"},
    {"images_dir": "/home/behiry/Downloads/plastic_bottle2", "class_id": 0, "num_augs": 8, "prefix": "bottle_man_"},
    {"images_dir": "/home/behiry/robot_dataset/sorted_frames/backpack", "class_id": 1, "num_augs": 8, "prefix": "backpack_vid_"},
    {"images_dir": "/home/behiry/Downloads/backpack", "class_id": 1, "num_augs": 8, "prefix": "backpack_man_"},
    {"images_dir": "/home/behiry/robot_dataset/sorted_frames/perfume", "class_id": 2, "num_augs": 8, "prefix": "perfume_vid_"},
    {"images_dir": "/home/behiry/Downloads/perfume2", "class_id": 2, "num_augs": 8, "prefix": "perfume_man_"},
    {"images_dir": "/home/behiry/robot_dataset/sorted_frames/clock", "class_id": 3, "num_augs": 8, "prefix": "clock_vid_"},
]

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
    A.Affine(scale=(0.85, 1.15), translate_percent=0.05, rotate=(-15, 15), p=0.7),
    A.Blur(blur_limit=3, p=0.3),
    A.HueSaturationValue(p=0.5),
    A.GaussNoise(p=0.2),
],
bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.3))

def read_label(lbl_path):
    bboxes, class_labels = [], []
    if not os.path.exists(lbl_path):
        return bboxes, class_labels
    with open(lbl_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                class_labels.append(int(parts[0]))
                bboxes.append(tuple(map(float, parts[1:])))
    return bboxes, class_labels

def write_label(lbl_path, bboxes, class_labels):
    with open(lbl_path, "w") as f:
        for cls, box in zip(class_labels, bboxes):
            f.write(f"{cls} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n")

total = 0
for ds in DATASETS:
    images = glob.glob(f"{ds['images_dir']}/*.jpg") + glob.glob(f"{ds['images_dir']}/*.jpeg")
    print(f"\n{ds['prefix']}: {len(images)} images x {ds['num_augs']} augs")
    count = 0
    for img_path in images:
        basename = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = f"{ds['images_dir']}/{basename}.txt"
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bboxes, class_labels = read_label(lbl_path)
        if not bboxes:
            continue
        orig_name = f"{ds['prefix']}orig_{basename}.jpg"
        cv2.imwrite(f"{OUTPUT_DIR}/images/{orig_name}", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        write_label(f"{OUTPUT_DIR}/labels/{orig_name.replace('.jpg','.txt')}", bboxes, class_labels)
        for i in range(ds['num_augs']):
            try:
                aug = transform(image=img, bboxes=bboxes, class_labels=class_labels)
                if not aug["bboxes"]:
                    continue
                aug_name = f"{ds['prefix']}{basename}_aug{i}.jpg"
                cv2.imwrite(f"{OUTPUT_DIR}/images/{aug_name}", cv2.cvtColor(aug["image"], cv2.COLOR_RGB2BGR))
                write_label(f"{OUTPUT_DIR}/labels/{aug_name.replace('.jpg','.txt')}", aug["bboxes"], aug["class_labels"])
                count += 1
            except Exception as e:
                pass
    print(f"  Generated: {count}")
    total += count

print(f"\nTotal augmented: {total}")
print(f"Saved to: {OUTPUT_DIR}")
