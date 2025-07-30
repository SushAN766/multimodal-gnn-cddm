import os
import json
from collections import defaultdict, Counter

# ✅ Configure Paths
DATASET_DIR = r"E:\multimodal-gnn-cddm\dataset"
IMAGES_DIR = os.path.join(DATASET_DIR, "images")

json_files = {
    "llava": os.path.join(DATASET_DIR, "Crop_Disease_train_llava.json"),
    "qwenvl": os.path.join(DATASET_DIR, "Crop_Disease_train_qwenvl.json"),
    "diagnosis": os.path.join(DATASET_DIR, "disease_diagnosis.json"),
    "knowledge": os.path.join(DATASET_DIR, "disease_knowledge.json")
}

# ✅ Ensure all paths exist
if not os.path.exists(DATASET_DIR):
    raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")
if not os.path.exists(IMAGES_DIR):
    raise FileNotFoundError(f"Images folder not found: {IMAGES_DIR}")
for key, path in json_files.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing JSON file: {path}")

# ✅ Extract class from image path
def extract_class_from_path(image_path):
    parts = image_path.split('/')
    for i, part in enumerate(parts):
        if i > 0 and parts[i-1] == "images":
            return part
    return "Unknown"

# ✅ Data containers
all_images = set()
class_counts = Counter()
total_conversations = 0
dataset_info = defaultdict(lambda: {"images": set(), "conversations": 0})
missing_files = set()

# ✅ Process JSON files
def process_json(json_path, dataset_name, is_qwenvl=False):
    global total_conversations
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        if is_qwenvl:
            # Extract image path from <img></img>
            text = " ".join(conv.get("value", "") for conv in item.get("conversations", []))
            img_start = text.find("<img>") + 5
            img_end = text.find("</img>")
            img = text[img_start:img_end] if img_start != -1 and img_end != -1 else ""
        else:
            img = item.get("image", "")

        if not img:
            continue

        cls = extract_class_from_path(img)
        all_images.add(img)
        class_counts[cls] += 1
        dataset_info[dataset_name]["images"].add(img)
        dataset_info[dataset_name]["conversations"] += len(item.get("conversations", [])) if "conversations" in item else 1

        # ✅ Check if image file exists
        img_file = os.path.join(IMAGES_DIR, cls, os.path.basename(img))
        if not os.path.exists(img_file):
            missing_files.add(img)

        total_conversations += len(item.get("conversations", [])) if "conversations" in item else 1

# ✅ Run processing for all JSON files
process_json(json_files["llava"], "LLaVA")
process_json(json_files["qwenvl"], "QwenVL", is_qwenvl=True)
process_json(json_files["diagnosis"], "Diagnosis")
process_json(json_files["knowledge"], "Knowledge")

# ✅ Scan physical image folders
folder_image_counts = {}
for class_folder in os.listdir(IMAGES_DIR):
    class_path = os.path.join(IMAGES_DIR, class_folder)
    if os.path.isdir(class_path):
        num_images = len([f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        folder_image_counts[class_folder] = num_images

# ✅ Calculate totals & percentages
total_json_images = sum(class_counts.values())
total_physical_images = sum(folder_image_counts.values())
json_percentage = {cls: (count / total_json_images) * 100 for cls, count in class_counts.items()}
physical_percentage = {cls: (count / total_physical_images) * 100 for cls, count in folder_image_counts.items()}

# ✅ Output results
print("\n===== DATASET STATISTICS =====")
print(f"✅ Total unique images referenced in JSON: {len(all_images)}")
print(f"✅ Total classes (JSON): {len(class_counts)}")
print(f"✅ Total images in filesystem: {total_physical_images}")
print(f"✅ Total conversations/questions: {total_conversations}\n")

print("--- ✅ Image Count Per Class (From JSON) ---")
for cls, count in class_counts.most_common():
    print(f"{cls}: {count} images ({json_percentage[cls]:.2f}%)")

print("\n--- ✅ Dataset Split Info (From JSON) ---")
for dataset, info in dataset_info.items():
    print(f"{dataset}: {len(info['images'])} unique images, {info['conversations']} conversations")

print("\n--- ✅ Physical Folder Counts ---")
for cls, count in folder_image_counts.items():
    print(f"{cls}: {count} images ({physical_percentage[cls]:.2f}%)")

# ✅ Extra classes present in folder but not in JSON
extra_classes = set(folder_image_counts.keys()) - set(class_counts.keys())
if extra_classes:
    print("\n⚠️ Classes in folder but not in JSON:", extra_classes)

# ✅ Missing images from JSON
if missing_files:
    print("\n⚠️ Missing images referenced in JSON but not found in folder:")
    for img in list(missing_files)[:20]:  # Show first 20 only
        print(img)
    print(f"... and {len(missing_files)-20} more" if len(missing_files) > 20 else "")

print("\n✅ Analysis Complete.")
