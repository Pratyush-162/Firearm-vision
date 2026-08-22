import os
import shutil
import cv2
import uuid

def setup_dirs(base_path):
    for split in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(base_path, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(base_path, split, 'labels'), exist_ok=True)

def process_dataset(src_dir, dest_dir, prefix, class_map=None, filter_classes=None):
    if not os.path.exists(src_dir):
        print(f"Directory not found: {src_dir}")
        return

    for split in ['train', 'valid', 'test']:
        img_src = os.path.join(src_dir, split, 'images')
        lbl_src = os.path.join(src_dir, split, 'labels')
        
        if not os.path.exists(img_src) or not os.path.exists(lbl_src):
            continue

        for filename in os.listdir(img_src):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            img_path = os.path.join(img_src, filename)
            lbl_filename = os.path.splitext(filename)[0] + '.txt'
            lbl_path = os.path.join(lbl_src, lbl_filename)
            
            new_filename = f"{prefix}_{filename}"
            new_lbl_filename = f"{prefix}_{lbl_filename}"
            
            dest_img_path = os.path.join(dest_dir, split, 'images', new_filename)
            dest_lbl_path = os.path.join(dest_dir, split, 'labels', new_lbl_filename)

            if not os.path.exists(lbl_path):
                continue
                
            # Process label file
            valid_lines = []
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        if filter_classes and cls_id not in filter_classes:
                            continue
                        if class_map and cls_id in class_map:
                            cls_id = class_map[cls_id]
                        
                        valid_lines.append(f"{cls_id} {' '.join(parts[1:])}\n")
                        
            shutil.copy2(img_path, dest_img_path)
            with open(dest_lbl_path, 'w') as f:
                f.writelines(valid_lines)
                
def extract_backgrounds(video_path, dest_dir, num_frames=100):
    if not os.path.exists(video_path):
        print("Video not found.")
        return
        
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        return
        
    step = max(1, total_frames // num_frames)
    
    count = 0
    frame_idx = 0
    while cap.isOpened() and count < num_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
            
        uid = uuid.uuid4().hex[:8]
        filename = f"bg_{uid}.jpg"
        
        img_dest = os.path.join(dest_dir, 'train', 'images', filename)
        lbl_dest = os.path.join(dest_dir, 'train', 'labels', f"bg_{uid}.txt")
        
        cv2.imwrite(img_dest, frame)
        with open(lbl_dest, 'w') as f:
            pass # Empty file for background
            
        count += 1
        frame_idx += step
        
    cap.release()
    print(f"Extracted {count} background frames from video.")

def create_yaml(dest_dir):
    yaml_content = f"""path: {os.path.abspath(dest_dir)}
train: train/images
val: valid/images
test: test/images

nc: 1
names: ['Gun']
"""
    with open(os.path.join(dest_dir, 'super_dataset.yaml'), 'w') as f:
        f.write(yaml_content)

if __name__ == "__main__":
    dest_dir = "/Users/pratyushbharadwaj/Documents/firearm_vision/data/merged"
    setup_dirs(dest_dir)
    
    print("Processing Lab1...")
    process_dataset("/Users/pratyushbharadwaj/Documents/firearm_vision/data/Lab1", 
                    dest_dir, "lab1", class_map={0: 0}, filter_classes={0})
                    
    print("Processing weapon-detection...")
    process_dataset("/Users/pratyushbharadwaj/Documents/firearm_vision/data/weapon-detection", 
                    dest_dir, "weap", class_map={0: 0, 1: 0}, filter_classes={0, 1})
                    
    print("Extracting backgrounds...")
    extract_backgrounds("/Users/pratyushbharadwaj/Documents/firearm_vision/test_video.mp4", dest_dir)
    
    create_yaml(dest_dir)
    print("Done merging datasets.")
