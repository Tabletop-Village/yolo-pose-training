"""Same synthetic generation as create_images.py, pointed at COCO val2017
backgrounds and a separate output dir, for a genuinely held-out val split
(no background image shared between train/val)."""
import os
import random

from multiprocessing import Pool
import tqdm

import create_images as ci

BACKGROUNDS_DIR = os.path.expanduser("~/yolo-pose-training/backgrounds/val2017")
OUTPUT_DIR = os.path.expanduser("~/yolo-pose-training/dataset/val")

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ci.output = OUTPUT_DIR + "/"
    images = os.listdir(BACKGROUNDS_DIR)
    random.shuffle(images)
    todo = [(os.path.join(BACKGROUNDS_DIR, i), str(c)) for c, i in enumerate(images)]
    p = Pool(6)
    for _ in tqdm.tqdm(p.imap_unordered(ci.loader, todo), total=len(todo)):
        pass
