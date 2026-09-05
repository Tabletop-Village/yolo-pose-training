"""
Generates the sample images used in the README: the real test photo with
the pose model's live detections drawn on it, and a small grid of labeled
training examples (ground-truth keypoints from the synthetic composite
dataset).
"""
import random

import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = '/home/user/projects/card-scanner-cuda/runs/pose/train-3/weights/best.pt'
TEST_IMAGE = '/home/user/siglip-scanner/model_comparison/test_image.jpg'
TRAIN_DIR = '/home/user/yolo-pose-training/dataset/train'
OUT_DIR = '/home/user/yolo-pose-training/docs'

CORNER_COLOR = (60, 220, 60)     # BGR
QUAD_COLOR = (60, 220, 60)
BOX_COLOR = (180, 180, 180)


def order_points(pts):
    rect = np.zeros((4, 2), dtype='float32')
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def draw_detection(image, box_xyxy, pts4, thickness_scale=1.0):
    x1, y1, x2, y2 = map(int, box_xyxy)
    t = max(1, int(2 * thickness_scale))
    cv2.rectangle(image, (x1, y1), (x2, y2), BOX_COLOR, t)

    rect = order_points(np.asarray(pts4, dtype='float32'))
    quad = rect.astype(int)
    cv2.polylines(image, [quad], isClosed=True, color=QUAD_COLOR, thickness=max(2, int(4 * thickness_scale)))
    for (x, y) in quad:
        cv2.circle(image, (x, y), max(3, int(8 * thickness_scale)), CORNER_COLOR, -1)
    return image


def make_test_image_sample():
    model = YOLO(MODEL_PATH)
    image = cv2.imread(TEST_IMAGE)
    results = model(image, device='cuda', verbose=False)
    annotated = image.copy()
    n = 0
    for result in results:
        if not result.boxes:
            continue
        for i, box in enumerate(result.boxes):
            if result.keypoints is None:
                continue
            pts4 = result.keypoints[i].xy[0].cpu().numpy()
            if pts4.shape[0] != 4:
                continue
            annotated = draw_detection(annotated, box.xyxy[0].tolist(), pts4, thickness_scale=2.0)
            n += 1
    # Downscale for a reasonable file size / README embed
    h, w = annotated.shape[:2]
    scale = 1400 / max(h, w)
    annotated = cv2.resize(annotated, (int(w * scale), int(h * scale)))
    out_path = f'{OUT_DIR}/sample_detection.jpg'
    cv2.imwrite(out_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f'{n} cards detected -> {out_path}')


def make_training_gallery(rows=2, cols=4, seed=7, cell_size=260):
    random.seed(seed)
    import os
    all_ids = [f[:-4] for f in os.listdir(TRAIN_DIR) if f.endswith('.jpg')]
    # Prefer examples with at least 2 cards, for a more illustrative gallery
    candidates = []
    for img_id in random.sample(all_ids, min(400, len(all_ids))):
        label_path = f'{TRAIN_DIR}/{img_id}.txt'
        with open(label_path) as f:
            lines = [l for l in f.read().strip().splitlines() if l]
        if len(lines) >= 2:
            candidates.append(img_id)
        if len(candidates) >= rows * cols:
            break

    cells = []
    for img_id in candidates[:rows * cols]:
        img = cv2.imread(f'{TRAIN_DIR}/{img_id}.jpg')
        h, w = img.shape[:2]
        with open(f'{TRAIN_DIR}/{img_id}.txt') as f:
            for line in f:
                vals = list(map(float, line.split()))
                if not vals:
                    continue
                kpt = vals[5:]
                pts4 = np.array([[kpt[j] * w, kpt[j + 1] * h] for j in range(0, len(kpt), 3)])
                box_cx, box_cy, box_w, box_h = vals[1:5]
                box_xyxy = [(box_cx - box_w / 2) * w, (box_cy - box_h / 2) * h,
                            (box_cx + box_w / 2) * w, (box_cy + box_h / 2) * h]
                draw_detection(img, box_xyxy, pts4, thickness_scale=0.6)
        square = cv2.resize(img, (cell_size, cell_size))
        cells.append(square)

    while len(cells) < rows * cols:
        cells.append(np.zeros((cell_size, cell_size, 3), dtype=np.uint8))

    grid_rows = [np.hstack(cells[r * cols:(r + 1) * cols]) for r in range(rows)]
    grid = np.vstack(grid_rows)
    out_path = f'{OUT_DIR}/training_samples.jpg'
    cv2.imwrite(out_path, grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f'{len(candidates[:rows*cols])} examples -> {out_path}')


if __name__ == '__main__':
    import os
    os.makedirs(OUT_DIR, exist_ok=True)
    make_test_image_sample()
    make_training_gallery()
