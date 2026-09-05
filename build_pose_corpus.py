"""
Rebuilds ~/yolo-pose-training/model_training/images/ as a symlink farm
sourced from every downloaded game's catalog, each capped to CAP images
so no single game (Pokemon, previously the only game in this dataset at
25,379 images, or Magic at 110,629) dominates the synthetic composite mix
that create_images.py/create_val_images.py generate keypoint-labeled
training examples from.

The pose model only needs "a card silhouette + 4 corners" per example,
not per-product identification, so a random capped subset of each game's
catalog is exactly as useful as the full thing for that purpose -- and
visual diversity across games (different borders/holo patterns/card
backs) should help it generalize the corner-detection task better than
Pokemon-only ever could.

Existing symlinks are cleared and rebuilt from scratch each run (so
re-running with a different CAP or game list doesn't leave stale entries
from a previous composition).
"""
import random
from pathlib import Path

SIGLIP_SCANNER = Path("/home/user/siglip-scanner")
TARGET_DIR = Path("/home/user/yolo-pose-training/model_training/images")
CAP = 10000
SEED = 42

# game key -> image root (recursively holds {group_id}/{product_id}.jpg)
GAME_ROOTS = {
    "pokemon": SIGLIP_SCANNER / "vectors/images/3",
    "onepiece": SIGLIP_SCANNER / "onepiece_eval/images/68",
    "magic": SIGLIP_SCANNER / "magic_eval/images/1",
    "riftbound": SIGLIP_SCANNER / "riftbound_eval/images/89",
    "lorcana": SIGLIP_SCANNER / "lorcana_eval/images/71",
    "yugioh": SIGLIP_SCANNER / "yugioh_eval/images/2",
    "digimon": SIGLIP_SCANNER / "digimon_eval/images/63",
    "fleshandblood": SIGLIP_SCANNER / "fleshandblood_eval/images/62",
    "starwarsunlimited": SIGLIP_SCANNER / "starwarsunlimited_eval/images/79",
}


def main():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    for existing in TARGET_DIR.iterdir():
        if existing.is_symlink() or existing.is_file():
            existing.unlink()

    random.seed(SEED)
    total = 0
    for game, root in GAME_ROOTS.items():
        if not root.exists():
            print(f"{game}: SKIPPED (no images at {root})")
            continue
        images = sorted(root.rglob("*.jpg")) + sorted(root.rglob("*.png"))
        sampled = images if len(images) <= CAP else random.sample(images, CAP)
        for src in sampled:
            link_name = f"{game}_{src.name}"
            (TARGET_DIR / link_name).symlink_to(src)
        print(f"{game}: {len(sampled)}/{len(images)} images linked")
        total += len(sampled)

    print(f"Total: {total} images in {TARGET_DIR}")


if __name__ == "__main__":
    main()
