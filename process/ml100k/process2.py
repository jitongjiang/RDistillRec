
import json
import random
import numpy as np
import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config import get_processed_file_path, get_seed
DATASET_NAME = "ml100k"

def get_all_items_from_processed_data(seed):

    all_items = set()


    for file_type in ["train", "val", "test"]:
        file_path = get_processed_file_path(DATASET_NAME, file_type)
        if not file_path.exists():
            print(f"⚠️  Warning: {file_path} not found, skipping...")
            continue

        with open(file_path, "r") as f:
            data = json.load(f)


        for item in data:
            all_items.update(item.get("result", []))
            all_items.update(item.get("front", []))


    all_items_list = sorted(list(all_items))
    random.Random(seed).shuffle(all_items_list)
    return all_items_list


def _build_user_positive_items():

    user_items = {}
    missing_user_id = False

    for file_type in ["train", "val", "test"]:
        file_path = get_processed_file_path(DATASET_NAME, file_type)
        if not file_path.exists():
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for sample in data:
            user_id = sample.get("user_id")
            if user_id is None:
                missing_user_id = True
                continue

            items = user_items.setdefault(user_id, set())
            items.update(sample.get("front", []))
            items.update(sample.get("result", []))

    return user_items, missing_user_id


def add_detail(file_path, num_candidates=100, seed=get_seed(), strict_negative=False):


    if not file_path.exists():
        print(f"⚠️  Warning: {file_path} does not exist, skipping...")
        return

    print(f"\n{'=' * 60}")
    print(f"Processing: {file_path.name}")
    print(f"{'=' * 60}")


    with open(file_path, "r") as f:
        data = json.load(f)

    print(f"📊 Processing {len(data)} samples...")


    print("📦 Collecting all items for negative sampling...")
    all_items = get_all_items_from_processed_data(seed)

    if not all_items:
        print("❌ Error: No items found! Make sure all train/val/test.json files exist.")
        return

    print(f"✅ Found {len(all_items)} unique items for negative sampling")
    print(f"🎲 Using random seed: {seed} for reproducible sampling")
    user_positive_items = {}
    if strict_negative:
        user_positive_items, missing_user_id = _build_user_positive_items()
        if missing_user_id:
            print("??  Warning: user_id missing in dataset; strict_negative disabled.")
            strict_negative = False


    for i, v in enumerate(data):

        sample_seed = seed + i
        random.seed(sample_seed)
        np.random.seed(sample_seed)


        result_list = v.get("result", [])
        if isinstance(result_list, list):
            primary_result = result_list[0] if result_list else None
            other_results = result_list[1:] if len(result_list) > 1 else []
        else:
            primary_result = result_list
            other_results = []
        rec_set = set()
        if primary_result is not None and primary_result != "":
            rec_set.add(primary_result)


        front_set = set(v.get("front", []))
        blocked_set = front_set.union(other_results)
        if strict_negative:
            user_id = v.get("user_id")
            if user_id is not None:
                blocked_set = blocked_set.union(user_positive_items.get(user_id, set()))


        attempts = 0
        max_attempts = num_candidates * 10

        while len(rec_set) < num_candidates and attempts < max_attempts:

            candidate = random.choice(all_items)


            if candidate not in blocked_set:
                rec_set.add(candidate)

            attempts += 1

        if len(rec_set) < num_candidates:
            print(f"⚠️  Warning: Sample {i} only has {len(rec_set)}/{num_candidates} candidates")


        rec_list = sorted(list(rec_set))
        np.random.shuffle(rec_list)


        data[i]["recommendations"] = rec_list


    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"✅ Saved {len(data)} samples with ~{num_candidates} candidates each")
    print(f"💾 Output: {file_path}\n")
