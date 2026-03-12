import json
import random

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import get_dataset_path, get_processed_file_path, get_seed

DATASET_NAME = "ml1m"


def deal_with(name):
    name = name.strip()
    if name.find("(") != -1:
        name = name[: name.find("(")]
    name = name.strip()
    if name.endswith("The"):
        name = name[:-5]
    if name.endswith("The Movie"):
        name = name[:-11]
    name = name.strip()
    return name


def get_ratings(file):
    header = ["userID", "itemID", "rating", "timestamp"]
    data = pd.read_csv(file, sep="::", names=header, engine="python")
    return data


def get_items(file):
    data = pd.read_csv(
        file,
        engine="python",
        sep="::",
        encoding="ISO-8859-1",
        names=["itemID", "title", "genres"],
    )
    return data


def get_users(file):
    data = pd.read_csv(
        file,
        engine="python",
        sep="::",
        encoding="ISO-8859-1",
        names=["userID", "gender", "age", "occupation", "zip_code"],
    )

    def occupations_map(occupation):
        occupations_dict = {
            1: "technician",
            0: "other",
            2: "writer",
            3: "executive",
            4: "administrator",
            5: "student",
            6: "lawyer",
            7: "educator",
            8: "scientist",
            9: "entertainment",
            10: "programmer",
            11: "librarian",
            12: "homemaker",
            13: "artist",
            14: "engineer",
            15: "marketing",
            16: "none",
            17: "healthcare",
            18: "retired",
            19: "salesman",
            20: "doctor",
        }
        return occupations_dict[occupation]

    data["occupation"] = data["occupation"].apply(
        lambda occupation: occupations_map(occupation)
    )
    data["gender"] = data["gender"].map({"M": "male", "F": "female"})


    return data


def get_all_items_from_processed_data(seed):
    all_items = set()
    for file_type in ["train", "val", "test"]:
        file_path = get_processed_file_path(DATASET_NAME, file_type)
        if not file_path.exists():
            continue
        with open(file_path, "r", encoding="utf-8") as f:
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

def add_detail(filename, seed=None, item_path=None, strict_negative=False):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if item_path is None:
        item_path = get_dataset_path("ml1m", "item")

    data = None
    with open(filename, "r") as f:
        data = json.load(f)

    all_items = get_all_items_from_processed_data(seed or get_seed())
    if not all_items:
        print("Error: No items found in processed data.")
        return

    user_positive_items = {}
    if strict_negative:
        user_positive_items, missing_user_id = _build_user_positive_items()
        if missing_user_id:
            print("Warning: user_id missing in dataset; strict_negative disabled.")
            strict_negative = False

    for i, v in enumerate(tqdm(data, desc="Adding details", leave=False)):
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

        front_set = set()
        for j in v["front"]:
            front_set.add(j)

        blocked_set = front_set.union(other_results)
        if strict_negative:
            user_id = v.get("user_id")
            if user_id is not None:
                blocked_set = blocked_set.union(user_positive_items.get(user_id, set()))
        while len(rec_set) < 100:
            candidate = random.choice(all_items)
            if candidate not in blocked_set:
                rec_set.add(candidate)

        rec_list = list(rec_set)
        np.random.shuffle(rec_list)

        data[i]["recommendations"] = rec_list

    print(filename, "len", len(data))

    with open(filename, "w") as f:
        json.dump(data, f)

