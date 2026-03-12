import ast
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config import get_dataset_path, get_processed_file_path, get_seed, DATA_PROCESS_CONFIG


def _open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _normalize_title(title: str) -> str:
    if title is None:
        return ""
    title = str(title).replace("\n", " ").strip()
    title = " ".join(title.split())
    return title


def _parse_obj(line: str) -> dict | None:
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(line)
        except (ValueError, SyntaxError):
            return None


def _load_metadata(meta_path: Path) -> dict:
    meta = {}
    missing_title = 0
    total = 0
    skipped_bad = 0

    with _open_text(meta_path) as rf:
        for line in rf:
            line = line.strip()
            if not line:
                continue
            total += 1
            obj = _parse_obj(line)
            if obj is None:
                skipped_bad += 1
                continue
            asin = obj.get("asin")
            title = _normalize_title(obj.get("title", ""))
            if not asin or not title:
                missing_title += 1
                continue
            meta[asin] = title

    print(
        "metadata loaded:"
        f" {len(meta)} items (missing title: {missing_title}, bad: {skipped_bad}, total: {total})"
    )
    return meta


def _load_reviews(reviews_path: Path, meta: dict, max_reviews: int | None = None) -> pd.DataFrame:
    records = []
    skipped_no_meta = 0
    skipped_bad = 0

    with _open_text(reviews_path) as rf:
        for line in rf:
            line = line.strip()
            if not line:
                continue
            obj = _parse_obj(line)
            if obj is None:
                skipped_bad += 1
                continue

            reviewer = obj.get("reviewerID")
            asin = obj.get("asin")
            rating = obj.get("overall")
            ts = obj.get("unixReviewTime")

            if reviewer is None or asin is None or rating is None or ts is None:
                skipped_bad += 1
                continue

            title = meta.get(asin)
            if not title:
                skipped_no_meta += 1
                continue

            try:
                rating_val = float(rating)
            except (TypeError, ValueError):
                skipped_bad += 1
                continue

            records.append(
                {
                    "userID": reviewer,
                    "itemID": title,
                    "rating": rating_val,
                    "timestamp": int(ts),
                }
            )

            if max_reviews is not None and len(records) >= max_reviews:
                break

    print(f"reviews loaded: {len(records)} (skipped no meta: {skipped_no_meta}, bad: {skipped_bad})")
    return pd.DataFrame.from_records(records)


class Dataset(object):
    DATASET_NAME = "electronics"

    def __init__(self, reviews_path: str | None = None, meta_path: str | None = None):
        reviews_path = Path(reviews_path) if reviews_path else get_dataset_path(self.DATASET_NAME, "reviews")
        meta_path = Path(meta_path) if meta_path else get_dataset_path(self.DATASET_NAME, "meta")

        if not reviews_path.exists():
            raise FileNotFoundError(f"Reviews file not found: {reviews_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        print(f"Loading {self.DATASET_NAME} reviews: {reviews_path.name}")
        print(f"Loading {self.DATASET_NAME} metadata: {meta_path.name}")

        meta = _load_metadata(meta_path)
        self.dataset = _load_reviews(reviews_path, meta)

        if self.dataset.empty:
            raise RuntimeError("No valid review records loaded. Check paths or metadata coverage.")

        self.n_users = self.dataset["userID"].nunique()
        self.n_items = self.dataset["itemID"].nunique()
        print("user number:", self.n_users)
        print("item number:", self.n_items)

    def process_data(
        self,
        order: bool = True,
        max_history_length: int | None = None,
        target_length: int | None = None,
        step: int | None = None,
        max_users: int | None = None,
        min_user_windows: int | None = None
    ):
        cfg = DATA_PROCESS_CONFIG.get(self.DATASET_NAME, {})
        history_length = max_history_length or cfg.get("history_length", 10)
        target_length = target_length or cfg.get("target_length", 5)
        step = step or cfg.get("step", 3)
        min_windows = (
            min_user_windows
            if min_user_windows is not None
            else cfg.get("min_user_windows", 4)
        )

        self.proc_dataset = self.dataset.copy()

        if max_users is not None and max_users > 0:
            rng = np.random.RandomState(get_seed())
            user_ids = self.proc_dataset["userID"].unique()
            if len(user_ids) > max_users:
                sampled = rng.choice(user_ids, size=max_users, replace=False)
                self.proc_dataset = self.proc_dataset[self.proc_dataset["userID"].isin(sampled)].copy()

        if order:
            self.proc_dataset = self.proc_dataset.sort_values(
                by=["timestamp", "userID", "itemID"]
            ).reset_index(drop=True)

        self.generate_data_train(
            history_length=history_length,
            target_length=target_length,
            step=step,
            min_user_windows=min_windows
        )

    def generate_data_train(
        self,
        history_length: int,
        target_length: int,
        step: int = 3,
        min_user_windows: int = 4
    ):
        train_set = []
        test_set = []
        validation_set = []

        processed_data = self.proc_dataset.copy()
        rrr = []

        for uid, group in processed_data.groupby("userID"):
            user_list = []
            indices = list(group.index)

            ind = len(indices) - history_length - target_length
            rrr.append(len(indices))
            window_length = history_length + target_length
            while ind >= 0:
                if ind + window_length <= len(indices):
                    user_list.append(list(indices[ind: ind + window_length]))
                else:
                    if len(indices) < window_length:
                        user_list.append(list(indices[ind:]) + list(indices[:ind]))
                    else:
                        user_list.append(
                            list(indices[ind:])
                            + list(indices[: window_length - (len(indices) - ind)])
                        )
                ind -= step

            if len(user_list) >= min_user_windows:
                start = """"""

                history = start
                result = []
                front = []
                for index, i in enumerate(user_list[0]):
                    item_name = group.loc[i, "itemID"]
                    if index >= history_length:
                        result.append(item_name)
                    else:
                        front.append(item_name)
                        rating = int(group.loc[i, "rating"])
                        history += f"({item_name}, {rating} star); "
                history = history[:-2] + "."
                validation_set.append({"history": history, "result": result, "front": front, "user_id": uid})

                history = start
                result = []
                front = []
                for index, i in enumerate(user_list[1]):
                    item_name = group.loc[i, "itemID"]
                    if index >= history_length:
                        result.append(item_name)
                    else:
                        front.append(item_name)
                        rating = int(group.loc[i, "rating"])
                        history += f"({item_name}, {rating} star); "
                history = history[:-2] + "."
                test_set.append({"history": history, "result": result, "front": front, "user_id": uid})

                for freq in user_list[2:]:
                    history = start
                    result = []
                    front = []
                    for index, i in enumerate(freq):
                        item_name = group.loc[i, "itemID"]
                        if index >= history_length:
                            result.append(item_name)
                        else:
                            front.append(item_name)
                            rating = int(group.loc[i, "rating"])
                            history += f"({item_name}, {rating} star); "
                    history = history[:-2] + "."
                    train_set.append({"history": history, "result": result, "front": front, "user_id": uid})

        print("min(rrr)", min(rrr) if rrr else 0)
        print("max(rrr)", max(rrr) if rrr else 0)
        print("train len:", len(train_set))
        print("test len:", len(test_set))
        print("val len:", len(validation_set))

        self.validation_set = validation_set
        self.test_set = test_set
        self.train_set = train_set

        train_path = get_processed_file_path(self.DATASET_NAME, "train")
        test_path = get_processed_file_path(self.DATASET_NAME, "test")
        val_path = get_processed_file_path(self.DATASET_NAME, "val")

        with open(train_path, "w", encoding="utf-8") as f:
            json.dump(self.train_set, f, ensure_ascii=False)
        with open(test_path, "w", encoding="utf-8") as f:
            json.dump(self.test_set, f, ensure_ascii=False)
        with open(val_path, "w", encoding="utf-8") as f:
            json.dump(self.validation_set, f, ensure_ascii=False)


if __name__ == "__main__":
    data = Dataset()
    data.process_data()
