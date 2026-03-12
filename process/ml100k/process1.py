import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path


_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config import get_dataset_path, get_processed_file_path, DATA_PROCESS_CONFIG


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


class Dataset(object):

    DATASET_NAME = "ml100k"

    def __init__(self, u2i=True):

        self.use_u2i = u2i


        print(f"Loading {self.DATASET_NAME} dataset...")


        self.users = self._load_users()
        self.genre = self._load_genres()
        self.items = self._load_items()
        self.dataset = self._load_data()

        self.n_users = self.dataset["userID"].nunique()
        self.n_items = self.dataset["itemID"].nunique()
        print("user number:", self.n_users)
        print("item number:", self.n_items)

    def _load_users(self):

        users = dict()
        user_file = get_dataset_path(self.DATASET_NAME, "user")
        with open(user_file, "r") as rf:
            for i in rf.readlines():
                i = i.strip()
                if not i or i == "":
                    continue
                load_user = i.split("|")
                for j, v in enumerate(load_user[:]):
                    load_user[j] = v.strip()
                    if load_user[j] == "M":
                        load_user[j] = "male"
                    if load_user[j] == "F":
                        load_user[j] = "female"
                users[int(load_user[0])] = load_user
        return users

    def _load_genres(self):

        genre = dict()
        genre_file = get_dataset_path(self.DATASET_NAME, "genre")
        with open(genre_file, "r") as rf:
            for i in rf.readlines():
                i = i.strip()
                if not i or i == "":
                    continue
                load_genre = i.split("|")
                genre[load_genre[1]] = load_genre[0]
        return genre

    def _load_items(self):

        items = dict()
        item_file = get_dataset_path(self.DATASET_NAME, "item")
        with open(item_file, "r", encoding="latin-1") as rf:
            for i in rf.readlines():
                i = i.strip()
                if not i or i == "":
                    continue
                load_item = i.split("|")

                item = [load_item[1]]
                genre_str = ""
                for j in range(18):
                    if load_item[-18 + j] == "1":
                        genre_str = genre_str + self.genre[str(j)] + "|"
                genre_str = genre_str[:-1]
                item.append(genre_str)
                item[0] = deal_with(item[0])
                items[int(load_item[0])] = item
        return items

    def _load_data(self):

        header = ["userID", "itemID", "rating", "timestamp"]
        data_file = get_dataset_path(self.DATASET_NAME, "data")
        data = pd.read_csv(data_file, sep="\t", names=header, engine="python")

        return data

    def process_data(
        self,
        order=True,
        max_history_length=None,
        target_length=None,
        step=None,
        min_user_windows=None
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

    def generate_data_train(self, history_length, target_length, step=3, min_user_windows=4):
        train_set = []
        test_set = []
        validation_set = []

        processed_data = self.proc_dataset.copy()
        rrr = []
        for uid, group in processed_data.groupby("userID"):
            user_list = []

            ind = len(group.index) - history_length - target_length
            rrr.append(len(group.index))
            window_length = history_length + target_length
            while ind >= 0:
                if ind + window_length <= len(group.index):
                    user_list.append(list(group.index[ind : ind + window_length]))
                else:
                    if len(group.index) < window_length:
                        user_list.append(
                            list(group.index[ind:]) + list(group.index[:ind])
                        )
                    else:
                        user_list.append(
                            list(group.index[ind:])
                            + list(group.index[: window_length - (len(group.index) - ind)])
                        )
                ind -= step
            if len(user_list) >= min_user_windows:

                start = """"""
                history = start
                result = []
                front = []
                for index, i in enumerate(user_list[0]):
                    if index >= history_length:
                        result.append(self.items[group.loc[i, "itemID"]][0])
                    else:
                        front.append(self.items[group.loc[i, "itemID"]][0])
                        v1 = self.items[group.loc[i, "itemID"]][0]
                        v2 = group.loc[i, "rating"]
                        history += f"({v1}, {v2} star); "
                history = history[:-2] + "."
                validation_set.append(
                    {"history": history, "result": result, "front": front, "user_id": int(uid)}
                )

                history = start
                result = []
                front = []
                for index, i in enumerate(user_list[1]):
                    if index >= history_length:
                        result.append(self.items[group.loc[i, "itemID"]][0])
                    else:
                        front.append(self.items[group.loc[i, "itemID"]][0])
                        v1 = self.items[group.loc[i, "itemID"]][0]
                        v2 = group.loc[i, "rating"]
                        history += f"({v1}, {v2} star); "
                history = history[:-2] + "."
                test_set.append({"history": history, "result": result, "front": front, "user_id": int(uid)})

                for freq in user_list[2:]:
                    history = start
                    result = []
                    front = []
                    for index, i in enumerate(freq):
                        if index >= history_length:
                            result.append(self.items[group.loc[i, "itemID"]][0])
                        else:
                            front.append(self.items[group.loc[i, "itemID"]][0])
                            v1 = self.items[group.loc[i, "itemID"]][0]
                            v2 = group.loc[i, "rating"]
                            history += f"({v1}, {v2} star); "
                    history = history[:-2] + "."
                    train_set.append(
                        {"history": history, "result": result, "front": front, "user_id": int(uid)}
                    )

        print("min(rrr)", min(rrr))
        print("max(rrr)", max(rrr))

        print("train len:", len(train_set))
        print("test len:", len(test_set))
        print("val len:", len(validation_set))

        self.validation_set = validation_set
        self.test_set = test_set
        self.train_set = train_set


        train_path = get_processed_file_path(self.DATASET_NAME, "train")
        test_path = get_processed_file_path(self.DATASET_NAME, "test")
        val_path = get_processed_file_path(self.DATASET_NAME, "val")

        with open(train_path, "w") as f:
            json.dump(self.train_set, f)
        with open(test_path, "w") as f:
            json.dump(self.test_set, f)
        with open(val_path, "w") as f:
            json.dump(self.validation_set, f)
