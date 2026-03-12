import json

import numpy as np
import pandas as pd
from tqdm import tqdm

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


def _load_default_data():
    rating_data = get_ratings(get_dataset_path("ml1m", "data"))
    user_data = get_users(get_dataset_path("ml1m", "user"))
    item_data = get_items(get_dataset_path("ml1m", "item"))
    return rating_data, user_data, item_data


class Dataset(object):
    def __init__(self, dataset=None, user_dataset=None, item_dataset=None, u2i=True, dataset_name="ml1m"):
        if dataset is None or user_dataset is None or item_dataset is None:
            rating_data, user_data, item_data = _load_default_data()
            if dataset is None:
                dataset = rating_data
            if user_dataset is None:
                user_dataset = user_data
            if item_dataset is None:
                item_dataset = item_data

        self.dataset = dataset
        self.user_dataset = user_dataset
        self.item_dataset = item_dataset
        self.use_u2i = u2i
        self.dataset_name = dataset_name
        self.n_users = self.dataset["userID"].nunique()
        self.n_items = self.dataset["itemID"].nunique()
        print("user number:", self.n_users)
        print("item number:", self.n_items)

    def process_data(
        self,
        order=True,
        leave_n=1,
        keep_n=5,
        max_history_length=None,
        target_length=None,
        step=None,
        premise_threshold=10,
        min_user_windows=None,
    ):

        cfg = DATA_PROCESS_CONFIG.get(self.dataset_name, {})
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


    def generate_data_train(self, history_length, target_length, step=8, min_user_windows=4):
        train_set = []
        test_set = []
        validation_set = []

        processed_data = self.proc_dataset.copy()

        rrr = []
        for uid, group in tqdm(
            processed_data.groupby("userID"),
            total=self.n_users,
            desc="Users",
            leave=False
        ):
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
                prompt = start
                result = []
                front = []
                for index, i in enumerate(user_list[0]):
                    m_name = self.item_dataset.loc[
                        self.item_dataset["itemID"] == group.loc[i, "itemID"]
                    ].values[0][1]
                    m_name = deal_with(m_name)
                    if index >= history_length:
                        result.append(m_name)
                    else:
                        front.append(m_name)
                        v1 = m_name
                        v2 = group.loc[i, "rating"]
                        prompt += f"({v1}, {v2} star); "
                prompt = prompt[:-2] + "."
                validation_set.append(
                    {"history": prompt, "result": result, "front": front, "user_id": int(uid)}
                )

                prompt = start
                result = []
                front = []
                for index, i in enumerate(user_list[1]):
                    m_name = self.item_dataset.loc[
                        self.item_dataset["itemID"] == group.loc[i, "itemID"]
                    ].values[0][1]
                    m_name = deal_with(m_name)
                    if index >= history_length:
                        result.append(m_name)
                    else:
                        front.append(m_name)
                        v1 = m_name
                        v2 = group.loc[i, "rating"]
                        prompt += f"({v1}, {v2} star); "
                prompt = prompt[:-2] + "."
                test_set.append({"history": prompt, "result": result, "front": front, "user_id": int(uid)})

                for freq in user_list[2:]:
                    prompt = start
                    result = []
                    front = []
                    for index, i in enumerate(freq):
                        m_name = self.item_dataset.loc[
                            self.item_dataset["itemID"] == group.loc[i, "itemID"]
                        ].values[0][1]
                        m_name = deal_with(m_name)
                        if index >= history_length:
                            result.append(m_name)
                        else:
                            front.append(m_name)
                            v1 = m_name
                            v2 = group.loc[i, "rating"]
                            prompt += f"({v1}, {v2} star); "
                    prompt = prompt[:-2] + "."
                    train_set.append(
                        {"history": prompt, "result": result, "front": front, "user_id": int(uid)}
                    )

        print("min(rrr)", min(rrr))
        print("max(rrr)", max(rrr))

        print("train len:", len(train_set))
        print("test len:", len(test_set))
        print("val len:", len(validation_set))

        self.validation_set = validation_set
        self.test_set = test_set
        self.train_set = train_set
        train_path = get_processed_file_path(self.dataset_name, "train")
        test_path = get_processed_file_path(self.dataset_name, "test")
        val_path = get_processed_file_path(self.dataset_name, "val")

        with open(train_path, "w", encoding="utf-8") as f:
            json.dump(self.train_set, f, ensure_ascii=False)
        with open(test_path, "w", encoding="utf-8") as f:
            json.dump(self.test_set, f, ensure_ascii=False)
        with open(val_path, "w", encoding="utf-8") as f:
            json.dump(self.validation_set, f, ensure_ascii=False)


if __name__ == "__main__":
    data = Dataset()
    data.process_data()
