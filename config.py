"""
Project Configuration File
"""
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.resolve()

# Dataset root directory.
# Keep empty only for local quick tests; for release/reproducibility, set an explicit absolute path.
DATA_DIR = Path("")
# Model root directory (used for local model paths).
# Keep empty only for local quick tests; for release/reproducibility, set an explicit absolute path.
MODEL_DIR = Path("")

SEED = 192


DATASET_PATHS = {
    "ml100k": {
        "root": (ml100k_root := DATA_DIR / "ml-100K"),
        "user": ml100k_root / "u.user",
        "item": ml100k_root / "u.item",
        "data": ml100k_root / "u.data",
        "genre": ml100k_root / "u.genre",
        "processed_dir": ml100k_root / "processed",
        "demo_dir": ml100k_root / "processed" / "demo",
        "checkpoints_dir": ml100k_root / "checkpoints"
    },
    "ml1m": {
        "root": (ml1m_root := DATA_DIR / "ml-1M"),
        "user": ml1m_root / "users.dat",
        "item": ml1m_root / "movies.dat",
        "data": ml1m_root / "ratings.dat",
        "processed_dir": ml1m_root / "processed",
        "demo_dir": ml1m_root / "processed" / "demo",
        "checkpoints_dir": ml1m_root / "checkpoints"
    },
    "electronics": {
        "root": (electronics_root := DATA_DIR / "electronics"),
        "reviews": electronics_root / "reviews_Electronics_5.json.gz",
        "meta": electronics_root / "meta_Electronics.json.gz",
        "processed_dir": electronics_root / "processed",
        "demo_dir": electronics_root / "processed" / "demo",
        "checkpoints_dir": electronics_root / "checkpoints"
    },
}


OUTPUT_FILES = {
    "processed": {
        "train": "train.json",
        "test": "test.json",
        "val": "val.json",
        "demo": {
            "cluster": "cluster.json",
            "manual": "manual.json"
        },
    },
    "checkpoints": {
        "weight_generator": "weight_generator",
        "train_save": "student",
        "online_weight_generator": "wg_online",
    },
    "logs": {
        "weight_generator": "weight_generator_training.log",
        "student_training": "student_training.log",
    },
}


DATA_PROCESS_CONFIG = {
    "ml100k": {
        "history_length": 10,
        "target_length": 5,
        "step": 3,
        "min_user_windows": 4
    },
    "ml1m": {
        "history_length": 10,
        "target_length": 5,
        "step": 3,
        "min_user_windows": 4
    },
    "electronics": {
        "history_length": 10,
        "target_length": 5,
        "step": 3,
        "min_user_windows": 4
    }
}


LOCAL_MODEL_PATHS = {
    "qwen2.5-3b": MODEL_DIR / "Qwen2.5-3B"
}

MODEL_CONFIG = {
    "cot_types": ["zero-shot", "cluster", "manual"],
    "student_model": "qwen2.5-3b",
}

PROMPT_CONFIG = {
    "api_key_env": "OPENAI_API_KEY",
    "api_key": "",
    # Local OpenAI-compatible endpoint for release defaults.
    # If serving for other devices in LAN, replace 127.0.0.1 with your host IP, e.g.:
    "base_url": "http://127.0.0.1:8000/v1",
    # "base_url": "https://api.chatanywhere.tech/v1",
    "model": "Qwen2___5-32B-Instruct",
    # "model": "gpt-3.5-turbo",
    # "model" = "gpt-4o"
}

CLUSTER_CONFIG = {
    "num_clusters": 6,
    "random_seed": 192,
    "encoder": "all-MiniLM-L6-v2"
}

def get_seed():
    return SEED


def get_dataset_path(dataset_name, file_type="root"):

    if dataset_name not in DATASET_PATHS:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if file_type not in DATASET_PATHS[dataset_name]:
        raise ValueError(f"Unknown file type '{file_type}' for dataset '{dataset_name}'")

    return DATASET_PATHS[dataset_name][file_type]


def ensure_processed_dir(dataset_name):

    processed_dir = get_dataset_path(dataset_name, "processed_dir")
    processed_dir.mkdir(parents=True, exist_ok=True)
    return processed_dir


def ensure_demo_dir(dataset_name):

    demo_dir = get_dataset_path(dataset_name, "demo_dir")
    demo_dir.mkdir(parents=True, exist_ok=True)
    return demo_dir


def ensure_checkpoints_dir(dataset_name):

    checkpoints_dir = get_dataset_path(dataset_name, "checkpoints_dir")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    return checkpoints_dir


def get_processed_file_path(dataset_name, file_type):

    processed_dir = ensure_processed_dir(dataset_name)
    if file_type not in OUTPUT_FILES["processed"]:
        raise ValueError(f"Unknown output file type: {file_type}")

    return processed_dir / OUTPUT_FILES["processed"][file_type]


def get_demo_path(dataset_name):

    demo_dir = ensure_demo_dir(dataset_name)
    return demo_dir


def get_demo_file_path(dataset_name, demo_type="cluster"):

    demo_dir = ensure_demo_dir(dataset_name)
    if demo_type not in OUTPUT_FILES["processed"]["demo"]:
        raise ValueError(f"Unknown demo type: {demo_type}")

    return demo_dir / OUTPUT_FILES["processed"]["demo"][demo_type]


def get_checkpoints_path(dataset_name):
    checkpoints_dir = ensure_checkpoints_dir(dataset_name)
    return checkpoints_dir


def get_weight_generator_path(dataset_name):
    checkpoints_dir = ensure_checkpoints_dir(dataset_name)
    return checkpoints_dir / OUTPUT_FILES["checkpoints"]["weight_generator"]


def get_online_weight_generator_path(dataset_name):
    checkpoints_dir = ensure_checkpoints_dir(dataset_name)
    return checkpoints_dir / OUTPUT_FILES["checkpoints"]["online_weight_generator"]


def get_train_save_path(dataset_name):
    checkpoints_dir = ensure_checkpoints_dir(dataset_name)
    return checkpoints_dir / OUTPUT_FILES["checkpoints"]["train_save"]


def get_data_process_config(dataset_name):

    return DATA_PROCESS_CONFIG.get(dataset_name, DATA_PROCESS_CONFIG["ml100k"])


def get_cluster_config():

    return CLUSTER_CONFIG.copy()


def get_prompt_config():

    return PROMPT_CONFIG.copy()


def get_student_model_name():

    return MODEL_CONFIG["student_model"]


def get_model_path(model_name):
    # Keep signature for compatibility; force a single local 3B model for release reproducibility.
    _ = model_name
    return LOCAL_MODEL_PATHS["qwen2.5-3b"]


def infer_dataset_name():

    import inspect


    frame = inspect.currentframe()


    for _ in range(10):
        if frame is None:
            break
        frame = frame.f_back
        if frame is None:
            break

        module_path = frame.f_code.co_filename
        parts = Path(module_path).parts


        try:
            datasets_idx = parts.index('datasets')
            if datasets_idx + 1 < len(parts):
                dataset_name = parts[datasets_idx + 1]
                if dataset_name in DATASET_PATHS:
                    return dataset_name
        except (ValueError, IndexError):
            continue

    return None


def validate_paths(dataset_name):

    dataset_config = DATASET_PATHS[dataset_name]
    missing_files = []

    for key, path in dataset_config.items():
        if key in ["processed_dir", "demo_dir"]:
            continue
        if not path.exists():
            missing_files.append(str(path))

    if missing_files:
        print(f"⚠️  Warning: Missing files for {dataset_name}:")
        for f in missing_files:
            print(f"   - {f}")
        return False

    print(f"✅ All files found for {dataset_name}")
    return True
