import os
import json
from tqdm import tqdm
from prompt import openai_api

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_demo_file(file_path):

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and 'demonstrations' in data:

        print("  📄 Detected new format (with metadata)")
        return data['demonstrations'], data.get('metadata')
    else:
        raise ValueError(f"Invalid demo file format in {file_path}")


def save_demo_file(file_path, demonstrations, metadata):

    if metadata is not None:

        output_data = {
            'demonstrations': demonstrations,
            'metadata': metadata
        }
    else:

        output_data = demonstrations

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)


def generate_prefix(file_path):

    print(f"\n{'=' * 60}")
    print("Starting Demo Generation")
    print(f"Generating for Cluster File: {file_path}")
    print(f"{'=' * 60}")


    demos, metadata = load_demo_file(file_path)

    print(f"\nTotal items to process: {len(demos)}\n")

    for demo in tqdm(demos, desc="Generating preferences"):
        prompt = (
                "viewing history: " + demo["demo_history"] + "\n"
                + "Please analyze the user's preferences in 100 words based on the viewing history.\n"
        )
        demo["preference"] = openai_api(prompt)


    save_demo_file(file_path, demos, metadata)

    print(f"\n✅ Generation complete! Saved {len(demos)} items to {file_path}")
