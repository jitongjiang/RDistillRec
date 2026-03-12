
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    get_dataset_path,
    get_demo_file_path,
    get_cluster_config,
    get_processed_file_path,
    get_prompt_config
)
from process.ml100k.process1 import Dataset as ml100kDataset
from process.ml100k.process2 import add_detail as ml100kAddDetail
from process.ml1m.process1 import Dataset as ml1mDataset
from process.ml1m.process2 import add_detail as ml1mAddDetail
from process.electronics.process1 import Dataset as electronicsDataset
from process.electronics.process2 import add_detail as electronicsAddDetail


import process

from cluster import main as cluster_main
from generate import CoTGenerator, OpenAIBackend
from demo import generate_prefix
import json
import os


def parse_args():
    prompt_config = get_prompt_config()
    parser = argparse.ArgumentParser(
        description="RecCoT-SD Data Processing Pipeline"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ml100k",
        choices=["ml100k", "ml1m", "electronics", "movies"],
        help="Dataset to process (default: ml100k)"
    )
    parser.add_argument(
        "--cot",
        type=str,
        default="zero-shot",
        choices=["zero-shot", "cluster", "manual"],
        help="CoT generation method (default: zero-shot)"
    )
    parser.add_argument(
        "--num-clusters",
        type=int,
        default=None,
        help="Number of clusters (default: from config)"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Random seed (default: from config)"
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default=None,
        help="Sentence encoder (default: from config)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=prompt_config.get("model", "gpt-3.5-turbo"),
        help="OpenAI model name (default: gpt-3.5-turbo)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=prompt_config.get("api_key") or None,
        help="OpenAI API key (if not set, read from environment)"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=prompt_config.get("base_url"),
        help="API base URL (for custom endpoints like vLLM)"
    )
    parser.add_argument(
        "--skip-process",
        action="store_true",
        help="Skip dataset processing step and reuse existing processed files"
    )
    parser.add_argument(
        "--skip-cluster",
        action="store_true",
        help="Skip clustering step and reuse existing cluster results/demo"
    )
    parser.add_argument(
        "--skip-demo",
        action="store_true",
        help="Skip demo/prefix generation and reuse existing demo file"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate cluster visualization"
    )
    parser.add_argument(
        "--cot-workers",
        type=int,
        default=prompt_config.get("cot_workers", 1),
        help="Concurrent requests for CoT generation"
    )
    parser.add_argument(
        "--strict-negative",
        action="store_true",
        help="Exclude other user positives when sampling candidates"
    )
    args = parser.parse_args()
    return args


def main(args):
    prompt_config = get_prompt_config()
    api_key_env = prompt_config.get("api_key_env", "OPENAI_API_KEY")
    print("\n" + "="*60)
    print("RecCoT-SD Processing Pipeline")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"CoT Method: {args.cot}")
    print("="*60 + "\n")


    if args.skip_process:
        print("Step 1: Skipping dataset processing (using existing processed files)...")
        print("-"*60)
    else:
        print("Step 1: Processing dataset and generating templates...")
        print("-"*60)


    cluster_config = get_cluster_config()
    random_seed = args.random_seed if args.random_seed is not None else cluster_config["random_seed"]

    train_file_path = get_processed_file_path(args.dataset, "train")
    val_file_path = get_processed_file_path(args.dataset, "val")
    test_file_path = get_processed_file_path(args.dataset, "test")

    if not args.skip_process:
        if args.dataset == "ml100k":
            dataset = ml100kDataset()
            dataset.process_data()


            ml100kAddDetail(train_file_path, seed=random_seed, strict_negative=args.strict_negative)
            ml100kAddDetail(val_file_path, seed=random_seed, strict_negative=args.strict_negative)
            ml100kAddDetail(test_file_path, seed=random_seed, strict_negative=args.strict_negative)

        elif args.dataset == "ml1m":
            dataset = ml1mDataset()
            dataset.process_data()

            ml1mAddDetail(train_file_path, seed=random_seed, strict_negative=args.strict_negative)
            ml1mAddDetail(val_file_path, seed=random_seed, strict_negative=args.strict_negative)
            ml1mAddDetail(test_file_path, seed=random_seed, strict_negative=args.strict_negative)

        elif args.dataset == "electronics":
            dataset = electronicsDataset()
            dataset.process_data()

            electronicsAddDetail(train_file_path, seed=random_seed, strict_negative=args.strict_negative)
            electronicsAddDetail(val_file_path, seed=random_seed, strict_negative=args.strict_negative)
            electronicsAddDetail(test_file_path, seed=random_seed, strict_negative=args.strict_negative)


        print("\n✅ Template construction complete!")
    else:
        print("\nTemplate construction skipped (using existing processed files).")


    print("\nStep 2: Generating Chain-of-Thought data...")
    print("-"*60)

    if args.cot == "cluster":
        print("Using cluster-based CoT generation...")


        cluster_config = get_cluster_config()
        num_clusters = args.num_clusters if args.num_clusters is not None else cluster_config["num_clusters"]
        random_seed = args.random_seed if args.random_seed is not None else cluster_config["random_seed"]
        encoder = args.encoder if args.encoder is not None else cluster_config["encoder"]

        print(f"  Num clusters: {num_clusters}")
        print(f"  Random seed: {random_seed}")
        print(f"  Encoder: {encoder}")

        if args.skip_cluster:
            print("Warning: Skipping clustering step (using existing cluster results/demo).")
        else:
            try:

               cluster_main(
                    dataset=args.dataset,
                    num_clusters=num_clusters,
                    encoder=encoder,
                    random_seed=random_seed,
                    visualize=args.visualize,
                    text_field='history'
                )

            except Exception as e:
                print(f"Warning: Clustering failed: {e}")
                import traceback
                traceback.print_exc()
                print("   Continuing without cluster-based CoT...")


        try:

            backend = OpenAIBackend(
                model=args.model,
                api_key=args.api_key or os.getenv(api_key_env),
                base_url=args.base_url
            )


            generator = CoTGenerator(backend=backend, checkpoint_interval=100)


            demo_file_path = get_demo_file_path(args.dataset, "cluster")


            if args.skip_demo:
                print("Warning: Skipping demo prefix generation (using existing demo file).")
            else:
                generate_prefix(demo_file_path)


            print("\nGenerating CoT for train set...")
            generator.generate_dataset(
                train_file_path,
                demo_file=demo_file_path,
                concurrency=args.cot_workers
            )

            print("\nGenerating CoT for validation set...")
            generator.generate_dataset(
                val_file_path,
                demo_file=demo_file_path,
                concurrency=args.cot_workers
            )

            print("\nGenerating CoT for test set...")
            generator.generate_dataset(
                test_file_path,
                demo_file=demo_file_path,
                concurrency=args.cot_workers
            )

        except Exception as e:
            print(f"❌ Error in generate_cot: {e}")
            import traceback
            traceback.print_exc()

    elif args.cot == "manual":
        print("Using manual CoT generation...")


        demo_file_path = str(get_demo_file_path(args.dataset, "manual"))

        try:

            backend = OpenAIBackend(
                model=args.model,
                api_key=args.api_key or os.getenv(api_key_env),
                base_url=args.base_url
            )


            generator = CoTGenerator(backend=backend, checkpoint_interval=100)


            print("\nGenerating CoT for train set...")
            generator.generate_dataset(
                train_file_path,
                demo_file=demo_file_path,
                concurrency=args.cot_workers
            )

            print("\nGenerating CoT for validation set...")
            generator.generate_dataset(
                val_file_path,
                demo_file=demo_file_path,
                concurrency=args.cot_workers
            )

            print("\nGenerating CoT for test set...")
            generator.generate_dataset(
                test_file_path,
                demo_file=demo_file_path,
                concurrency=args.cot_workers
            )

        except Exception as e:
            print(f"❌ Error in generate_cot: {e}")
            import traceback
            traceback.print_exc()

    else:
        print("Using zero-shot CoT generation...")
        try:

            backend = OpenAIBackend(
                model=args.model,
                api_key=args.api_key or os.getenv(api_key_env),
                base_url=args.base_url
            )


            generator = CoTGenerator(backend=backend, checkpoint_interval=100)


            print("\nGenerating CoT for train set...")
            generator.generate_dataset(
                train_file_path,
                demo_file=None,
                concurrency=args.cot_workers
            )

            print("\nGenerating CoT for validation set...")
            generator.generate_dataset(
                val_file_path,
                demo_file=None,
                concurrency=args.cot_workers
            )

            print("\nGenerating CoT for test set...")
            generator.generate_dataset(
                test_file_path,
                demo_file=None,
                concurrency=args.cot_workers
            )

        except Exception as e:
            print(f"❌ Error in generate_cot: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("✅ All processing complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    args = parse_args()
    print("args", args)
    print("====Input Arguments====")

    try:
        main(args=args)
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
