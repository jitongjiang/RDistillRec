

import os
import json
import shutil
import argparse
from pathlib import Path
from typing import List, Dict, Optional

import torch
from tqdm import tqdm

from config import (
    get_processed_file_path,
    get_weight_generator_path,
    get_model_path,
    get_student_model_name,
    get_seed
)
from token_weight import RationaleTokenWeighting


def apply_weights_to_dataset(
    dataset_path: str,
    checkpoint_path: str,
    student_model_name: str,
    output_path: Optional[str] = None,
    backup: bool = True
) -> Dict[str, int]:

    print(f"\n{'='*70}")
    print(f"Applying Token Weights to Dataset")
    print(f"{'='*70}")
    print(f"  Dataset: {dataset_path}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Student Model: {student_model_name}")
    print(f"{'='*70}\n")


    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    print(f"✓ Loaded {len(dataset)} samples from {dataset_path.name}")


    print(f"\n📦 Loading Weight Generator...")
    module = RationaleTokenWeighting(
        student_model_name=student_model_name
    )
    module.load_weights(checkpoint_path)
    module.eval()

    print(f"✓ Weight Generator loaded successfully\n")


    processed = 0
    skipped = 0
    errors = []

    print(f"🔄 Computing token weights for {len(dataset)} samples...")

    for idx, sample in enumerate(tqdm(dataset, desc="Processing samples")):
        preference = sample['preference']

        if not preference:
            raise ValueError(f"Sample {idx} has empty 'preference' field")

        try:

            weights, tokens = module.compute_token_weights(preference)


            encoding = module.tokenizer(
                preference,
                return_tensors='pt',
                truncation=True,
                max_length=512
            )
            token_ids = encoding['input_ids'][0].tolist()


            if len(weights) != len(token_ids):
                print(f"\n⚠️  Warning: Length mismatch for sample {idx}")
                print(f"   Weights: {len(weights)}, Token IDs: {len(token_ids)}")

                if len(weights) > len(token_ids):
                    weights = weights[:len(token_ids)]
                    tokens = tokens[:len(token_ids)]
                else:

                    avg_weight = sum(weights) / len(weights)
                    weights.extend([avg_weight] * (len(token_ids) - len(weights)))
                    tokens.extend(['[PAD]'] * (len(token_ids) - len(tokens)))


            sample['token_weights'] = weights
            sample['token_ids'] = token_ids
            sample['tokens'] = tokens
            sample['token_weights_meta'] = {
                'checkpoint': str(checkpoint_path),
                'tokenizer': student_model_name,
                'num_tokens': len(weights)
            }

            processed += 1

        except Exception as e:
            error_msg = f"Sample {idx}: {type(e).__name__}: {str(e)}"
            errors.append(error_msg)
            skipped += 1


    print(f"\n{'='*70}")
    print(f"Processing Summary")
    print(f"{'='*70}")
    print(f"  Total Samples: {len(dataset)}")
    print(f"  Processed: {processed}")
    print(f"  Skipped: {skipped}")
    print(f"  Success Rate: {processed / len(dataset) * 100:.2f}%")

    if errors:
        print(f"\n⚠️  Errors encountered ({len(errors)}):")
        for error in errors[:5]:
            print(f"   - {error}")
        if len(errors) > 5:
            print(f"   ... and {len(errors) - 5} more errors")


    if skipped > 0:
        failure_rate = skipped / len(dataset) * 100

        if failure_rate > 5.0:
            print(f"\n❌ ERROR: Failure rate too high!")
            print(f"  {skipped}/{len(dataset)} samples failed ({failure_rate:.1f}%)")
            print("\nFirst 10 errors:")
            for err in errors[:10]:
                print(f"  {err}")
            raise RuntimeError(
                f"Failed to generate token weights for {skipped} samples. "
                f"Failure rate ({failure_rate:.1f}%) exceeds threshold (5%)."
            )

    print(f"{'='*70}\n")


    output_path = output_path or dataset_path


    if backup and Path(output_path).exists():
        backup_path = Path(output_path).with_suffix('.backup.json')
        shutil.copy(output_path, backup_path)
        print(f"💾 Backup saved to: {backup_path}")


    temp_path = Path(output_path).with_suffix('.tmp.json')
    with open(temp_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    shutil.move(str(temp_path), output_path)
    print(f"✓ Saved {len(dataset)} samples to: {output_path}\n")

    return {
        'processed': processed,
        'skipped': skipped,
        'total': len(dataset)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Apply Token Weights to Dataset (KPOD Stage 3.3)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )


    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (ml100k, ml1m, etc.)')


    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Weight Generator checkpoint path (auto-detect if not specified)')


    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'],
                        help='Dataset splits to process')


    parser.add_argument('--no-backup', action='store_true',
                        help='Skip backup of original files')

    parser.add_argument('--student-model', type=str, default=None,
                        help='Student model name (override config)')

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print("KPOD Stage 3.3: Apply Token Weights to Dataset")
    print(f"{'='*80}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Splits: {', '.join(args.splits)}")
    print(f"{'='*80}\n")


    if args.checkpoint is None:
        wg_dir = get_weight_generator_path(args.dataset)

        if not wg_dir.exists():
            print(f"❌ Weight Generator directory not found: {wg_dir}")
            exit(1)


        checkpoints = list(wg_dir.glob('weight_generator_epoch*.pt'))

        if not checkpoints:
            print(f"❌ No checkpoints found in {wg_dir}")
            exit(1)


        def get_epoch_num(path):

            try:
                filename = path.stem

                if 'epoch' not in filename:
                    raise ValueError(f"Invalid format: {filename}")


                epoch_str = filename.split('epoch')[-1]
                epoch_num = int(epoch_str)

                return epoch_num

            except (ValueError, AttributeError) as e:
                print(f"⚠️  Skipping invalid checkpoint: {path.name} ({e})")
                return -1


        valid_checkpoints = [cp for cp in checkpoints if get_epoch_num(cp) >= 0]

        if not valid_checkpoints:
            print(f"❌ No valid Weight Generator checkpoints found in {wg_dir}")
            exit(1)

        args.checkpoint = max(valid_checkpoints, key=get_epoch_num)
        print(f"📌 Auto-detected checkpoint: {args.checkpoint}")

    else:
        args.checkpoint = Path(args.checkpoint)
        if not args.checkpoint.exists():
            print(f"❌ Checkpoint not found: {args.checkpoint}")
            exit(1)


    student_model = args.student_model or get_model_path(get_student_model_name())
    print(f"📌 Student model: {student_model}\n")


    all_stats = {}

    for split in args.splits:
        print(f"\n{'─'*80}")
        print(f"Processing: {split} split")
        print(f"{'─'*80}")

        try:
            dataset_path = get_processed_file_path(args.dataset, split)

            if not dataset_path.exists():
                print(f"⚠️  {split}.json not found, skipping...")
                continue

            stats = apply_weights_to_dataset(
                dataset_path=str(dataset_path),
                checkpoint_path=str(args.checkpoint),
                student_model_name=student_model,
                backup=not args.no_backup
            )

            all_stats[split] = stats

        except Exception as e:
            print(f"\n❌ Error processing {split}: {e}")
            import traceback
            traceback.print_exc()
            continue


    print(f"\n{'='*80}")
    print("✅ Stage 3.3 Complete!")
    print(f"{'='*80}")

    for split, stats in all_stats.items():
        success_rate = stats['processed'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"  {split.upper():5s}: {stats['processed']:4d}/{stats['total']:4d} processed ({success_rate:.1f}%)")

    print(f"{'='*80}")


if __name__ == "__main__":
    main()
