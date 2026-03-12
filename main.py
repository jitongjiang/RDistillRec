

import argparse
import sys
import json
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent))

from config import (
    get_processed_file_path,
    get_model_path,
    get_demo_file_path,
    get_weight_generator_path,
    get_online_weight_generator_path,
    get_train_save_path,
    get_student_model_name
)


from token_weight import train_weight_generator, RationaleTokenWeighting
from difficulty import DifficultyEncoder, analyze_difficulty_distribution


from progressive_trainer import CurriculumScheduler, StudentTrainer

HAS_PROGRESSIVE_TRAINER = True

def print_header(title: str):

    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}\n")


def print_stage_header(stage: str, description: str):

    print(f"\n{'='*80}")
    print(f"{stage}: {description}")
    print(f"{'='*80}\n")


def check_prerequisites(dataset_name: str, require_preference: bool = True) -> bool:

    print("🔍 Checking prerequisites (Stage 1-2)...\n")

    try:

        train_path = get_processed_file_path(dataset_name, 'train')
        val_path = get_processed_file_path(dataset_name, 'val')
        test_path = get_processed_file_path(dataset_name, 'test')

        if not train_path.exists():
            print(f"❌ Training data not found: {train_path}")
            return False

        if not val_path.exists():
            print(f"❌ Validation data not found: {val_path}")
            return False

        if not test_path.exists():
            print(f"❌ Test data not found: {test_path}")
            return False


        with open(train_path, 'r', encoding='utf-8') as f:
            train_data = json.load(f)

        if not train_data:
            print(f"❌ Training data is empty")
            return False

        sample = train_data[0]
        required_fields = ['history', 'result', 'cluster_id', 'distance_to_center']

        missing_fields = []
        for field in required_fields:
            if field not in sample:
                missing_fields.append(field)

        if missing_fields:
            raise ValueError(
                f"❌ Dataset missing required fields: {missing_fields}\n"
                f"   python run.py --dataset {dataset_name} --cot cluster"
            )


        if require_preference and ('preference' not in sample or not sample['preference']):
            raise ValueError(
                f"❌ 'preference' field is missing or empty.\n"
                f"   Stage 2 (CoT generation) not completed.\n"
            )

        print(f"✓ Data validation passed")
        print(f"  Train: {len(train_data)} samples")

        with open(val_path, 'r', encoding='utf-8') as f:
            val_data = json.load(f)
        print(f"  Val: {len(val_data)} samples")

        with open(test_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        print(f"  Test: {len(test_data)} samples\n")
        return True

    except Exception as e:
        print(f"❌ Error checking prerequisites: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_stage_completed(stage: str, dataset_name: str, args=None) -> bool:

    try:
        if stage == '3.1':

            train_path = get_processed_file_path(dataset_name, 'train')
            if not train_path.exists():
                return False

            with open(train_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            return data and 'difficulty' in data[0]

        elif stage == '3.2':


            train_path = get_processed_file_path(dataset_name, 'train')
            if not train_path.exists():
                return False

            with open(train_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            return data and 'token_weights' in data[0]

        elif stage == '4':

            if args:
                student_dir = Path(args.train_save_dir)
            else:
                student_dir = Path('./checkpoints/student')

            final_model = student_dir / 'final'
            return final_model.exists()

        return False

    except Exception as e:
        print(f"⚠️  Warning: Error checking stage {stage} completion: {e}")
        return False


def stage_3_1_encode_difficulty(args):

    print_stage_header("Stage 3.1", "Encode Sample Difficulty (TDAS)")


    if args.skip_completed and check_stage_completed('3.1', args.dataset, args):
        print(f"✓ Stage 3.1 already completed, skipping...\n")
        return True


    model_path = get_model_path(args.student_model)

    print(f"⚖️  Difficulty weights: α={args.diff_alpha}, β={args.diff_beta}, γ={args.diff_gamma}")
    print(f"🤖 Student model: {args.student_model}")
    print(f"📍 Model path: {model_path}")
    print(f"📊 Normalize: {not args.no_normalize}")
    print(f"📈 Analyze: {not args.no_analyze}\n")


    encoder = DifficultyEncoder(
        alpha=args.diff_alpha,
        beta=args.diff_beta,
        gamma=args.diff_gamma
    )

    try:

        demo_path = get_demo_file_path(args.dataset, demo_type="cluster")
        print(f"📁 Demo file: {demo_path}")


        train_path = get_processed_file_path(args.dataset, 'train')
        print(f"📊 Encoding training set: {train_path}")
        encoder.encode_dataset(
            dataset_path=str(train_path),
            demo_file_path=str(demo_path),
            normalize=not args.no_normalize
        )


        if not args.no_analyze:
            print(f"\n📈 Analyzing difficulty distribution...")
            analyze_difficulty_distribution(train_path)

        print(f"\n✅ Stage 3.1 Complete!")
        print(f"   Note: Only training set difficulty encoded (val/test not needed for curriculum learning)\n")
        return True

    except Exception as e:
        print(f"\n❌ Stage 3.1 Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def stage_3_2_train_weight_generator(args):

    print_stage_header("Stage 3.2", "Train KPOD Weight Generator & Apply Token Weights")


    import os
    is_torchrun = 'LOCAL_RANK' in os.environ
    use_ddp = args.use_ddp or is_torchrun

    if use_ddp:
        import torch.distributed as dist
        from datetime import timedelta
        if not dist.is_initialized():
            ddp_timeout_minutes = int(os.environ.get("DDP_TIMEOUT_MINUTES", "1440"))
            dist.init_process_group(
                backend='nccl',
                timeout=timedelta(minutes=ddp_timeout_minutes)
            )
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1


    if rank == 0:
        print(f"🔧 Training mode: {'DDP' if use_ddp else 'Single GPU'}")
        if use_ddp:
            print(f"  GPUs: {world_size}")


    if args.skip_completed and check_stage_completed('3.2', args.dataset, args):
        if rank == 0:
            print(f"✓ Stage 3.2 already completed (token_weights found in dataset), skipping...\n")
        return True


    save_dir = Path(args.wg_save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)


    train_path = get_processed_file_path(args.dataset, 'train')


    model_path = get_model_path(args.student_model)

    if rank == 0:
        print(f"📂 Training data: {train_path}")
        print(f"🤖 Student model: {args.student_model}")
        print(f"📍 Model path: {model_path}")
        print(f"💾 Save directory: {save_dir}")
        print(f"🎯 Epochs: {args.wg_epochs}")
        print(f"📦 Batch size: {args.wg_batch_size}")
        print(f"📈 Learning rate: {args.wg_lr}")
        print(f"⚖️  Alpha (mask ratio weight): {args.wg_alpha}")
        print(f"🌡️  Temperature: {args.wg_temperature}\n")

    try:

        print(f"\n{'='*70}")
        print(f"Step 1: Training Weight Generator")
        print(f"{'='*70}\n")

        train_weight_generator(
            dataset_path=str(train_path),
            save_dir=str(save_dir),
            student_model_name=model_path,
            epochs=args.wg_epochs,
            batch_size=args.wg_batch_size,
            learning_rate=args.wg_lr,
            alpha=args.wg_alpha,
            temperature=args.wg_temperature,
            use_ddp=use_ddp,
            stopword_penalty=args.wg_stopword_penalty,
            stopword_margin=args.wg_stopword_margin,
            stopword_quantile=args.wg_stopword_quantile,
            stopword_quantile_cap=args.wg_stopword_quantile_cap
        )

        if rank == 0:
            checkpoint_path = save_dir / f"weight_generator_epoch{args.wg_epochs}.pt"
            print(f"\n✓ Weight Generator training complete")
        print(f"  Checkpoint: {checkpoint_path}\n")


        print(f"\n{'='*70}")
        print(f"Step 2: Applying Token Weights to Dataset")
        print(f"{'='*70}\n")

        from apply_token_weights import apply_weights_to_dataset


        for split in ['train', 'val', 'test']:
            print(f"\n{'─'*70}")
            print(f"Processing: {split} split")
            print(f"{'─'*70}")

            dataset_path = get_processed_file_path(args.dataset, split)

            if not dataset_path.exists():
                print(f"⚠️  {split}.json not found, skipping...")
                continue

            stats = apply_weights_to_dataset(
                dataset_path=str(dataset_path),
                checkpoint_path=str(checkpoint_path),
                student_model_name=model_path,
                backup=True
            )

            print(f"✓ {split}: {stats['processed']}/{stats['total']} samples processed")

        print(f"\n{'='*70}")
        print(f"✅ Stage 3.2 Complete!")
        print(f"{'='*70}")
        print(f"  1. Weight Generator checkpoint: {checkpoint_path}")
        print(f"  2. Token weights applied to all datasets")
        print(f"{'='*70}\n")
        return True

    except Exception as e:
        print(f"\n❌ Stage 3.2 Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def stage_4_progressive_training(args):

    print_stage_header("Stage 4", "Progressive Student Training (TDAS + KPOD)")

    if not HAS_PROGRESSIVE_TRAINER:
        print("❌ progressive_trainer.py not found. Cannot run Stage 4.")
        print("   Please implement progressive_trainer.py first.\n")
        return False


    if args.skip_completed and check_stage_completed('4', args.dataset, args):
        print(f"✓ Stage 4 already completed, skipping...\n")
        return True


    train_path = get_processed_file_path(args.dataset, 'train')
    val_path = get_processed_file_path(args.dataset, 'val')

    try:
        with open(train_path, 'r', encoding='utf-8') as f:
            train_dataset = json.load(f)
        print(f"✓ Loaded training data: {len(train_dataset)} samples")

        with open(val_path, 'r', encoding='utf-8') as f:
            val_dataset = json.load(f)
        print(f"✓ Loaded validation data: {len(val_dataset)} samples")


        print("\n🔍 Validating dataset completeness...")
        train_required = ['difficulty', 'token_weights', 'history', 'preference', 'result']
        sample = train_dataset[0]
        missing_fields = [f for f in train_required if f not in sample]
        if missing_fields:
            print(f"❌ Training set missing required fields: {missing_fields}")
            print(f"   Please complete preprocessing stages first:")
            if 'difficulty' in missing_fields:
                print(f"     - Stage 3.1: Difficulty encoding")
                print(f"       python main.py --dataset {args.dataset} --stage 3.1")
            if 'token_weights' in missing_fields:
                print(f"     - Stage 3.2: Train Weight Generator")
                print(f"       python main.py --dataset {args.dataset} --stage 3.2")
                print(f"     - Stage 3.3: Apply token weights")
                print(f"       python apply_token_weights.py --dataset {args.dataset}")
            return False
        print(f"✓ Training set validation passed")


        val_required = ['token_weights', 'history', 'preference', 'result']
        val_sample = val_dataset[0]
        val_missing = [f for f in val_required if f not in val_sample]
        if val_missing:
            print(f"❌ Validation set missing required fields: {val_missing}")
            print(f"   Please complete preprocessing stages first:")
            if 'token_weights' in val_missing:
                print(f"     - Stage 3.2: Train Weight Generator")
                print(f"       python main.py --dataset {args.dataset} --stage 3.2")
                print(f"     - Stage 3.3: Apply token weights")
                print(f"       python apply_token_weights.py --dataset {args.dataset}")
            return False
        print(f"✓ Validation set validation passed")

        print(f"✓ Dataset validation complete")

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False


    model_path = get_model_path(args.student_model)


    import os
    import torch


    is_torchrun = 'LOCAL_RANK' in os.environ


    num_gpus = torch.cuda.device_count()


    use_fsdp = False
    if args.use_fsdp:
        if not is_torchrun:
            print("\n❌ FSDP requires torchrun distributed launch.")
            return False
        use_fsdp = True
        use_ddp = False
        auto_reason = "user specified --use-fsdp"
    elif not args.no_auto_ddp:
        if is_torchrun:

            use_ddp = True
            auto_reason = "torchrun detected"
        elif num_gpus == 1:

            use_ddp = True
            auto_reason = "single GPU (DDP for gradient checkpointing)"
        elif args.use_ddp:

            use_ddp = True
            auto_reason = "user specified --use-ddp"
        else:

            use_ddp = False
            auto_reason = "multi-GPU without torchrun (will use DataParallel)"
            print(f"\n⚠️  WARNING: Detected {num_gpus} GPUs but not using DDP")
            print(f"   CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node={num_gpus} main.py --dataset {args.dataset} --stage 4 --train-batch-size {args.train_batch_size}")
    else:
        use_ddp = args.use_ddp
        auto_reason = "auto-detection disabled"

    print(f"\n🎯 Training configuration:")
    print(f"  Student model: {args.student_model}")
    print(f"  Model path: {model_path}")
    print(f"  Epochs: {args.train_epochs}")
    print(f"  Batch size: {args.train_batch_size}")
    print(f"  Learning rate: {args.train_lr}")
    print(f"  Initial budget: {args.initial_budget}")
    print(f"  Growth power: {args.growth_power}")
    print(f"  Sample strategy: {args.sample_strategy}")
    print(f"  CoT supervision: Enabled")
    print(f"  Weighted loss: Enabled")
    if args.no_early_stopping:
        print(f"  Early stopping: Disabled")
    else:
        patience_note = args.early_stopping_patience if args.early_stopping_patience is not None else "auto"
        print(f"  Early stopping: Enabled (patience={patience_note})")
    backend = "FSDP" if use_fsdp else ("DDP" if use_ddp else "DataParallel")
    print(f"  Distributed backend: {backend} ({auto_reason})")
    print(f"  Gradient Checkpointing: {'Enabled (distributed)' if (use_ddp or use_fsdp) else 'Disabled (DataParallel)'}")
    print(f"  Save directory: {args.train_save_dir}\n")


    scheduler = CurriculumScheduler(
        total_epochs=args.train_epochs,
        initial_budget=args.initial_budget,
        growth_power=args.growth_power
    )
    sample_strategy = args.sample_strategy


    trainer = StudentTrainer(
        student_model_name=model_path,
        curriculum_scheduler=scheduler,
        use_weighted_loss=True,
        use_cot=True,
        use_sample_difficulty_weighting=True,
        use_ddp=use_ddp,
        use_fsdp=use_fsdp
    )

    try:
        online_wg_lr = args.online_wg_lr
        if online_wg_lr is None:
            online_wg_lr = args.wg_lr * 0.5

        online_wg_checkpoint = args.online_wg_checkpoint
        if online_wg_checkpoint is None:
            online_wg_checkpoint = str(Path(args.wg_save_dir) / f"weight_generator_epoch{args.wg_epochs}.pt")

        online_wg_save_dir = args.online_wg_save_dir
        if online_wg_save_dir is None:
            online_wg_save_dir = str(get_online_weight_generator_path(args.dataset))


        trainer.progressive_train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            num_epochs=args.train_epochs,
            batch_size=args.train_batch_size,
            learning_rate=args.train_lr,
            save_dir=args.train_save_dir,
            sample_selection_strategy=sample_strategy,
            disable_curriculum=False,
            enable_early_stopping=not args.no_early_stopping,
            early_stopping_patience=args.early_stopping_patience,
            online_wg=args.online_wg,
            online_wg_checkpoint=online_wg_checkpoint,
            online_wg_save_dir=online_wg_save_dir,
            online_wg_warmup_epochs=args.online_wg_warmup,
            online_wg_lr=online_wg_lr,
            online_wg_batch_size=args.online_wg_batch_size,
            online_wg_alpha=args.wg_alpha,
            online_wg_temperature=args.wg_temperature,
            online_wg_stopword_penalty=args.wg_stopword_penalty,
            online_wg_stopword_margin=args.wg_stopword_margin,
            online_wg_stopword_quantile=args.wg_stopword_quantile,
            online_wg_stopword_quantile_cap=args.wg_stopword_quantile_cap
        )


        final_model_path = Path(args.train_save_dir) / 'final'
        trainer.save_model(str(final_model_path))

        print(f"\n✅ Stage 4 Complete!")
        print(f"  Final model saved to: {final_model_path}\n")
        return True

    except Exception as e:
        print(f"\n❌ Stage 4 Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if 'trainer' in locals() and hasattr(trainer, "cleanup_distributed"):
            trainer.cleanup_distributed()


def parse_args():
    parser = argparse.ArgumentParser(
        description="RecCoT-SD Training Pipeline (Stage 3-5)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )


    parser.add_argument(
        '--dataset',
        type=str,
        default='ml100k',
        choices=['ml100k', 'ml1m', 'electronics', 'movies'],
        help='Dataset name'
    )

    parser.add_argument(
        '--stage',
        type=str,
        default='all',
        choices=['all', '3.1', '3.2', '3', '4'],
        help='Run specific stage only (3 = 3.1 + 3.2, all = 3.1 + 3.2 + 4)'
    )

    parser.add_argument(
        '--skip-completed',
        action='store_true',
        help='Skip stages that are already completed'
    )


    parser.add_argument(
        '--student-model',
        type=str,
        default=get_student_model_name(),
        help='Student model name (will be resolved to local path via config.py)'
    )


    parser.add_argument(
        '--diff-alpha',
        type=float,
        default=0.1,
        help='Composition difficulty weight (cluster variance)'
    )

    parser.add_argument(
        '--diff-beta',
        type=float,
        default=0.3,
        help='Relevance difficulty weight (distance to cluster center)'
    )

    parser.add_argument(
        '--diff-gamma',
        type=float,
        default=0.4,
        help='Training difficulty weight (token-level, from KPOD)'
    )

    parser.add_argument(
        '--no-normalize',
        action='store_true',
        help='Skip difficulty normalization to [0, 1]'
    )

    parser.add_argument(
        '--no-analyze',
        action='store_true',
        help='Skip difficulty distribution analysis and visualization'
    )


    parser.add_argument(
        '--wg-epochs',
        type=int,
        default=15,
        help='Weight Generator training epochs'
    )

    parser.add_argument(
        '--wg-batch-size',
        type=int,
        default=16,
        help='Weight Generator batch size'
    )

    parser.add_argument(
        '--wg-lr',
        type=float,
        default=5e-5,
        help='Weight Generator learning rate'
    )

    parser.add_argument(
        '--wg-alpha',
        type=float,
        default=0.5,
        help='Mask ratio loss weight (balance between answer prediction and mask ratio)'
    )

    parser.add_argument(
        '--wg-temperature',
        type=float,
        default=1.0,
        help='Gumbel-Softmax temperature for mask sampling'
    )

    parser.add_argument(
        '--wg-stopword-penalty',
        type=float,
        default=1.0,
        help='Stopword penalty weight (encourage lower stopword weights)'
    )

    parser.add_argument(
        '--wg-stopword-margin',
        type=float,
        default=0.2,
        help='Stopword margin vs content mean (for penalty)'
    )

    parser.add_argument(
        '--wg-stopword-quantile',
        type=float,
        default=0.8,
        help='Stopword quantile (e.g., 0.8 means 80%% quantile)'
    )

    parser.add_argument(
        '--wg-stopword-quantile-cap',
        type=float,
        default=0.25,
        help='Stopword quantile cap (target upper bound)'
    )

    parser.add_argument(
        '--wg-save-dir',
        type=str,
        default=None,
        help='Weight Generator save directory (default: from config based on dataset)'
    )


    parser.add_argument(
        '--train-epochs',
        type=int,
        default=30,
        help='Progressive training epochs'
    )

    parser.add_argument(
        '--train-batch-size',
        type=int,
        default=8,
        help='Training batch size'
    )

    parser.add_argument(
        '--train-lr',
        type=float,
        default=5e-5,
        help='Training learning rate'
    )

    parser.add_argument(
        '--initial-budget',
        type=float,
        default=0.2,
        help='Initial difficulty budget (fraction of samples, e.g., 0.2 = start with 20%% easiest)'
    )

    parser.add_argument(
        '--growth-power',
        type=float,
        default=2.0,
        help='Budget growth power (curriculum pacing, higher = slower growth)'
    )

    parser.add_argument(
        '--sample-strategy',
        type=str,
        default='threshold',
        choices=['threshold', 'top_k', 'sampling'],
        help='Sample selection strategy for curriculum learning'
    )

    parser.add_argument(
        '--no-early-stopping',
        action='store_true',
        default=True,
        help='Disable early stopping during training'
    )

    parser.add_argument(
        '--early-stopping-patience',
        type=int,
        default=None,
        help='Early stopping patience (epochs with no improvement). None = auto'
    )

    parser.add_argument(
        '--online-wg',
        action='store_true',
        help='Enable online WG update during Stage 4'
    )

    parser.add_argument(
        '--online-wg-warmup',
        type=int,
        default=1,
        help='Warmup epochs before online WG updates'
    )

    parser.add_argument(
        '--online-wg-lr',
        type=float,
        default=None,
        help='Online WG learning rate (default: wg_lr * 0.5)'
    )

    parser.add_argument(
        '--online-wg-batch-size',
        type=int,
        default=4,
        help='Online WG batch size (used as grad accumulation steps)'
    )

    parser.add_argument(
        '--online-wg-checkpoint',
        type=str,
        default=None,
        help='Initial WG checkpoint path for online update'
    )

    parser.add_argument(
        '--online-wg-save-dir',
        type=str,
        default=None,
        help='Online WG checkpoint save dir (default: <train_save_dir>/wg_online)'
    )

    parser.add_argument(
        '--use-ddp',
        action='store_true',
        help='Use DistributedDataParallel instead of DataParallel (auto-enabled if torchrun detected)'
    )

    parser.add_argument(
        '--use-fsdp',
        action='store_true',
        help='Use FullyShardedDataParallel to shard model/grad/optimizer states across GPUs (requires torchrun)'
    )

    parser.add_argument(
        '--no-auto-ddp',
        action='store_true',
        help='Disable automatic DDP detection (force DataParallel even with single GPU)'
    )

    parser.add_argument(
        '--train-save-dir',
        type=str,
        default=None,
        help='Student model save directory (default: from config based on dataset)'
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()


    if args.wg_save_dir is None:
        args.wg_save_dir = get_weight_generator_path(args.dataset)
    if args.train_save_dir is None:
        args.train_save_dir = get_train_save_path(args.dataset)
        if args.train_save_dir is None:
            base_save_dir = get_train_save_path(args.dataset)
            args.train_save_dir = base_save_dir

    print_header("RecCoT-SD Training Pipeline (Stage 3-5)")
    print(f"  Dataset: {args.dataset}")
    print(f"  Running Stage: {args.stage}")
    print(f"  Skip Completed: {args.skip_completed}")
    print(f"  Student Model: {args.student_model}")


    if not check_prerequisites(args.dataset, require_preference=True):
        print("\n" + "="*80)
        print("❌ Prerequisites check failed!")
        print("="*80)
        print("\nPlease complete Stage 1-2 first by running:")
        print(f"  python run.py --dataset {args.dataset} --cot cluster")
        print("\nThis will:")
        print("  1. Process the dataset and generate templates")
        print("  2. Perform K-means clustering on samples")
        print("  3. Generate Chain-of-Thought explanations using LLM")
        print("="*80 + "\n")
        return 1


    stages_to_run = []

    if args.stage == 'all':
        stages_to_run = ['3.1', '3.2', '4']
    elif args.stage == '3':
        stages_to_run = ['3.1', '3.2']
    else:
        stages_to_run = [args.stage]

    print(f"  Stages to run: {', '.join(stages_to_run)}")

    try:

        if '3.1' in stages_to_run:
            success = stage_3_1_encode_difficulty(args)
            if not success:
                print("\n❌ Pipeline stopped at Stage 3.1")
                return 1


        if '3.2' in stages_to_run:
            success = stage_3_2_train_weight_generator(args)
            if not success:
                print("\n❌ Pipeline stopped at Stage 3.2")
                return 1


        if '4' in stages_to_run:
            success = stage_4_progressive_training(args)
            if not success:
                print("\n❌ Pipeline stopped at Stage 4")
                return 1


        print_header("✅ Training Pipeline Complete!")

        print("Completed Stages:")
        for stage in stages_to_run:
            print(f"  ✓ Stage {stage}")
        print("\n" + "="*80 + "\n")

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        return 130

    except Exception as e:
        print(f"\n\n❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
