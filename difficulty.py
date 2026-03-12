

import json
import shutil
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from config import get_processed_file_path
from token_weight import RationaleTokenWeighting


class DifficultyEncoder:


    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.3,
        gamma: float = 0.4,
        normalization: str = 'minmax'
    ):


        total_weight = alpha + beta + gamma
        if not np.isclose(total_weight, 1.0):
            print(f"⚠️  Warning: Weights sum to {total_weight}, normalizing...")
            alpha = alpha / total_weight
            beta = beta / total_weight
            gamma = gamma / total_weight

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.normalization = normalization


        self.cluster_stats = {}

        print(f"\n{'='*70}")
        print("Initializing DifficultyEncoder (TDAS with Inter-Cluster)")
        print(f"{'='*70}")
        print(f"  Weights: α={alpha:.2f} (comp), β={beta:.2f} (rel), γ={gamma:.2f} (inter-cluster)")
        print(f"  Normalization: {normalization}")
        print(f"{'='*70}\n")

    def _load_cluster_stats_from_metadata(self, demo_file_path: str) -> Optional[Dict]:

        from pathlib import Path
        import json

        demo_file = Path(demo_file_path)

        if not demo_file.exists():
            print(f"⚠️  Warning: Demo file not found: {demo_file}")
            return None

        try:
            with open(demo_file, 'r', encoding='utf-8') as f:
                data = json.load(f)


            if isinstance(data, dict) and 'metadata' in data:
                metadata = data['metadata']

                if 'cluster_stats' in metadata:
                    cluster_stats = metadata['cluster_stats']


                    if 'clusters' in cluster_stats:
                        clusters = cluster_stats['clusters']
                        cluster_stats['clusters'] = {
                            int(k): v for k, v in clusters.items()
                        }

                    print(f"✓ Loaded cluster statistics from {demo_file.name}")
                    return cluster_stats
                else:
                    print(f"⚠️  Warning: 'cluster_stats' not found in metadata")
                    return None
            else:
                print(f"⚠️  Warning: metadata not available")
                return None

        except Exception as e:
            print(f"❌ Error loading cluster stats from metadata: {e}")
            return None

    def compute_composition_difficulty(self, cluster_id: int) -> float:

        cluster_id = int(cluster_id)

        if cluster_id not in self.cluster_stats:
            raise KeyError(
                f"Cluster {cluster_id} not found in cluster_stats.\n"
                f"Available clusters: {list(self.cluster_stats.keys())}\n"
                f"Please ensure clustering (Stage 2) was completed successfully."
            )

        return float(self.cluster_stats[cluster_id]['variance'])

    def compute_relevance_difficulty(self, sample: Dict) -> float:


        cluster_id = int(sample['cluster_id'])
        distance = float(sample['distance_to_center'])

        if cluster_id not in self.cluster_stats:
            raise KeyError(
                f"Cluster {cluster_id} not found in cluster_stats.\n"
                f"Available clusters: {list(self.cluster_stats.keys())}\n"
                f"Sample index: {sample.get('original_index', 'unknown')}"
            )


        max_distance = self.cluster_stats[cluster_id]['max_distance']
        if max_distance > 0:
            d_rel = distance / max_distance
        else:

            print(f"⚠️  Cluster {cluster_id} has max_distance=0 (single/identical samples)")
            d_rel = 0.0

        return float(d_rel)

    def compute_inter_cluster_difficulty(self, sample: Dict) -> float:

        min_dist_to_other = sample['min_distance_to_other_cluster']


        if not hasattr(self, 'global_stats') or self.global_stats is None:
            raise ValueError("global_stats not loaded. Please ensure demo file contains global statistics.")

        min_val = self.global_stats['min_inter_cluster_dist']
        max_val = self.global_stats['max_inter_cluster_dist']


        if max_val > min_val:
            normalized_dist = (min_dist_to_other - min_val) / (max_val - min_val)
        else:
            normalized_dist = 0.5


        d_inter_cluster = 1.0 - normalized_dist


        d_inter_cluster = np.clip(d_inter_cluster, 0.0, 1.0)

        return float(d_inter_cluster)

    def compute_total_difficulty(
        self,
        sample: Dict
    ) -> Dict[str, float]:


        d_comp = self.compute_composition_difficulty(sample['cluster_id'])
        d_rel = self.compute_relevance_difficulty(sample)
        d_inter_cluster = self.compute_inter_cluster_difficulty(sample)


        d_total = float(self.alpha * d_comp + self.beta * d_rel + self.gamma * d_inter_cluster)

        return {
            'composition': d_comp,
            'relevance': d_rel,
            'inter_cluster': d_inter_cluster,
            'total': d_total
        }

    def normalize_difficulties(self, dataset: List[Dict]) -> List[Dict]:

        print("🔄 Normalizing difficulties...")


        difficulties = {
            'composition': [],
            'relevance': [],
            'inter_cluster': [],
            'total': []
        }

        for sample in dataset:
            diff = sample.get('difficulty', {})
            for key in difficulties:
                if key in diff:
                    difficulties[key].append(diff[key])


        if self.normalization == 'minmax':
            for key in difficulties:
                if difficulties[key]:
                    min_val = np.min(difficulties[key])
                    max_val = np.max(difficulties[key])
                    print(f"  {key}: min={min_val:.4f}, max={max_val:.4f}")


                    if max_val > min_val:
                        for sample in dataset:
                            if 'difficulty' in sample and key in sample['difficulty']:
                                old_val = sample['difficulty'][key]
                                new_val = (old_val - min_val) / (max_val - min_val)
                                sample['difficulty'][key] = float(new_val)

        elif self.normalization == 'zscore':
            for key in difficulties:
                if difficulties[key]:
                    mean_val = np.mean(difficulties[key])
                    std_val = np.std(difficulties[key])
                    print(f"  {key}: mean={mean_val:.4f}, std={std_val:.4f}")


                    if std_val > 0:
                        for sample in dataset:
                            if 'difficulty' in sample and key in sample['difficulty']:
                                old_val = sample['difficulty'][key]
                                z_score = (old_val - mean_val) / std_val

                                new_val = 1 / (1 + np.exp(-z_score))
                                sample['difficulty'][key] = float(new_val)

        print("✓ Normalization complete")
        return dataset

    def encode_dataset(
        self,
        dataset_path: str,
        demo_file_path: str,
        output_path: str = None,
        normalize: bool = True
    ) -> List[Dict]:

        dataset_path = Path(dataset_path)
        if output_path is None:
            output_path = dataset_path
        else:
            output_path = Path(output_path)

        print(f"\n{'='*70}")
        print("Encoding Dataset Difficulty (TDAS with Inter-Cluster)")
        print(f"{'='*70}")
        print(f"  Input: {dataset_path}")
        print(f"  Demo file: {demo_file_path}")
        print(f"  Output: {output_path}")
        print(f"{'='*70}\n")


        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            print(f"✓ Loaded {len(dataset)} samples")
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            raise


        required_fields = [
            'cluster_id',
            'distance_to_center',
            'min_distance_to_other_cluster',
            'embedding'
        ]

        if dataset:
            sample = dataset[0]
            missing = [f for f in required_fields if f not in sample]
            if missing:
                raise ValueError(
                    f"❌ Dataset missing required fields: {missing}\n"
                    f"   Please re-run clustering:\n"
                    f"   python cluster.py --dataset [dataset_name]"
                )


        print("\n📊 Loading cluster statistics from demo file...")
        cluster_stats = self._load_cluster_stats_from_metadata(demo_file_path)

        if cluster_stats is None:
            raise ValueError(
                f"❌ Failed to load cluster statistics from {demo_file_path}\n"
                f"   Please ensure you have run clustering with the new version:\n"
                f"   python cluster.py --dataset [dataset_name]"
            )


        if 'global' not in cluster_stats:
            raise ValueError(
                "cluster_stats missing 'global' field.\n"
                "Please re-run clustering to generate global statistics:\n"
                "  python cluster.py --dataset [dataset_name]"
            )
        self.global_stats = cluster_stats['global']


        if 'clusters' in cluster_stats:
            self.cluster_stats = cluster_stats['clusters']
        else:
            self.cluster_stats = cluster_stats


        if self.cluster_stats:
            self.cluster_stats = {
                int(k) if not isinstance(k, int) else k: v
                for k, v in self.cluster_stats.items()
            }


            for cid, stats in self.cluster_stats.items():
                required_fields = ['variance', 'max_distance']
                missing = [f for f in required_fields if f not in stats]
                if missing:
                    raise ValueError(
                        f"Cluster {cid} missing required fields: {missing}\n"
                        f"Please re-run clustering to regenerate complete statistics."
                    )


        if 'min_inter_cluster_dist' not in self.global_stats or 'max_inter_cluster_dist' not in self.global_stats:
            raise ValueError(
                "global_stats missing 'min_inter_cluster_dist' and/or 'max_inter_cluster_dist' fields.\n"
                "These fields are required for computing inter-cluster difficulty.\n"
                "\n"
                "Please re-run clustering to generate complete statistics:\n"
                "  python cluster.py --dataset [dataset_name]\n"
                "\n"
                "Expected cluster.json structure:\n"
                "  metadata.cluster_stats.global: {\n"
                "    'min_inter_cluster_dist': float,\n"
                "    'max_inter_cluster_dist': float,\n"
                "    ...\n"
                "  }"
            )


        print("\n🔄 Computing difficulties...")
        for i, sample in enumerate(tqdm(dataset, desc="Encoding difficulties")):
            difficulty = self.compute_total_difficulty(sample)
            dataset[i]['difficulty'] = difficulty


        if normalize:
            dataset = self.normalize_difficulties(dataset)


        print(f"\n💾 Saving to {output_path}...")
        try:
            temp_path = output_path.with_suffix('.tmp.json')
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
            shutil.move(str(temp_path), output_path)
            print(f"✓ Saved successfully")
        except Exception as e:
            print(f"❌ Error saving: {e}")
            raise

        print(f"\n{'='*70}")
        print("✅ Difficulty Encoding Complete!")
        print(f"{'='*70}\n")

        return dataset


def analyze_difficulty_distribution(dataset_path: str):

    print(f"\n{'='*70}")
    print("Analyzing Difficulty Distribution")
    print(f"{'='*70}")

    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return


    difficulties = {
        'composition': [],
        'relevance': [],
        'inter_cluster': [],
        'total': []
    }

    for sample in dataset:
        diff = sample.get('difficulty', {})
        for key in difficulties:
            if key in diff:
                difficulties[key].append(diff[key])


    print("\nDifficulty Statistics:")
    for key, values in difficulties.items():
        if values:
            print(f"\n{key.capitalize()}:")
            print(f"  Mean: {np.mean(values):.4f}")
            print(f"  Std: {np.std(values):.4f}")
            print(f"  Min: {np.min(values):.4f}")
            print(f"  Max: {np.max(values):.4f}")
            print(f"  Median: {np.median(values):.4f}")


            q25, q50, q75 = np.percentile(values, [25, 50, 75])
            print(f"  Q1={q25:.4f}, Q2={q50:.4f}, Q3={q75:.4f}")


    print("\nDifficulty Levels (based on total difficulty):")
    total_difficulties = difficulties['total']
    if total_difficulties:
        easy = sum(d < 0.33 for d in total_difficulties)
        medium = sum(0.33 <= d < 0.67 for d in total_difficulties)
        hard = sum(d >= 0.67 for d in total_difficulties)

        total = len(total_difficulties)
        print(f"  Easy (< 0.33): {easy} ({easy/total*100:.1f}%)")
        print(f"  Medium (0.33-0.67): {medium} ({medium/total*100:.1f}%)")
        print(f"  Hard (>= 0.67): {hard} ({hard/total*100:.1f}%)")


    print("\nDifficulty by Cluster:")
    cluster_difficulties = defaultdict(list)
    for sample in dataset:
        cluster_id = sample.get('cluster_id')
        total_diff = sample.get('difficulty', {}).get('total')
        if cluster_id is not None and total_diff is not None:
            cluster_difficulties[cluster_id].append(total_diff)

    for cluster_id in sorted(cluster_difficulties.keys()):
        diffs = cluster_difficulties[cluster_id]
        print(f"  Cluster {cluster_id}: mean={np.mean(diffs):.4f}, "
              f"std={np.std(diffs):.4f}, "
              f"size={len(diffs)}")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stage 3.1: Difficulty Encoding (TDAS with Inter-Cluster)")
    parser.add_argument('--dataset', type=str, default='ml100k', help='Dataset name')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'],
                        help='Data split to process')
    parser.add_argument('--alpha', type=float, default=0.3, help='Composition difficulty weight')
    parser.add_argument('--beta', type=float, default=0.3, help='Relevance difficulty weight')
    parser.add_argument('--gamma', type=float, default=0.4, help='Inter-cluster difficulty weight')
    parser.add_argument('--model', type=str, default='google/flan-t5-large',
                        help='Student model name (kept for backward compatibility)')
    parser.add_argument('--no-normalize', action='store_true', help='Skip normalization')
    parser.add_argument('--analyze', action='store_true', help='Analyze difficulty distribution')

    args = parser.parse_args()


    file_path = get_processed_file_path(args.dataset, args.split)

    print(f"\n{'='*80}")
    print("Stage 3.1: Difficulty Encoding (TDAS with Inter-Cluster)")
    print(f"{'='*80}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Split: {args.split}")
    print(f"  File: {file_path}")
    print(f"{'='*80}\n")


    encoder = DifficultyEncoder(
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma
    )


    try:
        demo_file = get_processed_file_path(args.dataset, 'cluster')
        encoder.encode_dataset(
            dataset_path=file_path,
            demo_file_path=str(demo_file),
            normalize=not args.no_normalize
        )


        if args.analyze:
            analyze_difficulty_distribution(file_path)

    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()