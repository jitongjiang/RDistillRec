

import numpy as np
import torch
import random
import json
import os
import argparse
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

import config
from config import *

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class Clusterer:


    def __init__(self, num_clusters=6, encoder_name="all-MiniLM-L6-v2",
                 seed=SEED, n_init=10, max_iter=300):

        self.num_clusters = num_clusters
        self.encoder_name = encoder_name
        self.seed = seed
        self.n_init = n_init
        self.max_iter = max_iter


        print(f"📂 Loading sentence encoder: {encoder_name}")
        self.encoder = SentenceTransformer(encoder_name)


        self.kmeans = KMeans(
            n_clusters=num_clusters,
            random_state=seed,
            n_init=n_init,
            max_iter=max_iter,
            verbose=0
        )

        self.cluster_centers = None
        self.fitted = False

    def encode_texts(self, texts, batch_size=32, show_progress_bar=True):

        print(f"\n🔄 Encoding {len(texts)} samples...")
        embeddings = self.encoder.encode(
            texts,
            show_progress_bar=show_progress_bar,
            batch_size=batch_size,
            convert_to_numpy=True
        )
        print(f"✓ Embeddings shape: {embeddings.shape}")
        return embeddings

    def fit(self, embeddings):

        print(f"\n🎯 Performing K-means clustering (k={self.num_clusters})...")
        self.kmeans.fit(embeddings)
        self.cluster_centers = self.kmeans.cluster_centers_
        self.fitted = True
        print(f"✓ Clustering completed")
        return self

    def predict_and_compute_distances(self, embeddings):

        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        print(f"📏 Computing labels and distances...")


        distances_to_all_centers = self.kmeans.transform(embeddings)


        labels = np.argmin(distances_to_all_centers, axis=1)


        distance_to_center = distances_to_all_centers[np.arange(len(labels)), labels]


        min_distance_to_other_cluster = np.zeros(len(labels))
        for i in range(len(labels)):
            cluster_id = labels[i]

            other_distances = [
                distances_to_all_centers[i][j]
                for j in range(self.num_clusters)
                if j != cluster_id
            ]

            if other_distances:
                min_distance_to_other_cluster[i] = min(other_distances)
            else:

                min_distance_to_other_cluster[i] = distance_to_center[i] * 2.0

        distances_info = {
            'distances_to_all_centers': distances_to_all_centers,
            'distance_to_center': distance_to_center,
            'min_distance_to_other_cluster': min_distance_to_other_cluster
        }

        return labels, distances_info

    def compute_cluster_statistics(self, embeddings, labels, distances_info):

        cluster_stats = {}
        all_distances = []


        for cluster_id in range(self.num_clusters):

            cluster_indices = np.where(labels == cluster_id)[0]

            if len(cluster_indices) == 0:
                continue


            distances = distances_info['distance_to_center'][cluster_indices]
            all_distances.extend(distances)


            variance = float(np.var(distances))

            cluster_stats[int(cluster_id)] = {
                'size': len(cluster_indices),
                'mean_distance': float(np.mean(distances)),
                'std_distance': float(np.std(distances)),
                'min_distance': float(np.min(distances)),
                'max_distance': float(np.max(distances)),
                'median_distance': float(np.median(distances)),
                'variance': variance
            }


        all_distances = np.array(all_distances)


        min_inter_cluster_dists = distances_info['min_distance_to_other_cluster']

        global_stats = {
            'total_samples': len(labels),
            'num_clusters': self.num_clusters,
            'global_mean_distance': float(np.mean(all_distances)),
            'global_std_distance': float(np.std(all_distances)),
            'global_max_distance': float(np.max(all_distances)),

            'min_inter_cluster_dist': float(np.min(min_inter_cluster_dists)),
            'max_inter_cluster_dist': float(np.max(min_inter_cluster_dists))
        }

        return {
            'clusters': cluster_stats,
            'global': global_stats
        }

    def extract_demonstrations(self, enriched_dataset):


        clustered_data = defaultdict(list)
        for sample in enriched_dataset:
            cluster_id = sample['cluster_id']
            clustered_data[cluster_id].append(sample)


        for cluster_id in clustered_data:
            clustered_data[cluster_id].sort(key=lambda x: x['distance_to_center'])


        demos = []

        for cluster_id in range(self.num_clusters):
            if cluster_id not in clustered_data or len(clustered_data[cluster_id]) == 0:
                continue


            representative = clustered_data[cluster_id][0]

            demos.append({
                'cluster_id': int(cluster_id),
                'demo_history': representative['history'],
                'distance': float(representative['distance_to_center']),
                'original_index': int(representative['original_index']),
                'cluster_center': self.cluster_centers[cluster_id].tolist()
            })

        return demos


class ClusterVisualizer:


    def __init__(self, figsize=(16, 6), seed=SEED):

        self.figsize = figsize
        self.seed = seed

    def visualize(self, embeddings, labels, distances_info, num_clusters, save_path=None):

        try:
            from sklearn.decomposition import PCA
        except ImportError:
            print("⚠️  Warning: sklearn not available for visualization")
            return

        print("  Performing PCA dimensionality reduction...")

        pca = PCA(n_components=2, random_state=self.seed)
        embeddings_2d = pca.fit_transform(embeddings)


        fig, axes = plt.subplots(1, 2, figsize=self.figsize)


        ax1 = axes[0]
        scatter = ax1.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=labels,
            cmap='tab10',
            alpha=0.6,
            s=30,
            edgecolors='none'
        )
        ax1.set_title(f'Cluster Distribution (k={num_clusters})', fontsize=14, fontweight='bold')
        ax1.set_xlabel('PCA Component 1')
        ax1.set_ylabel('PCA Component 2')
        ax1.grid(alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax1, label='Cluster ID')
        cbar.set_ticks(range(num_clusters))


        ax2 = axes[1]
        cluster_distances = [[] for _ in range(num_clusters)]
        for i, label in enumerate(labels):
            cluster_distances[label].append(
                distances_info['distances_to_all_centers'][i][label]
            )

        positions = range(num_clusters)
        bp = ax2.boxplot(
            cluster_distances,
            positions=positions,
            widths=0.6,
            patch_artist=True,
            boxprops=dict(facecolor='lightblue', alpha=0.7),
            medianprops=dict(color='red', linewidth=2)
        )
        ax2.set_title('Distance to Center Distribution per Cluster', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Cluster ID')
        ax2.set_ylabel('Distance to Centroid')
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_xticks(range(num_clusters))

        plt.tight_layout()

        if save_path:
            try:
                os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"  ✓ Visualization saved to: {save_path}")
            except Exception as e:
                print(f"  ⚠️  Warning: Failed to save visualization: {e}")

        plt.close()

def save_clustering_results(enriched_dataset, demonstrations, metadata,
                            cluster_stats, dataset_file, demo_file):


    print(f"\n💾 Saving results to: {dataset_file}")
    try:
        with open(dataset_file, 'w', encoding='utf-8') as f:
            json.dump(enriched_dataset, f, ensure_ascii=False, indent=2)

        file_size = os.path.getsize(dataset_file) / (1024 * 1024)
        print(f"✓ Saved {len(enriched_dataset)} samples (file size: {file_size:.2f} MB)")
    except Exception as e:
        print(f"❌ Error saving dataset file: {e}")
        raise


    if demo_file:
        print(f"\n💾 Saving demonstrations with metadata to: {demo_file}")
        try:
            os.makedirs(os.path.dirname(demo_file) if os.path.dirname(demo_file) else ".", exist_ok=True)


            output_data = {
                'demonstrations': demonstrations,
                'metadata': {
                    **metadata,
                    'cluster_stats': cluster_stats
                }
            }

            with open(demo_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"✓ Saved {len(demonstrations)} demonstrations with metadata")
            print(f"  Metadata includes: {', '.join(output_data['metadata'].keys())}")
        except Exception as e:
            print(f"❌ Error saving demo file: {e}")
            raise


def print_cluster_statistics(cluster_stats, enriched_dataset):


    clustered_data = defaultdict(list)
    for sample in enriched_dataset:
        cluster_id = sample['cluster_id']
        clustered_data[cluster_id].append(sample)


    for cluster_id in clustered_data:
        clustered_data[cluster_id].sort(key=lambda x: x['distance_to_center'])

    print("\n" + "=" * 70)
    print("Clustering Statistics")
    print("=" * 70)


    if 'global' in cluster_stats:
        global_stats = cluster_stats['global']
        print(f"\nGlobal Statistics:")
        print(f"  Total samples: {global_stats['total_samples']}")
        print(f"  Number of clusters: {global_stats['num_clusters']}")
        print(f"  Global mean distance: {global_stats['global_mean_distance']:.4f}")
        print(f"  Global std distance: {global_stats['global_std_distance']:.4f}")
        print(f"  Global max distance: {global_stats['global_max_distance']:.4f}")


    if 'clusters' not in cluster_stats or 'global' not in cluster_stats:
        raise ValueError(
            "Invalid cluster_stats format. Expected keys: 'clusters', 'global'.\n"
            "Please re-run clustering with the latest version."
        )
    clusters = cluster_stats['clusters']

    for cluster_id in sorted(clusters.keys()):
        stats = clusters[cluster_id]
        print(f"\nCluster {cluster_id}:")
        print(f"  Size: {stats['size']} samples ({stats['size'] / len(enriched_dataset) * 100:.1f}%)")
        print(f"  Distance to center: mean={stats['mean_distance']:.4f}, "
              f"std={stats['std_distance']:.4f}, median={stats['median_distance']:.4f}")
        print(f"  Distance range: [{stats['min_distance']:.4f}, {stats['max_distance']:.4f}]")


        if 'variance' in stats:
            print(f"  Variance (composition difficulty): {stats['variance']:.4f}")


        print(f"  Most representative samples (closest to center):")
        for i in range(min(3, len(clustered_data[cluster_id]))):
            sample = clustered_data[cluster_id][i]
            history = sample['history']
            history_preview = history[:80]
            if len(history) > 80:
                history_preview += "..."
            print(f"    {i + 1}. Distance={sample['distance_to_center']:.4f}")
            print(f"       Preview: {history_preview}")


def main(
    dataset: str = 'ml100k',
    num_clusters: int = 6,
    encoder: str = 'all-MiniLM-L6-v2',
    random_seed: int = None,
    visualize: bool = False,
    text_field: str = 'history'
):


    seed = random_seed if random_seed is not None else SEED
    set_seed(seed)

    print("=" * 70)
    print("Starting Clustering and Ranking Process")
    print("=" * 70)
    print(f"  Dataset: {dataset}")
    print(f"  Num clusters: {num_clusters}")
    print(f"  Encoder: {encoder}")
    print(f"  Random seed: {seed}")
    print(f"  Text field: {text_field}")
    print("=" * 70)


    clusterer = Clusterer(
        num_clusters=num_clusters,
        encoder_name=encoder,
        seed=seed
    )


    train_file = config.get_processed_file_path(dataset, "train")
    print(f"📂 Reading data from: {train_file}")

    try:
        with open(train_file, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        print(f"✓ Loaded {len(train_data)} samples")
    except FileNotFoundError:
        print(f"❌ Error: {train_file} not found")
        exit(1)
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


    texts = []
    for sample in train_data:
        text = sample[text_field]
        texts.append(text)

    embeddings = clusterer.encode_texts(texts, show_progress_bar=True)


    clusterer.fit(embeddings)


    labels, distances_info = clusterer.predict_and_compute_distances(embeddings)


    enriched_dataset = []
    for sample_idx, sample in enumerate(tqdm(train_data, desc="Processing samples")):
        cluster_id = int(labels[sample_idx])


        enriched_sample = {
            **sample,
            'original_index': sample_idx,
            'cluster_id': cluster_id,
            'distance_to_center': float(distances_info['distance_to_center'][sample_idx]),
            'min_distance_to_other_cluster': float(distances_info['min_distance_to_other_cluster'][sample_idx]),
            'embedding': embeddings[sample_idx].tolist()
        }

        enriched_dataset.append(enriched_sample)


    cluster_stats = clusterer.compute_cluster_statistics(embeddings, labels, distances_info)
    print_cluster_statistics(cluster_stats, enriched_dataset)

    metadata = {
        'num_clusters': clusterer.num_clusters,
        'num_samples': len(train_data),
        'encoder_name': clusterer.encoder_name,
        'seed': clusterer.seed,
        'cluster_stats': cluster_stats
    }


    labels = np.array([sample['cluster_id'] for sample in enriched_dataset])
    distances_info = {
        'distance_to_center': np.array([sample['distance_to_center'] for sample in enriched_dataset]),
        'min_distance_to_other_cluster': np.array([sample['min_distance_to_other_cluster'] for sample in enriched_dataset])
    }


    demonstrations = clusterer.extract_demonstrations(enriched_dataset)


    demo_file = config.get_demo_file_path(dataset, "cluster")
    save_clustering_results(
        enriched_dataset=enriched_dataset,
        demonstrations=demonstrations,
        metadata=metadata,
        cluster_stats=cluster_stats,
        dataset_file=train_file,
        demo_file=demo_file
    )


    if visualize:
        print(f"\n📈 Generating visualization...")
        try:
            visualizer = ClusterVisualizer()
            vis_save_path = str(Path(train_file).parent / "cluster_visualization.png")
            visualizer.visualize(
                embeddings=embeddings,
                labels=labels,
                distances_info=distances_info,
                num_clusters=num_clusters,
                save_path=vis_save_path
            )
        except Exception as e:
            print(f"⚠️  Warning: Visualization failed: {e}")

    print("\n" + "=" * 70)
    print("✅ Clustering Process Complete!")
    print("=" * 70)
