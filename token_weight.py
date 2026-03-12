import os
import json
import shutil
import string
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import random
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from config import get_seed, get_student_model_name

from config import get_processed_file_path


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class WeightGenerator(nn.Module):


    def __init__(self, hidden_size: int = 1024, dropout: float = 0.1):

        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, embeddings):

        return self.mlp(embeddings)


class RationaleTokenWeighting(nn.Module):


    def __init__(
        self,
        student_model_name: str = get_student_model_name(),
        hidden_size: int = 1024,
        alpha: float = 0.1,
        temperature: float = 1.0,
        device: str = None,
        use_ddp: bool = False,
        stopword_penalty: float = 1.0,
        stopword_margin: float = 0.2,
        stopword_quantile: float = 0.8,
        stopword_quantile_cap: float = 0.25,
        stopword_file: Optional[str] = None,
        external_tokenizer=None,
        external_transformer=None
    ):

        super().__init__()


        self.use_ddp = use_ddp
        self.local_rank = 0
        self.world_size = 1

        if self.use_ddp:
            import torch.distributed as dist


            if not dist.is_available():
                raise RuntimeError("DDP requested but torch.distributed not available")


            if 'LOCAL_RANK' not in os.environ:
                raise RuntimeError(
                    "DDP requested but LOCAL_RANK not found. "
                    "Please run with: torchrun --nproc_per_node=N script.py"
                )


            if not dist.is_initialized():
                dist.init_process_group(backend='nccl')

            self.local_rank = int(os.environ['LOCAL_RANK'])
            self.world_size = dist.get_world_size()


            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f'cuda:{self.local_rank}')

            if self.local_rank == 0:
                print(f"\n{'='*70}")
                print("Initializing RationaleTokenWeighting Module")
                print(f"{'='*70}")
                print(f"  Student Model: {student_model_name}")
                print(f"  Hidden Size: {hidden_size}")
                print(f"  Alpha (mask ratio weight): {alpha}")
                print(f"  Temperature: {temperature}")
                print(f"  🔧 DDP initialized: {self.world_size} GPUs")
                print(f"  🎯 Rank {self.local_rank} using device: {self.device}")
        else:

            if device is None:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                self.device = torch.device(device)

            print(f"\n{'='*70}")
            print("Initializing RationaleTokenWeighting Module")
            print(f"{'='*70}")
            print(f"  Student Model: {student_model_name}")
            print(f"  Hidden Size: {hidden_size}")
            print(f"  Alpha (mask ratio weight): {alpha}")
            print(f"  Temperature: {temperature}")
            print(f"  Device: {self.device}")

        self.alpha = alpha
        self.temperature = temperature
        self.stopword_penalty = stopword_penalty
        self.stopword_margin = stopword_margin
        self.stopword_quantile = stopword_quantile
        self.stopword_quantile_cap = stopword_quantile_cap
        self.stopwords = self._load_stopwords(stopword_file)


        if not self.use_ddp or self.local_rank == 0:
            print(f"  Loading tokenizer and model...")
        if external_tokenizer is not None:
            self.tokenizer = external_tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(student_model_name)


        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token


        if external_transformer is not None:
            self.transformer = external_transformer
        else:
            if not self.use_ddp or self.local_rank == 0:
                print(f"  Loading AutoModelForCausalLM...")
            self.transformer = AutoModelForCausalLM.from_pretrained(student_model_name)


        if external_transformer is None:
            self.transformer = self.transformer.to(self.device)


        if external_transformer is None:
            for p in self.transformer.parameters():
                p.requires_grad = False

        if not self.use_ddp or self.local_rank == 0:
            print(f"  ✓ Loaded as CausalLM model on {self.device} (frozen)")


        self.embedding_layer = self.transformer.get_input_embeddings()

        self.hidden_size = self.embedding_layer.embedding_dim


        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        ).to(self.device)


        self.weight_generator = WeightGenerator(
            hidden_size=self.hidden_size,
            dropout=0.1
        ).to(self.device)


        if self.use_ddp:
            from torch.nn.parallel import DistributedDataParallel as DDP

            if self.local_rank == 0:
                print(f"  📦 Wrapping trainable modules with DDP...")

            self.self_attention = DDP(
                self.self_attention,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False
            )

            self.weight_generator = DDP(
                self.weight_generator,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False
            )

            if self.local_rank == 0:
                print(f"  ✓ Trainable modules distributed across {self.world_size} GPUs")

        if not self.use_ddp or self.local_rank == 0:
            print(f"✓ Module initialized successfully")
            print(f"{'='*70}\n")

    def encode_rationale(self, rationale_text: str):


        inputs = self.tokenizer(
            rationale_text,
            return_tensors='pt',
            truncation=True,
            max_length=512
        ).to(self.device)

        input_ids = inputs['input_ids'][0]


        with torch.no_grad():
            embeddings = self.embedding_layer(inputs['input_ids'])[0]


        return embeddings, input_ids

    def _load_stopwords(self, stopword_file: Optional[str]) -> set:

        if stopword_file:
            path = Path(stopword_file)
        else:
            path = Path(__file__).resolve().parent / "stopwords_nltk_en.txt"

        if path.exists():
            words = {
                line.strip().lower()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }

            words.update({"'m", "'re", "'ve", "'ll"})
            return words


        return {
            "the", "a", "an", "and", "or", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
            "been", "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "should", "could", "may", "might", "must", "can"
        }

    def identify_token_types(self, input_ids: torch.Tensor):


        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)


        content_mask = torch.zeros(len(tokens), dtype=torch.bool, device=input_ids.device)
        stopword_mask = torch.zeros(len(tokens), dtype=torch.bool, device=input_ids.device)
        special_mask = torch.zeros(len(tokens), dtype=torch.bool, device=input_ids.device)

        punct = set(string.punctuation)
        in_quote = False

        for i, token in enumerate(tokens):

            clean_token = token.replace('Ġ', '').replace('▁', '').strip()


            if token in ['<s>', '</s>', '<pad>', '<unk>', '[CLS]', '[SEP]', '<|endoftext|>', '<|im_start|>', '<|im_end|>']:
                special_mask[i] = True
                continue


            is_punct_token = bool(clean_token) and all(ch in punct for ch in clean_token)
            quote_count = clean_token.count('"') if is_punct_token else 0
            if quote_count > 0:
                special_mask[i] = True
                if quote_count % 2 == 1:
                    in_quote = not in_quote
                continue
            if is_punct_token:
                special_mask[i] = True
                continue


            if clean_token.lower() in self.stopwords:
                if in_quote:
                    content_mask[i] = True
                else:
                    stopword_mask[i] = True
                continue


            content_mask[i] = True

        return content_mask, stopword_mask, special_mask

    def generate_weights(self, embeddings, input_ids):

        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False


        raw_weights = self.weight_generator(embeddings)
        raw_weights = raw_weights.squeeze(-1)

        if squeeze_output:
            raw_weights = raw_weights.squeeze(0)


        content_mask, stopword_mask, special_mask = self.identify_token_types(input_ids)


        weights = torch.where(
            special_mask,
            torch.full_like(raw_weights, 0.01),
            raw_weights
        )

        return weights, content_mask, stopword_mask, special_mask

    def compute_answer_prediction_loss(
            self,
            question: str,
            gated_rationale_embeddings: torch.Tensor,
            answer: str
    ):


        q_ids = self.tokenizer(
            question,
            return_tensors="pt",
            truncation=True,
            max_length=256
        ).input_ids.to(self.device)[0]

        a_ids = self.tokenizer(
            answer,
            return_tensors="pt",
            truncation=True,
            max_length=128
        ).input_ids.to(self.device)[0]


        with torch.no_grad():
            q_emb = self.embedding_layer(q_ids.unsqueeze(0))[0]
            a_emb = self.embedding_layer(a_ids.unsqueeze(0))[0]


        prefix_ratios = [1.0]
        losses = []

        for r in prefix_ratios:

            k = max(1, int(len(gated_rationale_embeddings) * r))
            r_prefix = gated_rationale_embeddings[:k]


            input_embeds = torch.cat([q_emb, r_prefix, a_emb], dim=0)


            q_len = len(q_ids)
            r_len = k
            a_len = len(a_ids)
            total_len = q_len + r_len + a_len


            labels = torch.full((total_len,), -100, dtype=torch.long, device=self.device)
            labels[q_len + r_len:] = a_ids


            outputs = self.transformer(
                inputs_embeds=input_embeds.unsqueeze(0),
                labels=labels.unsqueeze(0)
            )

            losses.append(outputs.loss)

        return torch.stack(losses).mean()

    def compute_mask_ratio_loss(self, loss_m, weights, content_mask, stopword_mask, special_mask, epoch):


        content_weights = weights[content_mask]
        num_content = content_mask.sum().item()

        if num_content > 0:

            budget_ratio = max(0.70 - epoch * 0.015, 0.55)
            budget = budget_ratio * num_content


            budget_loss = (content_weights.sum() - budget).pow(2)
        else:
            budget_loss = torch.tensor(0.0, device=weights.device)

        if num_content > 0:
            min_mean = 0.6 * budget_ratio
            content_prior_loss = torch.relu(min_mean - content_weights.mean()).pow(2)
        else:
            content_prior_loss = torch.tensor(0.0, device=weights.device)


        stopword_weights = weights[stopword_mask]
        stopword_loss = torch.tensor(0.0, device=weights.device)
        stopword_quantile_loss = torch.tensor(0.0, device=weights.device)
        stopword_gap = 0.0
        if stopword_mask.sum().item() > 0 and num_content > 0:
            stop_mean = stopword_weights.mean()
            content_mean = content_weights.mean()
            stopword_gap = (stop_mean - content_mean).item()
            stopword_loss = torch.relu(stop_mean - content_mean + self.stopword_margin).pow(2)
            try:
                stopword_q = torch.quantile(stopword_weights, self.stopword_quantile)
                stopword_quantile_loss = torch.relu(stopword_q - self.stopword_quantile_cap).pow(2)
            except Exception:
                stopword_q = torch.tensor(0.0, device=weights.device)


        lambda_budget = 0.05
        lambda_content = 0.001

        loss = (
            lambda_budget * budget_loss
            + lambda_content * content_prior_loss
            + self.stopword_penalty * (stopword_loss + stopword_quantile_loss)
        )


        stats = {
            "weights_mean": weights.mean().item(),
            "content_weights_mean": content_weights.mean().item() if num_content > 0 else 0.0,
            "weights_var": weights.var().item(),
            "weights_max": weights.max().item(),
            "weights_min": weights.min().item(),
            "reg_loss": loss.item(),
            "budget_loss": budget_loss.item(),
            "content_prior_loss": content_prior_loss.item(),
            "stopword_loss": stopword_loss.item(),
            "stopword_quantile_loss": stopword_quantile_loss.item(),
            "loss_m": loss_m.item(),
            "topk_ratio": (weights > weights.mean()).float().mean().item(),
            "content_weight_mean": weights[content_mask].mean().item() if content_mask.sum() > 0 else 0.0,
            "non_content_weight_mean": weights[~content_mask].mean().item() if (~content_mask).sum() > 0 else 0.0,
            "stopword_mean": stopword_weights.mean().item() if stopword_mask.sum() > 0 else 0.0,
            "stopword_quantile": stopword_q.item() if stopword_mask.sum() > 0 else 0.0,
            "stopword_gap": stopword_gap,
            "special_weight_mean": weights[special_mask].mean().item() if special_mask.sum() > 0 else 0.0,
            "num_content_tokens": content_mask.sum().item(),
            "num_stopword_tokens": stopword_mask.sum().item(),
            "num_special_tokens": special_mask.sum().item()
        }

        return loss, stats

    def compute_total_loss(
        self,
        question: str,
        rationale: str,
        answer: str,
        epoch: int
    ):


        embeddings, input_ids = self.encode_rationale(rationale)


        weights, content_mask, stopword_mask, special_mask = self.generate_weights(embeddings, input_ids)


        gated_embeddings = weights.unsqueeze(-1) * embeddings


        loss_m = self.compute_answer_prediction_loss(
            question=question,
            gated_rationale_embeddings=gated_embeddings,
            answer=answer
        )


        loss_reg, stats = self.compute_mask_ratio_loss(
            loss_m, weights, content_mask, stopword_mask, special_mask, epoch
        )


        loss_total = loss_m + self.alpha * loss_reg


        return loss_total, loss_m, weights, stats

    def compute_token_weights(self, rationale_text: str, return_token_ids: bool = False):

        self.eval()

        with torch.no_grad():

            embeddings, input_ids = self.encode_rationale(rationale_text)


            weights, _, _, _ = self.generate_weights(embeddings, input_ids)


            tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        weights_list = weights.cpu().numpy().tolist()
        if return_token_ids:
            return weights_list, tokens, input_ids.cpu().tolist()
        return weights_list, tokens

    def save_weights(self, save_path: str):


        if self.use_ddp:
            weight_generator_state = self.weight_generator.module.state_dict()
            self_attention_state = self.self_attention.module.state_dict()
        else:
            weight_generator_state = self.weight_generator.state_dict()
            self_attention_state = self.self_attention.state_dict()

        torch.save({
            'weight_generator': weight_generator_state,
            'self_attention': self_attention_state,
            'alpha': self.alpha,
            'temperature': self.temperature,
            'use_ddp': self.use_ddp,
            'world_size': self.world_size
        }, save_path)

        if not self.use_ddp or self.local_rank == 0:
            print(f"💾 Saved weight generator to: {save_path}")

    def load_weights(self, load_path: str, verbose: bool = True):

        checkpoint = torch.load(load_path, map_location=self.device)


        if self.use_ddp:
            self.weight_generator.module.load_state_dict(checkpoint['weight_generator'])
            self.self_attention.module.load_state_dict(checkpoint['self_attention'])
        else:
            self.weight_generator.load_state_dict(checkpoint['weight_generator'])
            self.self_attention.load_state_dict(checkpoint['self_attention'])

        self.alpha = checkpoint['alpha']
        self.temperature = checkpoint['temperature']


        if verbose:
            if 'world_size' in checkpoint:
                print(f"  Checkpoint was trained with: {checkpoint.get('world_size', 1)} GPU(s)")
            elif 'num_gpus' in checkpoint:
                print(f"  Checkpoint was trained with: {checkpoint.get('num_gpus', 1)} GPU(s)")
            print(f"✓ Loaded weight generator from: {load_path}")


def _apply_weights_to_split(
    module: RationaleTokenWeighting,
    dataset_path: str,
    checkpoint_path: str,
    student_model_name: str
):


    dataset_path = Path(dataset_path)
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    print(f"✓ Loaded {len(dataset)} samples from {dataset_path.name}")


    module.eval()


    processed = 0
    skipped = 0
    errors = []

    print(f"🔄 Computing token weights for {len(dataset)} samples...")

    for idx, sample in enumerate(tqdm(dataset, desc="Processing samples")):
        preference = sample.get('preference', '')

        if not preference:
            errors.append(f"Sample {idx}: Empty preference field")
            skipped += 1
            continue

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

                if len(weights) > len(token_ids):
                    weights = weights[:len(token_ids)]
                    tokens = tokens[:len(token_ids)]
                else:
                    avg_weight = sum(weights) / len(weights) if weights else 0.5
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
            raise RuntimeError(
                f"Failed to generate token weights for {skipped} samples. "
                f"Failure rate ({failure_rate:.1f}%) exceeds threshold (5%)."
            )

    print(f"{'='*70}\n")


    backup_path = dataset_path.with_suffix('.backup.json')
    if dataset_path.exists():
        shutil.copy(dataset_path, backup_path)
        print(f"💾 Backup saved to: {backup_path}")


    temp_path = dataset_path.with_suffix('.tmp.json')
    with open(temp_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    shutil.move(str(temp_path), dataset_path)
    print(f"✓ Saved {len(dataset)} samples to: {dataset_path}\n")


class CheckpointManager:


    def __init__(
        self,
        save_dir: str,
        budget_ratio: float = 0.4,
        use_ddp: bool = False,
        rank: int = 0
    ):

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.budget_ratio = budget_ratio
        self.use_ddp = use_ddp
        self.rank = rank


        self.best_task_score = -float('inf')
        self.best_gate_health_score = -float('inf')
        self.best_balanced_score = -float('inf')


        self.best_task_path = self.save_dir / "weight_generator_best_task.pt"
        self.best_gate_path = self.save_dir / "weight_generator_best_gate_health.pt"
        self.best_balanced_path = self.save_dir / "weight_generator_best_balanced.pt"


        self.stats_dir = self.save_dir / "gate_stats"
        self.stats_dir.mkdir(exist_ok=True)

    def _compute_gate_health_score(self, stats: Dict) -> float:


        var = stats.get('weights_var', 0.0)
        if 0.05 <= var <= 0.15:
            var_score = 1.0
        else:
            var_score = max(0.0, 1.0 - abs(var - 0.1) / 0.1)


        mean = stats.get('weights_mean', 0.0)
        if 0.25 <= mean <= 0.5:
            mean_score = 1.0
        else:
            mean_score = max(0.0, 1.0 - abs(mean - 0.375) / 0.375)


        health_score = (var_score + mean_score) / 2.0

        return health_score

    def _compute_balanced_score(
        self,
        loss_m: float,
        stats: Dict
    ) -> float:


        task_score = -loss_m


        weights_mean = stats.get('weights_mean', self.budget_ratio)
        budget_penalty = 0.5 * abs(weights_mean - self.budget_ratio)


        balanced_score = task_score - budget_penalty

        return balanced_score

    def should_save(
        self,
        epoch: int,
        loss_m: float,
        stats: Dict
    ) -> Dict[str, bool]:


        if self.use_ddp and self.rank != 0:
            return {'task': False, 'gate_health': False, 'balanced': False}


        task_score = -loss_m
        gate_health_score = self._compute_gate_health_score(stats)
        balanced_score = self._compute_balanced_score(loss_m, stats)


        save_flags = {
            'task': task_score > self.best_task_score,
            'gate_health': gate_health_score > self.best_gate_health_score,
            'balanced': balanced_score > self.best_balanced_score
        }


        if save_flags['task']:
            self.best_task_score = task_score
        if save_flags['gate_health']:
            self.best_gate_health_score = gate_health_score
        if save_flags['balanced']:
            self.best_balanced_score = balanced_score


        print(f"\n[Epoch {epoch+1}] Checkpoint Evaluation:")
        print(f"  Task (loss_m): {loss_m:.4f} (best: {-self.best_task_score:.4f})")
        print(f"  Gate Health: {gate_health_score:.4f} (best: {self.best_gate_health_score:.4f})")
        print(f"  Balanced: {balanced_score:.4f} (best: {self.best_balanced_score:.4f})")

        if any(save_flags.values()):
            will_save = [k for k, v in save_flags.items() if v]
            print(f"  💾 Will save: {will_save}")

        return save_flags

    def save_checkpoint(
        self,
        module,
        epoch: int,
        checkpoint_type: str,
        loss_m: float,
        stats: Dict
    ):


        if checkpoint_type == 'task':
            save_path = self.best_task_path
        elif checkpoint_type == 'gate_health':
            save_path = self.best_gate_path
        elif checkpoint_type == 'balanced':
            save_path = self.best_balanced_path
        else:
            raise ValueError(f"Unknown checkpoint type: {checkpoint_type}")


        module.save_weights(str(save_path))


        stats_data = {
            'epoch': epoch + 1,
            'checkpoint_type': checkpoint_type,
            'task_metric': {
                'loss_m': loss_m,
                'loss_total': loss_m + module.alpha * stats.get('reg_loss', 0.0)
            },
            'gate_stats': stats,
            'scores': {
                'task_score': -loss_m,
                'gate_health_score': self._compute_gate_health_score(stats),
                'balanced_score': self._compute_balanced_score(loss_m, stats)
            }
        }


        stats_file = self.stats_dir / f"stats_best_{checkpoint_type}.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats_data, f, indent=2, ensure_ascii=False)

        print(f"    ✓ Saved {checkpoint_type} checkpoint: {save_path.name}")
        print(f"    ✓ Saved stats: {stats_file.name}")

    def save_epoch_stats(
        self,
        epoch: int,
        loss_m: float,
        stats: Dict
    ):

        if self.use_ddp and self.rank != 0:
            return

        stats_data = {
            'epoch': epoch + 1,
            'task_metric': {
                'loss_m': loss_m
            },
            'gate_stats': stats
        }

        stats_file = self.stats_dir / f"stats_epoch{epoch+1}.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats_data, f, indent=2, ensure_ascii=False)


def train_weight_generator(
    dataset_path: str,
    save_dir: str,
    student_model_name: str = "qwen2.5-3b",
    epochs: int = 10,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    alpha: float = 0.1,
    temperature: float = 1.0,
    use_ddp: bool = False,
    budget_ratio: float = 0.4,
    stopword_penalty: float = 1.0,
    stopword_margin: float = 0.2,
    stopword_quantile: float = 0.8,
    stopword_quantile_cap: float = 0.25,
    stopword_file: Optional[str] = None
):


    import torch.distributed as dist
    if use_ddp and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1


    if rank == 0:
        print(f"\n{'='*70}")
        print("Training Weight Generator")
        print(f"{'='*70}")
        print(f"  Dataset: {dataset_path}")
        print(f"  Save Dir: {save_dir}")
        print(f"  Epochs: {epochs}")
        print(f"  Batch Size: {batch_size}")
        print(f"  Learning Rate: {learning_rate}")
        print(f"  DDP Mode: {use_ddp}")
        if use_ddp:
            print(f"  World Size: {world_size}")
        print(f"  Stopword Penalty: {stopword_penalty}")
        print(f"  Stopword Margin: {stopword_margin}")
        print(f"  Stopword Quantile: {stopword_quantile}")
        print(f"  Stopword Quantile Cap: {stopword_quantile_cap}")
        print(f"{'='*70}\n")


    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)


    set_seed(get_seed())


    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    if rank == 0:
        print(f"✓ Loaded {len(dataset)} samples")


    module = RationaleTokenWeighting(
        student_model_name=student_model_name,
        alpha=alpha,
        temperature=temperature,
        use_ddp=use_ddp,
        stopword_penalty=stopword_penalty,
        stopword_margin=stopword_margin,
        stopword_quantile=stopword_quantile,
        stopword_quantile_cap=stopword_quantile_cap,
        stopword_file=stopword_file
    )

    if use_ddp:
        optimizer = torch.optim.Adam([
            {'params': module.weight_generator.module.parameters()},
            {'params': module.self_attention.module.parameters()}
        ], lr=learning_rate)
    else:
        optimizer = torch.optim.Adam([
            {'params': module.weight_generator.parameters()},
            {'params': module.self_attention.parameters()}
        ], lr=learning_rate)


    checkpoint_manager = CheckpointManager(
        save_dir=save_dir,
        budget_ratio=budget_ratio,
        use_ddp=use_ddp,
        rank=rank
    )


    log_file = None
    if rank == 0:
        log_file = save_dir / "training.log"
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("Weight Generator Training Log\n")
            f.write("="*70 + "\n")
            f.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset: {dataset_path}\n")
            f.write(f"Student Model: {student_model_name}\n")
            f.write(f"Total Epochs: {epochs}\n")
            f.write(f"Batch Size: {batch_size}\n")
            f.write(f"Learning Rate: {learning_rate}\n")
            f.write(f"Alpha: {alpha}\n")
            f.write(f"Budget Ratio: {budget_ratio}\n")
            f.write(f"Stopword Penalty: {stopword_penalty}\n")
            f.write(f"Stopword Margin: {stopword_margin}\n")
            f.write(f"Stopword Quantile: {stopword_quantile}\n")
            f.write(f"Stopword Quantile Cap: {stopword_quantile_cap}\n")
            f.write(f"DDP Mode: {use_ddp}\n")
            if use_ddp:
                f.write(f"World Size: {world_size}\n")
            f.write("="*70 + "\n\n")
        print(f"📝 Training log: {log_file}")


    for epoch in range(epochs):
        module.train()
        epoch_loss = 0
        epoch_loss_m = 0
        failed_samples = []


        epoch_stats_sum = {}
        num_successful_samples = 0

        if rank == 0:
            print(f"\nEpoch {epoch+1}/{epochs}")
        pbar = tqdm(dataset, desc=f"Training Epoch {epoch+1}", disable=(rank != 0))

        for i, sample in enumerate(pbar):
            question = sample.get('history', '')
            rationale = sample.get('preference', '')
            result_list = sample.get('result', [])
            answer = result_list[0] if (isinstance(result_list, list) and len(result_list) > 0) else ''

            if not rationale or not answer:
                continue

            try:

                loss_total, loss_m, weights, stats = module.compute_total_loss(
                    question=question,
                    rationale=rationale,
                    answer=answer,
                    epoch=epoch
                )


                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()


                epoch_loss += loss_total.item()
                epoch_loss_m += loss_m.item()


                for key, value in stats.items():
                    if key not in epoch_stats_sum:
                        epoch_stats_sum[key] = 0.0
                    epoch_stats_sum[key] += value
                num_successful_samples += 1


                pbar.set_postfix({
                    'loss': f'{loss_total.item():.4f}',
                    'loss_m': f'{loss_m.item():.4f}'
                })


                del loss_total, loss_m, weights, stats


            except Exception as e:
                error_msg = f"Sample {i}: {type(e).__name__}: {str(e)}"
                failed_samples.append(error_msg)

                if len(failed_samples) <= 3:
                    print(f"\n⚠️  {error_msg}")

                continue


        failure_rate = len(failed_samples) / len(dataset) * 100

        if failure_rate > 10:
            if rank == 0:
                print(f"\n❌ ERROR: High failure rate in epoch {epoch+1}: {failure_rate:.1f}%")
                print("First 5 failures:")
                for err in failed_samples[:5]:
                    print(f"  {err}")
            raise RuntimeError(
                f"Training aborted: {len(failed_samples)}/{len(dataset)} samples failed"
            )
        elif failed_samples and rank == 0:
            print(f"\n⚠️  {len(failed_samples)} samples failed ({failure_rate:.1f}%)")


        avg_loss = epoch_loss / num_successful_samples if num_successful_samples > 0 else 0
        avg_loss_m = epoch_loss_m / num_successful_samples if num_successful_samples > 0 else 0


        epoch_avg_stats = {
            key: value / num_successful_samples if num_successful_samples > 0 else 0.0
            for key, value in epoch_stats_sum.items()
        }

        if rank == 0:
            print(f"\n  Epoch {epoch+1} Summary:")
            print(f"    Avg Loss: {avg_loss:.4f}")
            print(f"    Avg Loss_m: {avg_loss_m:.4f}")


            print(f"\n  Gate Stats:")
            print(f"    Weights Mean: {epoch_avg_stats.get('weights_mean', 0.0):.4f}")
            print(f"    Weights Var: {epoch_avg_stats.get('weights_var', 0.0):.4f}")
            print(f"    Content Weight Mean: {epoch_avg_stats.get('content_weight_mean', 0.0):.4f}")
            print(f"    Stopword Mean: {epoch_avg_stats.get('stopword_mean', 0.0):.4f}")
            print(f"    Stopword Gap: {epoch_avg_stats.get('stopword_gap', 0.0):.4f}")
            print(f"    Stopword Quantile: {epoch_avg_stats.get('stopword_quantile', 0.0):.4f}")
            print(f"    Special Weight Mean: {epoch_avg_stats.get('special_weight_mean', 0.0):.4f}")


        if rank == 0:

            save_flags = checkpoint_manager.should_save(
                epoch=epoch,
                loss_m=avg_loss_m,
                stats=epoch_avg_stats
            )

            for ckpt_type, should_save in save_flags.items():
                if should_save:
                    checkpoint_manager.save_checkpoint(
                        module=module,
                        epoch=epoch,
                        checkpoint_type=ckpt_type,
                        loss_m=avg_loss_m,
                        stats=epoch_avg_stats
                    )


            epoch_checkpoint_path = save_dir / f"weight_generator_epoch{epoch+1}.pt"
            module.save_weights(str(epoch_checkpoint_path))


            checkpoint_manager.save_epoch_stats(
                epoch=epoch,
                loss_m=avg_loss_m,
                stats=epoch_avg_stats
            )


            if log_file is not None:
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"Epoch: {epoch+1}/{epochs}\n")
                    f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Successful Samples: {num_successful_samples}/{len(dataset)}\n")
                    if failed_samples:
                        f.write(f"Failed Samples: {len(failed_samples)} ({failure_rate:.2f}%)\n")
                    f.write(f"Budget Ratio: {budget_ratio:.4f}\n")
                    f.write(f"\nLoss Metrics:\n")
                    f.write(f"  Avg Total Loss: {avg_loss:.4f}\n")
                    f.write(f"  Avg Loss_m (masked): {avg_loss_m:.4f}\n")
                    f.write(f"\nGate Statistics:\n")
                    f.write(f"  Weights Mean: {epoch_avg_stats.get('weights_mean', 0.0):.4f}\n")
                    f.write(f"  Weights Variance: {epoch_avg_stats.get('weights_var', 0.0):.4f}\n")
                    f.write(f"  Weights Max: {epoch_avg_stats.get('weights_max', 0.0):.4f}\n")
                    f.write(f"  Weights Min: {epoch_avg_stats.get('weights_min', 0.0):.4f}\n")
                    f.write(f"  Content Weight Mean: {epoch_avg_stats.get('content_weight_mean', 0.0):.4f}\n")
                    f.write(f"  Stopword Mean: {epoch_avg_stats.get('stopword_mean', 0.0):.4f}\n")
                    f.write(f"  Stopword Gap: {epoch_avg_stats.get('stopword_gap', 0.0):.4f}\n")
                    f.write(f"  Stopword Quantile: {epoch_avg_stats.get('stopword_quantile', 0.0):.4f}\n")
                    f.write(f"  Special Weight Mean: {epoch_avg_stats.get('special_weight_mean', 0.0):.4f}\n")
                    f.write(f"  Non-Content Weight Mean: {epoch_avg_stats.get('non_content_weight_mean', 0.0):.4f}\n")
                    f.write(f"  Num Content Tokens (avg): {epoch_avg_stats.get('num_content_tokens', 0.0):.1f}\n")
                    f.write("="*70 + "\n\n")


        if use_ddp:
            dist.barrier()

    if rank == 0:
        print(f"\n{'='*70}")
        print("✅ Training Complete!")
        print(f"{'='*70}")
        print("\nBest Checkpoints:")
        print(f"  Task (min loss_m): {-checkpoint_manager.best_task_score:.4f}")
        print(f"    → {checkpoint_manager.best_task_path.name}")
        print(f"  Gate Health: {checkpoint_manager.best_gate_health_score:.4f}")
        print(f"    → {checkpoint_manager.best_gate_path.name}")
        print(f"  Balanced: {checkpoint_manager.best_balanced_score:.4f}")
        print(f"    → {checkpoint_manager.best_balanced_path.name}")
        print(f"{'='*70}\n")


        if log_file is not None:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write("\n" + "="*70 + "\n")
                f.write("Training Complete!\n")
                f.write("="*70 + "\n")
                f.write(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"\nBest Checkpoints:\n")
                f.write(f"  Task (min loss_m): {-checkpoint_manager.best_task_score:.4f}\n")
                f.write(f"    Checkpoint: {checkpoint_manager.best_task_path.name}\n")
                f.write(f"  Gate Health: {checkpoint_manager.best_gate_health_score:.4f}\n")
                f.write(f"    Checkpoint: {checkpoint_manager.best_gate_path.name}\n")
                f.write(f"  Balanced: {checkpoint_manager.best_balanced_score:.4f}\n")
                f.write(f"    Checkpoint: {checkpoint_manager.best_balanced_path.name}\n")
                f.write("="*70 + "\n")


    if rank == 0:
        print(f"\n{'='*70}")
        print("Applying Token Weights to Dataset")
        print(f"{'='*70}\n")


        latest_checkpoint = save_dir / f"weight_generator_epoch{epochs}.pt"


        dataset_path = Path(dataset_path)
        dataset_dir = dataset_path.parent


        for split in ['train', 'val', 'test']:
            split_path = dataset_dir / f"{split}.json"

            if not split_path.exists():
                print(f"⚠️  {split}.json not found, skipping...")
                continue

            print(f"\n{'─'*70}")
            print(f"Processing: {split} split")
            print(f"{'─'*70}")

            try:
                _apply_weights_to_split(
                    module=module,
                    dataset_path=str(split_path),
                    checkpoint_path=str(latest_checkpoint),
                    student_model_name=student_model_name
                )
            except Exception as e:
                print(f"❌ Error processing {split}: {e}")
                import traceback
                traceback.print_exc()
                continue

        print(f"\n{'='*70}")
        print("✅ Token Weights Applied Successfully!")
        print(f"{'='*70}\n")


    if use_ddp:
        dist.barrier()

    return module
