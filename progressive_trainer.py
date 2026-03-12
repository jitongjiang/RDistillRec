"""
progressive_trainer.py - 渐进式训练模块 (Stage 4)

本模块实现TDAS和KPOD的渐进式知识蒸馏训练。
基于论文:
- Task Difficulty Aware Self-Paced Learning (TDAS, CIKM 2023)
- Keypoint-based Progressive Chain-of-Thought Distillation for LLMs (KPOD, ICML 2024)

主要类：
- CurriculumScheduler: 课程调度器（TDAS样本级渐进）
- StudentTrainer: Student模型训练器（KPOD加权损失）

使用示例：
    scheduler = CurriculumScheduler(total_epochs=30, initial_budget=0.2)
    trainer = StudentTrainer(student_model, scheduler)
    trainer.progressive_train(train_dataset, val_dataset, num_epochs=30)
"""

import json
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from config import get_processed_file_path, get_seed
import warnings

def set_seed(seed):

    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CurriculumScheduler:


    def __init__(
        self,
        total_epochs: int,
        initial_budget: float = 0.2,
        final_budget: float = 1.0,
        growth_power: float = 2.0
    ):

        self.total_epochs = total_epochs
        self.initial_budget = initial_budget
        self.final_budget = final_budget
        self.growth_power = growth_power
        self.current_epoch = 0

        print(f"\n{'='*70}")
        print("Initializing CurriculumScheduler (TDAS)")
        print(f"{'='*70}")
        print(f"  Total Epochs: {total_epochs}")
        print(f"  Initial Budget: {initial_budget:.2f}")
        print(f"  Final Budget: {final_budget:.2f}")
        print(f"  Growth Power: {growth_power:.2f}")
        print(f"{'='*70}\n")

    def get_difficulty_budget(self, epoch: int) -> float:


        if self.total_epochs <= 0:
            return self.final_budget
        progress = (epoch + 1) / self.total_epochs
        progress = max(0.0, min(1.0, progress))


        growth = progress ** self.growth_power


        budget = self.initial_budget + (self.final_budget - self.initial_budget) * growth

        return budget

    def select_training_samples(
        self,
        dataset: List[Dict],
        budget: float,
        strategy: str = 'top_k'
    ) -> List[Dict]:

        if strategy == 'threshold':

            selected = [
                sample for sample in dataset
                if sample['difficulty']['total'] <= budget
            ]

        elif strategy == 'top_k':

            sorted_samples = sorted(
                dataset,
                key=lambda x: x['difficulty']['total']
            )
            k = int(len(dataset) * budget)
            if k <= 0 and len(dataset) > 0 and budget > 0:
                k = 1
            selected = sorted_samples[:k]

        elif strategy == 'sampling':

            difficulties = np.array([
                sample['difficulty']['total']
                for sample in dataset
            ])


            weights = 1.0 - difficulties
            weights = weights / weights.sum()


            k = int(len(dataset) * budget)


            indices = np.random.choice(
                len(dataset),
                size=min(k, len(dataset)),
                replace=False,
                p=weights
            )

            selected = [dataset[i] for i in indices]

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return selected

    def select_training_indices(
        self,
        dataset: List[Dict],
        budget: float,
        strategy: str = 'top_k'
    ) -> List[int]:

        if strategy == 'threshold':
            return [
                idx for idx, sample in enumerate(dataset)
                if sample['difficulty']['total'] <= budget
            ]

        if strategy == 'top_k':
            sorted_indices = sorted(
                range(len(dataset)),
                key=lambda i: dataset[i]['difficulty']['total']
            )
            k = int(len(dataset) * budget)
            if k <= 0 and len(dataset) > 0 and budget > 0:
                k = 1
            return sorted_indices[:k]

        if strategy == 'sampling':
            difficulties = np.array([
                sample['difficulty']['total']
                for sample in dataset
            ])
            weights = 1.0 - difficulties
            weights = weights / weights.sum()
            k = int(len(dataset) * budget)
            indices = np.random.choice(
                len(dataset),
                size=min(k, len(dataset)),
                replace=False,
                p=weights
            )
            return indices.tolist()

        raise ValueError(f"Unknown strategy: {strategy}")

    def update_epoch(self):

        self.current_epoch += 1


class RecommendationDataset(Dataset):


    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        max_length: int = 512,
        max_target_length: int = 256,
        use_cot: bool = True
    ):

        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_target_length = max_target_length
        self.use_cot = use_cot

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]


        history = sample.get('history', '')
        preference = sample.get('preference', '')
        if preference is None:
            preference = ''
        if not isinstance(preference, str):
            preference = str(preference)
        result_list = sample.get('result', [])

        if self.use_cot:
            input_text = (
                f"User History: {history}\n"
                f"First explain your reasoning about user preferences, then generate recommendations:\n"
            )
        else:
            input_text = (
                f"User History: {history}\n"
                f"Generate recommendations:\n"
            )

        if isinstance(result_list, list) and len(result_list) > 0:
            first_result = result_list[0]
            recommendations_text = first_result if isinstance(first_result, str) else str(first_result)
        else:
            warnings.warn(
                "result_list is empty or not a list, no recommendations generated",
                UserWarning
            )
            recommendations_text = ''

        if self.use_cot:
            target_text = (
                f"Reasoning: {preference}\n"
                f"Recommendations: {recommendations_text}"
            )
        else:
            target_text = f"Recommendations: {recommendations_text}"

        difficulty = sample.get('difficulty', {}).get('total', 0.5)

        token_weights_list = sample.get('token_weights') or []
        if not isinstance(token_weights_list, list):
            token_weights_list = []

        token_ids_list = sample.get('token_ids')
        if (
            isinstance(token_ids_list, list)
            and len(token_ids_list) == len(token_weights_list)
            and hasattr(self.tokenizer, 'all_special_ids')
        ):
            special_ids = set(self.tokenizer.all_special_ids)
            token_weights_list = [
                w for tid, w in zip(token_ids_list, token_weights_list)
                if tid not in special_ids
            ]

        full_text = input_text + target_text
        full_encoding = self.tokenizer(
            full_text,
            max_length=self.max_length + self.max_target_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )

        input_ids = full_encoding['input_ids'].squeeze(0)
        attention_mask = full_encoding['attention_mask'].squeeze(0)

        labels = input_ids.clone()
        input_length = input_encoding['input_ids'].size(1)
        labels[:input_length] = -100

        total_len = full_encoding['input_ids'].size(1)
        token_level_weights = torch.ones(total_len)

        if self.use_cot:
            reasoning_prefix = self.tokenizer(
                "Reasoning: ",
                add_special_tokens=False
            )
            preference_part = self.tokenizer(
                preference,
                add_special_tokens=False
            )

            prefix_len = len(reasoning_prefix['input_ids'])
            preference_len = len(preference_part['input_ids'])

            start_idx = input_length + prefix_len
            end_idx = min(start_idx + preference_len, total_len)
            actual_pref_len = end_idx - start_idx

            pref_weights = torch.tensor(token_weights_list[:actual_pref_len], dtype=torch.float)
            if len(pref_weights) < actual_pref_len:
                avg_weight = pref_weights.mean() if len(pref_weights) > 0 else 1.0
                padding = torch.ones(actual_pref_len - len(pref_weights)) * avg_weight
                pref_weights = torch.cat([pref_weights, padding])

            token_level_weights[start_idx:end_idx] = pref_weights

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'difficulty': torch.tensor(difficulty, dtype=torch.float),
            'token_weights': token_level_weights
        }


class StudentTrainer:


    def __init__(
        self,
        student_model_name: str,
        curriculum_scheduler: CurriculumScheduler,
        device: str = None,
        use_weighted_loss: bool = True,
        use_sample_difficulty_weighting: bool = True,
        use_multi_gpu: bool = True,
        use_ddp: bool = False,
        use_fsdp: bool = False,
        use_cot: bool = True
    ):

        import os
        from datetime import timedelta


        self.use_fsdp = use_fsdp
        self.use_ddp = use_ddp and not use_fsdp
        if self.use_ddp or self.use_fsdp:

            if not torch.distributed.is_initialized():
                ddp_timeout_minutes = int(os.environ.get("DDP_TIMEOUT_MINUTES", "1440"))
                torch.distributed.init_process_group(
                    backend='nccl',
                    timeout=timedelta(minutes=ddp_timeout_minutes)
                )

            self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
            self.world_size = torch.distributed.get_world_size()
            self.global_rank = torch.distributed.get_rank()


            self.device = f'cuda:{self.local_rank}'
            torch.cuda.set_device(self.local_rank)
            self.num_gpus = self.world_size
            self.use_multi_gpu = True
        else:

            num_gpus = torch.cuda.device_count()

            if device is None:
                if num_gpus > 0:
                    self.device = 'cuda'
                    self.num_gpus = num_gpus
                else:
                    self.device = 'cpu'
                    self.num_gpus = 0
            else:
                self.device = device
                self.num_gpus = num_gpus if 'cuda' in device else 0

            self.use_multi_gpu = use_multi_gpu and self.num_gpus > 1
            self.local_rank = 0
            self.world_size = 1
            self.global_rank = 0

        self.use_cot = use_cot
        self.use_weighted_loss = use_weighted_loss
        if not self.use_cot and self.use_weighted_loss:
            if self.global_rank == 0:
                print("  ⚠️  CoT supervision disabled, forcing weighted loss off")
            self.use_weighted_loss = False
        self.use_sample_difficulty_weighting = use_sample_difficulty_weighting
        self.curriculum_scheduler = curriculum_scheduler
        self.student_model_name = student_model_name


        if self.global_rank == 0:
            print(f"\n{'='*70}")
            print("Initializing StudentTrainer (TDAS + KPOD)")
            print(f"{'='*70}")
            print(f"  Student Model: {student_model_name}")
            print(f"  Device: {self.device}")

            if self.num_gpus > 0:
                visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'all')
                print(f"  CUDA_VISIBLE_DEVICES: {visible_devices}")
                print(f"  Available GPUs: {self.num_gpus}")

                if self.use_fsdp:
                    print(f"  Multi-GPU Training: Enabled (FullyShardedDataParallel)")
                    print(f"  World Size: {self.world_size}")
                    print(f"  Local Rank: {self.local_rank}")
                elif self.use_ddp:
                    print(f"  Multi-GPU Training: Enabled (DistributedDataParallel)")
                    print(f"  World Size: {self.world_size}")
                    print(f"  Local Rank: {self.local_rank}")
                elif self.use_multi_gpu:
                    print(f"  Multi-GPU Training: Enabled (DataParallel - Legacy)")
                else:
                    print(f"  Multi-GPU Training: Disabled (single GPU)")

            print(f"  CoT Supervision: {self.use_cot}")
            print(f"  Weighted Loss: {self.use_weighted_loss}")


        if self.global_rank == 0:
            print(f"  Loading student model...")

        self.tokenizer = AutoTokenizer.from_pretrained(student_model_name)


        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(student_model_name)
        if config.is_encoder_decoder:
            raise ValueError(
                f"Seq2Seq/encoder-decoder model is not supported: {student_model_name}. "
                "Please use a decoder-only CausalLM."
            )

        if self.global_rank == 0:
            print(f"  Detected CausalLM model (decoder-only)")
        self.student_model = AutoModelForCausalLM.from_pretrained(student_model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.student_model.config.pad_token_id = self.tokenizer.eos_token_id
            if self.global_rank == 0:
                print(f"  Set pad_token = eos_token for CausalLM")


        if (self.use_ddp or self.use_fsdp) and hasattr(self.student_model, 'gradient_checkpointing_enable'):


            if self.use_fsdp:
                try:
                    self.student_model.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False}
                    )
                    if self.global_rank == 0:
                        print("  ✓ Gradient checkpointing enabled (non-reentrant for FSDP)")
                except TypeError:

                    self.student_model.gradient_checkpointing_enable()
                    if self.global_rank == 0:
                        print("  ✓ Gradient checkpointing enabled (fallback mode)")
            else:
                self.student_model.gradient_checkpointing_enable()
                if self.global_rank == 0:
                    print(f"  ✓ Gradient checkpointing enabled (saves ~50% memory)")

            if hasattr(self.student_model, "config"):
                self.student_model.config.use_cache = False
        elif not (self.use_ddp or self.use_fsdp):
            if self.global_rank == 0:
                print(f"  ⚠️  Gradient checkpointing DISABLED (incompatible with DataParallel)")


        self.student_model.to(self.device)


        if self.use_fsdp:
            from torch.distributed.fsdp import (
                FullyShardedDataParallel as FSDP,
                ShardingStrategy,
                MixedPrecision,
                BackwardPrefetch,
            )

            mp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            fsdp_mp_policy = MixedPrecision(
                param_dtype=mp_dtype,
                reduce_dtype=mp_dtype,
                buffer_dtype=mp_dtype
            )

            self.student_model = FSDP(
                self.student_model,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                mixed_precision=fsdp_mp_policy,
                backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
                device_id=self.local_rank,
                use_orig_params=True
            )
            if self.global_rank == 0:
                print(f"  ✓ Model wrapped with FullyShardedDataParallel (FULL_SHARD)")
        elif self.use_ddp:

            from torch.nn.parallel import DistributedDataParallel as DDP
            self.student_model = DDP(
                self.student_model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False
            )
            if self.global_rank == 0:
                print(f"  ✓ Model wrapped with DistributedDataParallel")
        elif self.use_multi_gpu:

            if self.global_rank == 0:
                print(f"  Wrapping model with DataParallel (Legacy)...")
            self.student_model = nn.DataParallel(self.student_model)
            if self.global_rank == 0:
                print(f"  ✓ Model distributed across {self.num_gpus} GPUs")

        if self.global_rank == 0:
            print(f"✓ Student model loaded successfully")
            print(f"{'='*70}\n")

    def _init_online_wg(
        self,
        checkpoint_path: str,
        alpha: float,
        temperature: float,
        stopword_penalty: float,
        stopword_margin: float,
        stopword_quantile: float,
        stopword_quantile_cap: float
    ):

        from token_weight import RationaleTokenWeighting

        model_for_wg = self.student_model
        if hasattr(model_for_wg, "module"):
            model_for_wg = model_for_wg.module

        module = RationaleTokenWeighting(
            student_model_name=self.student_model_name,
            alpha=alpha,
            temperature=temperature,
            use_ddp=False,
            stopword_penalty=stopword_penalty,
            stopword_margin=stopword_margin,
            stopword_quantile=stopword_quantile,
            stopword_quantile_cap=stopword_quantile_cap,
            external_tokenizer=self.tokenizer,
            external_transformer=model_for_wg,
            device=self.device
        )
        module.load_weights(checkpoint_path, verbose=(self.global_rank == 0))
        return module

    def _attach_token_weights(
        self,
        module,
        samples: List[Dict],
        desc: str
    ) -> None:

        module.eval()
        pbar = tqdm(samples, desc=desc, disable=(self.global_rank != 0))
        for sample in pbar:
            preference = sample.get('preference', '')
            if not preference:
                continue
            try:
                weights, _, token_ids = module.compute_token_weights(
                    preference,
                    return_token_ids=True
                )
                sample['token_weights'] = weights
                sample['token_ids'] = token_ids
            except Exception:
                continue

    def _update_weight_generator_on_samples(
        self,
        module,
        samples: List[Dict],
        epoch: int,
        learning_rate: float,
        accumulation_steps: int = 1
    ) -> None:

        if self.use_ddp and self.global_rank != 0:
            return
        module.train()

        optimizer = torch.optim.Adam([
            {'params': module.weight_generator.parameters()},
            {'params': module.self_attention.parameters()}
        ], lr=learning_rate)

        model_for_wg = module.transformer
        was_training = model_for_wg.training
        prev_requires_grad = [p.requires_grad for p in model_for_wg.parameters()]
        for p in model_for_wg.parameters():
            p.requires_grad = False
        model_for_wg.eval()

        optimizer.zero_grad()
        pbar = tqdm(samples, desc=f"WG Update Epoch {epoch+1}", disable=(self.global_rank != 0))
        step = 0
        for sample in pbar:
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
                loss_total.backward()
                step += 1

                if step % max(1, accumulation_steps) == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                if self.global_rank == 0:
                    pbar.set_postfix({
                        'loss': f'{loss_total.item():.4f}',
                        'loss_m': f'{loss_m.item():.4f}'
                    })

                del loss_total, loss_m, weights, stats
            except Exception:
                continue

        if step % max(1, accumulation_steps) != 0:
            optimizer.step()
            optimizer.zero_grad()


        for p, req in zip(model_for_wg.parameters(), prev_requires_grad):
            p.requires_grad = req
        if was_training:
            model_for_wg.train()

    def _sync_online_wg_from_checkpoint(self, module, checkpoint_path: str) -> None:

        if not (self.use_ddp or self.use_fsdp):
            return
        import torch.distributed as dist
        dist.barrier()
        module.load_weights(checkpoint_path, verbose=(self.global_rank == 0))
        dist.barrier()

    def _get_base_model(self):

        return self.student_model.module if hasattr(self.student_model, "module") else self.student_model

    def _save_state_dict_to_path(self, checkpoint_path: Path) -> None:

        if self.use_fsdp:
            from torch.distributed.fsdp import (
                FullyShardedDataParallel as FSDP,
                StateDictType,
                FullStateDictConfig,
            )

            save_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.student_model, StateDictType.FULL_STATE_DICT, save_cfg):
                full_state = self.student_model.state_dict()
            if self.global_rank == 0:
                torch.save(full_state, checkpoint_path)
        else:
            model_to_save = self._get_base_model()
            if self.global_rank == 0:
                torch.save(model_to_save.state_dict(), checkpoint_path)

    def compute_weighted_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        difficulties: torch.Tensor,
        token_weights: torch.Tensor
    ) -> torch.Tensor:


        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_token_weights = token_weights[..., 1:].contiguous()


        loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)


        batch_size = shift_labels.size(0)
        seq_len = shift_labels.size(1)
        vocab_size = shift_logits.size(-1)


        with torch.no_grad():
            valid_mask = (shift_labels != -100).float()
            token_weight_mask = shift_token_weights * valid_mask


        per_token_loss = loss_fct(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1)
        )
        per_token_loss = per_token_loss.view(batch_size, seq_len)


        token_den = token_weight_mask.sum(dim=1).clamp_min(1e-8)
        per_sample_loss = (per_token_loss * token_weight_mask).sum(dim=1) / token_den


        if self.use_sample_difficulty_weighting:
            sample_weights = (1.0 + difficulties).clamp(0.8, 1.8)
        else:
            sample_weights = torch.ones_like(per_sample_loss)

        weighted_loss = (per_sample_loss * sample_weights).mean()


        del per_token_loss, valid_mask, token_weight_mask, token_den, per_sample_loss, sample_weights
        del shift_logits, shift_labels, shift_token_weights

        return weighted_loss

    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        accumulation_steps: int = 2
    ) -> Dict[str, float]:

        self.student_model.train()


        optimizer.zero_grad()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        total_loss = 0
        num_batches = 0


        if self.use_ddp or self.use_fsdp:
            pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}", disable=(self.global_rank != 0))
        else:
            pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")

        for batch_idx, batch in enumerate(pbar):

            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            difficulties = batch['difficulty'].to(self.device)


            token_weights = batch['token_weights'].to(self.device)


            outputs = self.student_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )


            loss = self.compute_weighted_loss(
                logits=outputs.logits,
                labels=labels,
                difficulties=difficulties,
                token_weights=token_weights
            )

            loss = loss / accumulation_steps

            loss.backward()


            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

                if scheduler is not None:
                    scheduler.step()


                torch.cuda.empty_cache()


            loss_value = loss.item()
            total_loss += loss_value
            num_batches += 1


            pbar.set_postfix({
                'loss': f'{loss_value:.4f}',
                'avg_loss': f'{total_loss / num_batches:.4f}'
            })


            del input_ids, attention_mask, labels, difficulties, outputs, loss
            if (batch_idx + 1) % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        avg_loss = total_loss / num_batches


        if self.use_ddp or self.use_fsdp:
            avg_loss_tensor = torch.tensor(avg_loss, device=self.device)
            torch.distributed.all_reduce(avg_loss_tensor, op=torch.distributed.ReduceOp.AVG)
            avg_loss = avg_loss_tensor.item()

        return {
            'loss': avg_loss
        }

    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:

        self.student_model.eval()

        total_loss = 0
        num_batches = 0

        with torch.no_grad():

            if self.use_ddp or self.use_fsdp:
                pbar = tqdm(val_loader, desc="Evaluating", disable=(self.global_rank != 0))
            else:
                pbar = tqdm(val_loader, desc="Evaluating")

            for batch in pbar:

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)


                outputs = self.student_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )


                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100
                )

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches


        if self.use_ddp or self.use_fsdp:
            avg_loss_tensor = torch.tensor(avg_loss, device=self.device)
            torch.distributed.all_reduce(avg_loss_tensor, op=torch.distributed.ReduceOp.AVG)
            avg_loss = avg_loss_tensor.item()

        return {
            'val_loss': avg_loss
        }

    def progressive_train(
        self,
        train_dataset: List[Dict],
        val_dataset: List[Dict],
        num_epochs: int,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        save_dir: str = './checkpoints/student_no_token_weight',
        sample_selection_strategy: str = 'threshold',
        disable_curriculum: bool = False,
        enable_early_stopping: bool = True,
        early_stopping_patience: Optional[int] = None,
        online_wg: bool = False,
        online_wg_checkpoint: Optional[str] = None,
        online_wg_save_dir: Optional[str] = None,
        online_wg_warmup_epochs: int = 1,
        online_wg_lr: Optional[float] = None,
        online_wg_batch_size: int = 4,
        online_wg_alpha: float = 0.1,
        online_wg_temperature: float = 1.0,
        online_wg_stopword_penalty: float = 1.0,
        online_wg_stopword_margin: float = 0.2,
        online_wg_stopword_quantile: float = 0.8,
        online_wg_stopword_quantile_cap: float = 0.25
    ):

        print(f"\n{'='*70}")
        print("Progressive Training (TDAS + KPOD)")
        print(f"{'='*70}")
        print(f"  Total Epochs: {num_epochs}")
        print(f"  Batch Size: {batch_size}")
        print(f"  Learning Rate: {learning_rate}")
        print(f"  Sample Selection: {sample_selection_strategy}")
        print(f"  CoT Supervision: {'enabled' if self.use_cot else 'disabled (recommendations only)'}")
        if enable_early_stopping:
            patience_note = early_stopping_patience if early_stopping_patience is not None else "auto"
            print(f"  Early Stopping: enabled (patience={patience_note})")
        else:
            print(f"  Early Stopping: disabled")
        print(f"{'='*70}\n")


        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)


        set_seed(get_seed())

        wg_module = None
        if online_wg and self.use_fsdp:
            if self.global_rank == 0:
                print("  ⚠️  Online WG disabled: FSDP mode not supported for online WG updates")
            online_wg = False

        if online_wg and not self.use_weighted_loss:
            if self.global_rank == 0:
                print("  ⚠️  Online WG disabled: weighted loss is off")
            online_wg = False

        if online_wg and self.use_weighted_loss:
            if online_wg_checkpoint is None or not Path(online_wg_checkpoint).exists():
                if self.global_rank == 0:
                    print(f"  ??  Online WG disabled: checkpoint not found: {online_wg_checkpoint}")
                online_wg = False
            else:
                if online_wg_save_dir is None:
                    online_wg_save_dir = save_dir / "wg_online"
                online_wg_save_dir = Path(online_wg_save_dir)
                online_wg_save_dir.mkdir(parents=True, exist_ok=True)

                if online_wg_lr is None:
                    online_wg_lr = learning_rate

                if self.global_rank == 0:
                    print(f"  Online WG: enabled")
                    if self.use_ddp or self.use_fsdp:
                        mode_name = "FSDP" if self.use_fsdp else "DDP"
                        print(f"    mode: {mode_name} (rank0 updates, all ranks reload)")
                    print(f"    checkpoint: {online_wg_checkpoint}")
                    print(f"    save dir: {online_wg_save_dir}")
                    print(f"    lr: {online_wg_lr}")
                    print(f"    warmup epochs: {online_wg_warmup_epochs}")

                wg_module = self._init_online_wg(
                    checkpoint_path=online_wg_checkpoint,
                    alpha=online_wg_alpha,
                    temperature=online_wg_temperature,
                    stopword_penalty=online_wg_stopword_penalty,
                    stopword_margin=online_wg_stopword_margin,
                    stopword_quantile=online_wg_stopword_quantile,
                    stopword_quantile_cap=online_wg_stopword_quantile_cap
                )

        if not self.use_weighted_loss:

            for sample in train_dataset:
                token_weights = sample.get("token_weights")
                if isinstance(token_weights, list) and len(token_weights) > 0:
                    sample["token_weights"] = [1] * len(token_weights)
                else:
                    sample["token_weights"] = []

            for sample in val_dataset:
                token_weights = sample.get("token_weights")
                if isinstance(token_weights, list) and len(token_weights) > 0:
                    sample["token_weights"] = [1] * len(token_weights)
                else:
                    sample["token_weights"] = []


        val_torch_dataset = RecommendationDataset(
            val_dataset,
            self.tokenizer,

            max_length = 768,
            use_cot=self.use_cot
        )


        if self.use_ddp or self.use_fsdp:
            from torch.utils.data.distributed import DistributedSampler
            val_sampler = DistributedSampler(
                val_torch_dataset,
                num_replicas=self.world_size,
                rank=self.global_rank,
                shuffle=False
            )
            val_loader = DataLoader(
                val_torch_dataset,
                batch_size=batch_size,
                sampler=val_sampler,
                shuffle=False
            )
        else:
            val_loader = DataLoader(
                val_torch_dataset,
                batch_size=batch_size,
                shuffle=False
            )


        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                self.student_model.parameters(),
                lr=learning_rate
            )
            if self.global_rank == 0:
                print(f"  ✓ Using 8-bit AdamW (saves ~75% optimizer memory)")
        except ImportError:

            optimizer = torch.optim.AdamW(
                self.student_model.parameters(),
                lr=learning_rate
            )
            if self.global_rank == 0:
                print(f"  ⚠️  Using standard AdamW (install bitsandbytes for memory optimization)")
                print(f"     pip install bitsandbytes")


        best_val_loss = float('inf')
        patience_counter = 0
        if enable_early_stopping:
            if early_stopping_patience is not None:
                patience = max(1, int(early_stopping_patience))
            elif self.use_weighted_loss:
                patience = 5
            else:
                patience = 30
        else:
            patience = None


        next_selected_indices = None
        for epoch in range(num_epochs):
            if self.global_rank == 0:
                print(f"\n{'='*70}")
                print(f"Epoch {epoch+1}/{num_epochs}")
                print(f"{'='*70}")

            if disable_curriculum:
                if next_selected_indices is None:
                    selected_indices = list(range(len(train_dataset)))
                else:
                    selected_indices = next_selected_indices
                selected_samples = [train_dataset[i] for i in selected_indices]
                if self.global_rank == 0:
                    print("  Curriculum: disabled (using all samples)")
                    print(f"  Selected Samples: {len(selected_samples)} / {len(train_dataset)}")
            else:

                if next_selected_indices is None:
                    budget = self.curriculum_scheduler.get_difficulty_budget(epoch)
                    selected_indices = self.curriculum_scheduler.select_training_indices(
                        train_dataset,
                        budget,
                        strategy=sample_selection_strategy
                    )
                else:
                    budget = self.curriculum_scheduler.get_difficulty_budget(epoch)
                    selected_indices = next_selected_indices

                selected_samples = [train_dataset[i] for i in selected_indices]
                if self.global_rank == 0:
                    print(f"  Difficulty Budget: {budget:.4f}")
                    print(f"  Selected Samples: {len(selected_samples)} / {len(train_dataset)}")

            if online_wg and self.use_weighted_loss:
                if self.global_rank == 0:
                    print("  Online WG: computing token weights for train samples...")
                self._attach_token_weights(
                    wg_module,
                    selected_samples,
                    desc=f"WG Weights (Train) E{epoch+1}"
                )


            train_torch_dataset = RecommendationDataset(
                selected_samples,
                self.tokenizer,
                    max_length=768,
                use_cot=self.use_cot
            )


            if self.use_ddp or self.use_fsdp:
                from torch.utils.data.distributed import DistributedSampler
                train_sampler = DistributedSampler(
                    train_torch_dataset,
                    num_replicas=self.world_size,
                    rank=self.global_rank,
                    shuffle=True,
                    seed=get_seed()
                )

                train_sampler.set_epoch(epoch)

                train_loader = DataLoader(
                    train_torch_dataset,
                    batch_size=batch_size,
                    sampler=train_sampler,
                    shuffle=False
                )
            else:
                train_loader = DataLoader(
                    train_torch_dataset,
                    batch_size=batch_size,
                    shuffle=True
                )


            train_metrics = self.train_epoch(
                train_loader,
                optimizer,
                epoch
            )

            if online_wg and self.use_weighted_loss:
                if self.global_rank == 0:
                    print("  Online WG: computing token weights for val samples...")
                self._attach_token_weights(
                    wg_module,
                    val_dataset,
                    desc=f"WG Weights (Val) E{epoch+1}"
                )


            val_metrics = self.evaluate(val_loader)


            if self.global_rank == 0:
                print(f"\n  Train Loss: {train_metrics['loss']:.4f}")
                print(f"  Val Loss: {val_metrics['val_loss']:.4f}")


            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                patience_counter = 0

                checkpoint_path = save_dir / f"student_best.pt"
                self._save_state_dict_to_path(checkpoint_path)
                if self.global_rank == 0:
                    print(f"  💾 Saved best model to: {checkpoint_path}")

            else:
                if enable_early_stopping:
                    patience_counter += 1
                    if self.global_rank == 0:
                        print(f"  ⚠️  No improvement for {patience_counter} epochs")

                    if patience_counter >= patience:
                        if self.global_rank == 0:
                            print(f"\n⚠️  Early stopping triggered after {epoch+1} epochs")
                        break


            if (epoch + 1) % 5 == 0:
                checkpoint_path = save_dir / f"student_epoch{epoch+1}.pt"
                self._save_state_dict_to_path(checkpoint_path)
                if self.global_rank == 0:
                    print(f"  💾 Saved checkpoint to: {checkpoint_path}")


            if epoch < num_epochs - 1:
                if disable_curriculum:
                    next_selected_indices = selected_indices
                else:
                    next_budget = self.curriculum_scheduler.get_difficulty_budget(epoch + 1)
                    next_selected_indices = self.curriculum_scheduler.select_training_indices(
                        train_dataset,
                        next_budget,
                        strategy=sample_selection_strategy
                    )

                if online_wg and self.use_weighted_loss and epoch >= online_wg_warmup_epochs:
                    next_selected_samples = [train_dataset[i] for i in next_selected_indices]
                    if self.global_rank == 0:
                        print("  Online WG: updating with next-epoch samples...")
                    self._update_weight_generator_on_samples(
                        wg_module,
                        next_selected_samples,
                        epoch=epoch,
                        learning_rate=online_wg_lr,
                        accumulation_steps=max(1, online_wg_batch_size)
                    )
                    if online_wg_save_dir:
                        wg_checkpoint_path = Path(online_wg_save_dir) / f"weight_generator_epoch{epoch+1}.pt"
                        if self.global_rank == 0:
                            wg_module.save_weights(str(wg_checkpoint_path))
                        if self.use_ddp or self.use_fsdp:
                            self._sync_online_wg_from_checkpoint(wg_module, str(wg_checkpoint_path))


            if not disable_curriculum:
                self.curriculum_scheduler.update_epoch()

        if self.global_rank == 0:
            print(f"\n{'='*70}")
            print("✅ Progressive Training Complete!")
            print(f"  Best Val Loss: {best_val_loss:.4f}")
            print(f"{'='*70}\n")


    def save_model(self, save_path: str):

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        if self.use_fsdp:
            from torch.distributed.fsdp import (
                FullyShardedDataParallel as FSDP,
                StateDictType,
                FullStateDictConfig,
            )
            save_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.student_model, StateDictType.FULL_STATE_DICT, save_cfg):
                full_state = self.student_model.state_dict()

            if self.global_rank == 0:
                model_to_save = self._get_base_model()
                model_to_save.save_pretrained(str(save_path), state_dict=full_state)
                self.tokenizer.save_pretrained(str(save_path))
                print(f"💾 Saved model to: {save_path}")

            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            return

        if self.global_rank == 0:
            model_to_save = self._get_base_model()
            model_to_save.save_pretrained(str(save_path))
            self.tokenizer.save_pretrained(str(save_path))
            print(f"💾 Saved model to: {save_path}")

    def cleanup_distributed(self):

        if (self.use_ddp or self.use_fsdp) and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    def load_model(self, load_path: str):

        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(load_path)

        if config.is_encoder_decoder:
            raise ValueError(
                f"Seq2Seq/encoder-decoder model is not supported: {load_path}. "
                "Please use a decoder-only CausalLM checkpoint."
            )
        model = AutoModelForCausalLM.from_pretrained(load_path)


        if self.use_multi_gpu:
            model = nn.DataParallel(model)

        self.student_model = model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        print(f"✓ Loaded model from: {load_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stage 4: Progressive Training (TDAS + KPOD)")
    parser.add_argument('--dataset', type=str, default='ml100k', help='Dataset name')
    parser.add_argument('--model', type=str, default='qwen2.5-3b',
                        help='Student model name')
    parser.add_argument('--epochs', type=int, default=30, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--initial-budget', type=float, default=0.2,
                        help='Initial difficulty budget')
    parser.add_argument('--growth-power', type=float, default=2.0,
                        help='Budget growth power')
    parser.add_argument('--strategy', type=str, default='threshold',
                        choices=['threshold', 'top_k', 'sampling'],
                        help='Sample selection strategy')
    parser.add_argument('--save-dir', type=str, default='./checkpoints/student',
                        help='Save directory')
    parser.add_argument('--no-early-stopping', action='store_true',
                        help='Disable early stopping during training')
    parser.add_argument('--early-stopping-patience', type=int, default=None,
                        help='Early stopping patience (epochs with no improvement). None = auto')

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print("Stage 4: Progressive Training (TDAS + KPOD)")
    print(f"{'='*80}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Student Model: {args.model}")
    print(f"{'='*80}\n")


    train_path = get_processed_file_path(args.dataset, 'train')
    val_path = get_processed_file_path(args.dataset, 'val')

    try:
        with open(train_path, 'r', encoding='utf-8') as f:
            train_dataset = json.load(f)
        print(f"✓ Loaded training data: {len(train_dataset)} samples")

        with open(val_path, 'r', encoding='utf-8') as f:
            val_dataset = json.load(f)
        print(f"✓ Loaded validation data: {len(val_dataset)} samples")

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        exit(1)


    scheduler = CurriculumScheduler(
        total_epochs=args.epochs,
        initial_budget=args.initial_budget,
        growth_power=args.growth_power
    )


    trainer = StudentTrainer(
        student_model_name=args.model,
        curriculum_scheduler=scheduler,
        use_weighted_loss=True,
        use_cot=True
    )


    try:
        trainer.progressive_train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            save_dir=args.save_dir,
            sample_selection_strategy=args.strategy,
            enable_early_stopping=not args.no_early_stopping,
            early_stopping_patience=args.early_stopping_patience
        )


        final_model_path = Path(args.save_dir) / 'final'
        trainer.save_model(str(final_model_path))

    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

