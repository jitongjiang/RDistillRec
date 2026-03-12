

import json
import shutil
import time
import os
import argparse
from abc import ABC, abstractmethod
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from tqdm import tqdm
from openai import OpenAI
from config import get_prompt_config


class BaseLLMBackend(ABC):


    def __init__(self, model_name, max_retries=5, retry_delay=2.0, backoff_factor=2.0):

        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.backoff_factor = backoff_factor

    @abstractmethod
    def generate(self, prompt, temperature=0.7, max_tokens=200, **kwargs):

        pass

    def generate_with_retry(
        self,
        prompt,
        validator=None,
        validation_error_msg="Generated text failed validation.",
        **kwargs
    ):

        current_delay = self.retry_delay

        for attempt in range(self.max_retries):
            try:
                generated_text = self.generate(prompt, **kwargs)
                if validator is not None and not validator(generated_text):
                    raise ValueError(validation_error_msg)
                return generated_text
            except Exception as e:
                if attempt == self.max_retries - 1:
                    print(f"❌ Error: Failed after {self.max_retries} retries: {e}")
                    raise

                print(f"⚠️  Attempt {attempt + 1} failed: {e}")
                print(f"   Retrying in {current_delay:.1f}s...")
                time.sleep(current_delay)
                current_delay *= self.backoff_factor


class OpenAIBackend(BaseLLMBackend):


    def __init__(self, model=None, api_key=None, base_url=None, **kwargs):

        prompt_config = get_prompt_config()
        if model is None:
            model = prompt_config.get("model", "gpt-3.5-turbo")

        if base_url is None:
            base_url = prompt_config.get("base_url")

        super().__init__(model_name=model, **kwargs)


        if api_key is None:
            api_key = prompt_config.get('api_key')
            if not api_key:
                api_key_env = prompt_config.get('api_key_env', 'OPENAI_API_KEY')
                api_key = os.getenv(api_key_env, '')

        client_kwargs = {'api_key': api_key}
        if base_url:
            client_kwargs['base_url'] = base_url

        self.client = OpenAI(**client_kwargs)
        self.api_key = api_key
        self.base_url = base_url

        print(f"✓ Initialized OpenAI backend (model={model})")

    def generate(self, prompt, temperature=0.7, max_tokens=200, timeout=60.0, **kwargs):

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            **kwargs
        )

        return response.choices[0].message.content


class CoTGenerator:


    def __init__(self, backend, checkpoint_interval=5):

        self.backend = backend
        self.checkpoint_interval = checkpoint_interval


        self.prompt_template = (
            "viewing history: {history}\n"
            "Please analyze the user's preferences in 100 words based on the viewing history."
        )

    def build_prompt(self, history, demonstrations=None):

        prompt_parts = []


        if demonstrations:
            prompt_parts.append("For example:\n")
            for demo in demonstrations:
                if 'preference' in demo:
                    demo_prompt = (
                        f"viewing history: {demo['demo_history']}\n"
                        f"Please analyze the user's preferences in 100 words based on the viewing history.\n"
                        f"{demo['preference']}\n"
                    )
                    prompt_parts.append(demo_prompt)


        prompt_parts.append(self.prompt_template.format(history=history))

        return "\n".join(prompt_parts)

    def generate_single(self, sample, demonstrations=None, **gen_kwargs):

        history = sample['history']
        prompt = self.build_prompt(history, demonstrations)


        preference = self.backend.generate_with_retry(
            prompt,
            validator=self._ends_with_period,
            validation_error_msg="Generated CoT does not end with a period.",
            **gen_kwargs
        )

        return preference

    @staticmethod
    def _ends_with_period(text):
        if not isinstance(text, str):
            return False

        normalized = text.rstrip()
        if not normalized:
            return False

        return normalized.endswith((".", "\u3002"))

    def generate_dataset(self, dataset_file, demo_file, resume=True, concurrency=1, **gen_kwargs):

        dataset_file = Path(dataset_file)

        print(f"\n{'=' * 60}")
        print("Starting CoT Generation")
        print(f"Input file: {dataset_file}")
        if demo_file:
            print(f"Demo file: {demo_file}")
        print(f"{'=' * 60}")


        demonstrations = self._load_demonstrations(demo_file) if demo_file else None


        with open(dataset_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)


        total = len(dataset)
        processed = 0
        skipped = 0
        concurrency = max(1, int(concurrency))

        try:
            if concurrency == 1:
                for i, sample in tqdm(enumerate(dataset), total=total, desc="Generating CoT"):

                    if resume and 'preference' in sample and sample['preference']:
                        skipped += 1
                        continue


                    preference = self.generate_single(sample, demonstrations, **gen_kwargs)
                    dataset[i]['preference'] = preference
                    processed += 1


                    if (i + 1) % self.checkpoint_interval == 0 or i == total - 1:
                        self._save_checkpoint(dataset, dataset_file)
            else:
                to_process = [
                    i for i, sample in enumerate(dataset)
                    if not (resume and 'preference' in sample and sample['preference'])
                ]
                skipped = total - len(to_process)
                processed_since_checkpoint = 0

                def _worker(idx):
                    preference = self.generate_single(dataset[idx], demonstrations, **gen_kwargs)
                    return idx, preference

                pbar = tqdm(total=total, desc="Generating CoT")
                if skipped:
                    pbar.update(skipped)
                    pbar.set_postfix({"skipped": skipped, "processed": processed})
                iterator = iter(to_process)
                in_flight = set()

                with ThreadPoolExecutor(max_workers=concurrency) as executor:
                    for _ in range(min(concurrency, len(to_process))):
                        try:
                            idx = next(iterator)
                        except StopIteration:
                            break
                        in_flight.add(executor.submit(_worker, idx))

                    while in_flight:
                        done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                        for fut in done:
                            idx, preference = fut.result()
                            dataset[idx]['preference'] = preference
                            processed += 1
                            processed_since_checkpoint += 1
                            pbar.update(1)
                            if processed % 50 == 0 or processed == len(to_process):
                                pbar.set_postfix({"skipped": skipped, "processed": processed})

                            if processed_since_checkpoint >= self.checkpoint_interval:
                                self._save_checkpoint(dataset, dataset_file)
                                processed_since_checkpoint = 0

                            try:
                                next_idx = next(iterator)
                            except StopIteration:
                                next_idx = None
                            if next_idx is not None:
                                in_flight.add(executor.submit(_worker, next_idx))

                if processed_since_checkpoint > 0:
                    self._save_checkpoint(dataset, dataset_file)
                pbar.close()

            print(f"\n{'=' * 60}")
            print("✅ Generation Complete!")
            print(f"{'=' * 60}")
            print(f"📝 Processed: {processed} new items")
            print(f"⏭️  Skipped: {skipped} existing items")
            print(f"📄 Total items: {total}")
            print(f"💾 Saved to: {dataset_file}")
            print(f"{'=' * 60}\n")

            return {
                'total': total,
                'processed': processed,
                'skipped': skipped
            }

        except Exception as e:
            print(f"\n{'=' * 60}")
            print("❌ Error Occurred!")
            print(f"{'=' * 60}")
            print(f"Error: {e}")
            print(f"Processed: {processed} items before error")
            print(f"{'=' * 60}\n")
            raise

    def _load_demonstrations(self, demo_file):

        demo_file = Path(demo_file)

        if not demo_file.exists():
            print(f"⚠️  Warning: Demo file not found: {demo_file}")
            return None

        with open(demo_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, dict) and 'demonstrations' in data:
            demonstrations = data['demonstrations']
            print(f"📄  Success loaded demonstrations")
        else:
            print(f"⚠️  Warning: Unknown demo file format")
            return None

        print(f"✓ Loaded {len(demonstrations)} demonstrations")
        return demonstrations

    def _save_checkpoint(self, dataset, output_file):

        output_file = Path(output_file)


        temp_file = output_file.with_suffix('.tmp.json')
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)


        shutil.move(str(temp_file), output_file)

    def _save_final(self, dataset, output_file):

        self._save_checkpoint(dataset, output_file)
