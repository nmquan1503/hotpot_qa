import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from functools import partial

import config
from data.tokenizer import Tokenizer
from data.dataset import QADataset


def seed_worker(worker_id):
    torch.manual_seed(torch.initial_seed() % 2**32)


def qa_collate_fn(batch, pad_id):
    input_ids = [x["input_ids"] for x in batch]

    max_spans = max(
        max(len(x["start_positions"]), 1)
        for x in batch
    )

    start_positions = torch.full(
        (len(batch), max_spans), -100, dtype=torch.long
    )
    end_positions = torch.full(
        (len(batch), max_spans), -100, dtype=torch.long
    )

    for i, x in enumerate(batch):
        n = len(x["start_positions"])
        if n > 0:
            start_positions[i, :n] = torch.tensor(
                x["start_positions"], dtype=torch.long
            )
            end_positions[i, :n] = torch.tensor(
                x["end_positions"], dtype=torch.long
            )

    return {
        "input_ids": pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=pad_id,
        ),
        "lengths": torch.tensor(
            [x.size(0) for x in input_ids],
            dtype=torch.long,
        ),
        "start_positions": start_positions,
        "end_positions": end_positions,
        "answer_type": torch.tensor(
            [x["answer_type"] for x in batch],
            dtype=torch.long,
        ),
        "targets": [x["target"] for x in batch],
        "context_start": torch.tensor(
            [x["context_start"] for x in batch],
            dtype=torch.long,
        ),
        "context_end": torch.tensor(
            [x["context_end"] for x in batch],
            dtype=torch.long,
        ),
    }


def build_dataloader(tokenizer: Tokenizer, mode="train"):
    paths = {
        "train": config.TRAIN_PATH,
        "dev": config.DEV_PATH,
        "eval": config.EVAL_PATH,
    }

    if mode not in paths:
        raise ValueError(f"Unsupported mode: {mode}")

    dataset = QADataset(paths[mode], tokenizer)

    generator = None
    if config.SEED is not None:
        generator = torch.Generator()
        generator.manual_seed(config.SEED)

    return DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=(mode == "train"),
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        collate_fn=partial(
            qa_collate_fn,
            pad_id=tokenizer.pad_id,
        ),
        generator=generator,
        worker_init_fn=seed_worker,
    )