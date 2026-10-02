import torch
import torch.nn.functional as F

import config
from models.model import Model
from data.tokenizer import Tokenizer
from data.dataloader import build_dataloader
from training.trainer import Trainer


def qa_loss(outputs, batch):
    batch_size = outputs["answer_type_logits"].size(0)

    type_loss = F.cross_entropy(
        outputs["answer_type_logits"],
        batch["answer_type"],
        reduction='sum',
    )

    start_logits = outputs["start_logits"]
    end_logits = outputs["end_logits"]
    lengths = batch["lengths"]

    mask = torch.arange(
        start_logits.size(1),
        device=start_logits.device,
    ).unsqueeze(0) < lengths.unsqueeze(1)
    start_logits = start_logits.masked_fill(~mask, float("-inf"))
    end_logits = end_logits.masked_fill(~mask, float("-inf"))

    start_loss = F.cross_entropy(
        start_logits,
        batch["start_position"],
        ignore_index=-100,
        reduction='sum',
    )

    end_loss = F.cross_entropy(
        end_logits,
        batch["end_position"],
        ignore_index=-100,
        reduction='sum',
    )

    return (type_loss + start_loss + end_loss) / batch_size


def train():
    if config.SEED is not None:
        torch.manual_seed(config.SEED)
        torch.cuda.manual_seed_all(config.SEED)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    tokenizer = Tokenizer()

    train_loader = build_dataloader(
        tokenizer,
        mode="train",
    )

    dev_loader = build_dataloader(
        tokenizer,
        mode="dev",
    )

    model = Model()

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )
    print(f"Total params: {total_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        optimizer=optimizer,
        loss_fn=qa_loss,
    )

    trainer.train()


if __name__ == "__main__":
    train()