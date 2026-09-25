import torch
import torch.nn.functional as F

import config
from models.model import Model
from data.tokenizer import Tokenizer
from data.dataloader import build_dataloader
from training.trainer import Trainer


def qa_loss(outputs, batch):
    type_loss = F.cross_entropy(
        outputs["answer_type_logits"],
        batch["answer_type"],
    )

    start_loss = F.cross_entropy(
        outputs["start_logits"],
        batch["start_position"],
        ignore_index=-100,
    )

    end_loss = F.cross_entropy(
        outputs["end_logits"],
        batch["end_position"],
        ignore_index=-100,
    )

    return (type_loss + start_loss + end_loss) / 3.0


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