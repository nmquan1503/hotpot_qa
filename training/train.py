import torch
import torch.nn.functional as F

import config
from models.model import Model
from data.tokenizer import Tokenizer
from data.dataloader import build_dataloader
from training.trainer import Trainer


def or_ce_span_loss(start_logits, end_logits, start_positions, end_positions):
    start_logprobs = F.log_softmax(start_logits, dim=-1)
    end_logprobs = F.log_softmax(end_logits, dim=-1)

    valid = start_positions != -100

    s_safe = start_positions.clamp(min=0)
    e_safe = end_positions.clamp(min=0)

    s_logp = torch.gather(start_logprobs, 1, s_safe)
    e_logp = torch.gather(end_logprobs, 1, e_safe)

    span_logp = s_logp + e_logp
    span_logp = span_logp.masked_fill(~valid, float("-inf"))

    log_sum = torch.logsumexp(span_logp, dim=1)
    loss = -log_sum

    has_span = valid.any(dim=1)
    loss = loss * has_span.float()

    return loss.sum() / has_span.sum().clamp(min=1)


def qa_loss(outputs, batch):
    type_loss = F.cross_entropy(
        outputs["answer_type_logits"],
        batch["answer_type"],
    )

    span_loss = or_ce_span_loss(
        outputs["start_logits"],
        outputs["end_logits"],
        batch["start_positions"],
        batch["end_positions"],
    )

    return 10.0 * type_loss + span_loss


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
        weight_decay=config.WEIGHT_DECAY,
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