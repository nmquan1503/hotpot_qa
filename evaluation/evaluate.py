import re
import string
from collections import Counter

import torch
from tqdm import tqdm

import config
from models.model import Model
from data.tokenizer import Tokenizer
from data.dataloader import build_dataloader
from minimal_attention.inference import AnalysisConfig


MAX_SPAN_LEN = 15


def normalize(text):
    text = text.lower()
    text = "".join(c for c in text if c not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match_score(pred, gold):
    return normalize(pred) == normalize(gold)


def f1_score(pred, gold):
    normalized_pred = normalize(pred)
    normalized_gold = normalize(gold)

    if (
        normalized_pred in ["yes", "no", "noanswer"]
        and normalized_pred != normalized_gold
    ):
        return 0.0

    if (
        normalized_gold in ["yes", "no", "noanswer"]
        and normalized_pred != normalized_gold
    ):
        return 0.0

    pred_tokens = normalized_pred.split()
    gold_tokens = normalized_gold.split()

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)

    return 2 * precision * recall / (precision + recall)


def predict_span(start_logits, end_logits, context_start, context_end):
    batch_size, seq_len = start_logits.shape

    pos = torch.arange(seq_len, device=start_logits.device)

    valid = (
        (pos[None, :, None] >= context_start[:, None, None])
        & (pos[None, None, :] <= context_end[:, None, None])
        & (pos[None, None, :] >= pos[None, :, None])
        & (pos[None, None, :] - pos[None, :, None] <= MAX_SPAN_LEN)
    )

    scores = start_logits[:, :, None] + end_logits[:, None, :]
    scores = scores.masked_fill(~valid, float("-inf"))

    flat = scores.view(batch_size, -1)
    best = flat.argmax(dim=1)

    start = best // seq_len
    end = best % seq_len

    return start, end


@torch.no_grad()
def _warmup(model, loader, device):
    model.eval()
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        lengths = batch["lengths"].to(device)
        model(input_ids=input_ids, lengths=lengths)
        break
    torch.cuda.synchronize(device)


@torch.no_grad()
def evaluate(model, loader, tokenizer, device):
    model.eval()

    num_heads = config.MODEL_DIM // config.HEAD_DIM
    num_bins = 100

    analysis_cfg = AnalysisConfig() if config.ANALYSIS else None
    attn_gate_thresholds = getattr(config, "ENC_ATTN_GATE_THRESHOLDS", None)

    global_attn_mass = [
        torch.zeros(num_heads, num_bins, device=device)
        for _ in range(config.NUM_LAYERS)
    ]
    global_attn_count = [
        torch.zeros(num_heads, num_bins, device=device)
        for _ in range(config.NUM_LAYERS)
    ]
    global_gate_freq = [
        torch.zeros(num_heads, num_bins, device=device)
        for _ in range(config.NUM_LAYERS)
    ]

    total_em = 0.0
    total_f1 = 0.0
    total = 0

    total_kept_ratio_sum = 0.0
    total_batches = 0

    for batch in tqdm(loader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        lengths = batch["lengths"].to(device)

        outputs = model(
            input_ids=input_ids,
            lengths=lengths,
            attn_gate_thresholds=attn_gate_thresholds,
            analysis_cfg=analysis_cfg,
        )

        if analysis_cfg is not None:
            batch_stats = outputs["stats"]["layers"]
            for layer_idx in range(config.NUM_LAYERS):
                if "non_causal_attn_gate_analysis" not in batch_stats[layer_idx]:
                    continue
                layer_stats = batch_stats[layer_idx]["non_causal_attn_gate_analysis"]
                global_attn_mass[layer_idx] += layer_stats["attn_mass"]
                global_attn_count[layer_idx] += layer_stats["attn_count"]
                global_gate_freq[layer_idx] += layer_stats["gate_freq"]

            total_kept_ratio_sum += outputs["stats"]["overall"]["kept_ratio"]
            total_batches += 1

        answer_type = outputs["answer_type_logits"].argmax(dim=1)

        start, end = predict_span(
            outputs["start_logits"],
            outputs["end_logits"],
            batch["context_start"].to(device),
            batch["context_end"].to(device),
        )

        for i, gold in enumerate(batch["targets"]):
            at = answer_type[i].item()
            if at == 1:
                pred = "yes"
            elif at == 2:
                pred = "no"
            else:
                s, e = start[i].item(), end[i].item()
                if s > e:
                    pred = "noanswer"
                else:
                    pred = tokenizer.decode(input_ids[i, s:e + 1].tolist())

            total_em += float(exact_match_score(pred, gold))
            total_f1 += f1_score(pred, gold)
            total += 1

    if total == 0:
        raise RuntimeError("Evaluation dataset is empty.")

    if analysis_cfg is not None:
        layers_stats = []
        for layer_idx in range(config.NUM_LAYERS):
            layers_stats.append({
                "non_causal_attn_gate_analysis": {
                    "attn_mass": global_attn_mass[layer_idx],
                    "attn_count": global_attn_count[layer_idx],
                    "gate_freq": global_gate_freq[layer_idx],
                }
            })

        torch.save(
            {
                "layers": layers_stats,
                "num_bins": num_bins,
            },
            "gate_attn_stats.pt",
        )

        print("Đã lưu gate_attn_stats.pt")
        if total_batches > 0:
            avg_kept_ratio = total_kept_ratio_sum / total_batches
            print(f"Kept ratio trung bình: {avg_kept_ratio:.4f}")

    return total_em / total, total_f1 / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = Tokenizer()
    loader = build_dataloader(tokenizer, mode="eval")

    model = Model().to(device)

    checkpoint = torch.load(config.BEST_MODEL_PATH, map_location=device)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)

    if config.WARMUP_FULL:
        print("Warming up...")
        _warmup(model, loader, device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    em, f1 = evaluate(
        model=model,
        loader=loader,
        tokenizer=tokenizer,
        device=device,
    )

    print(f"EM: {em * 100:.2f}")
    print(f"F1: {f1 * 100:.2f}")


if __name__ == "__main__":
    main()