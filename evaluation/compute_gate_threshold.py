import torch

import config
from models.model import Model
from data.tokenizer import Tokenizer
from data.dataloader import build_dataloader
from minimal_attention.inference import AnalysisConfig


def compute_gate_threshold():
    device = "cuda"

    tokenizer = Tokenizer()
    dev_loader = build_dataloader(tokenizer, mode="dev")

    model = Model().to(device)

    model.load_state_dict(
        torch.load(
            config.BEST_MODEL_PATH,
            map_location=device,
        )
    )
    model.eval()

    analysis_cfg = AnalysisConfig(
        gate_attn_num_bins=config.ATTN_GATE_NUM_BINS,
    )

    inputs = [batch["input_ids"] for batch in dev_loader]
    lengths = [batch["lengths"] for batch in dev_loader]

    threshold = model.encoder.compute_attn_gate_threshold(
        inputs,
        lengths,
        config.ATTN_MASS_THRESHOLD,
        analysis_cfg,
    )

    print("Encoder gate threshold:")
    print(threshold)


if __name__ == "__main__":
    compute_gate_threshold()