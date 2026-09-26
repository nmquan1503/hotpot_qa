import torch.nn as nn
import config

from minimal_attention.models import Encoder, EncoderConfig
from minimal_attention.modules import RMSNorm
from data.tokenizer import Tokenizer


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        tokenizer = Tokenizer()

        self.encoder = Encoder(EncoderConfig(
            vocab_size=tokenizer.vocab_size,
            model_dim=config.MODEL_DIM,
            head_dim=config.HEAD_DIM,
            attn_log_gate_penalty=config.ATTN_LOG_GATE_PENALTY,
            ssm_state_dim=config.SSM_STATE_DIM,
            ssm_conv_kernel_size=config.SSM_CONV_KERNEL_SIZE,
            ssm_num_groups=config.SSM_NUM_GROUPS,
            ssm_chunk_size=config.SSM_CHUNK_SIZE,
            num_layers=config.NUM_LAYERS,
            dropout_rate=config.DROPOUT_RATE,
            device="cuda",
        ))

        self.start_norm = RMSNorm(config.MODEL_DIM)
        self.end_norm = RMSNorm(config.MODEL_DIM)
        self.type_norm = RMSNorm(config.MODEL_DIM)

        self.start_head = nn.Linear(config.MODEL_DIM, 1)
        self.end_head = nn.Linear(config.MODEL_DIM, 1)
        self.type_head = nn.Linear(config.MODEL_DIM, 3)

        self.to("cuda")

        self.encoder.warmup(config.BATCH_SIZE)

    def forward(
        self,
        input_ids,
        lengths,
        attn_gate_thresholds=None,
        analysis_cfg=None,
    ):
        out = self.encoder(
            input_ids=input_ids,
            lengths=lengths,
            attn_gate_thresholds=attn_gate_thresholds,
            analysis_cfg=analysis_cfg,
        )

        if analysis_cfg is not None:
            hidden_states, stats = out
        else:
            hidden_states, stats = out, None

        start_logits = self.start_head(
            self.start_norm(hidden_states)
        ).squeeze(-1)

        end_logits = self.end_head(
            self.end_norm(hidden_states)
        ).squeeze(-1)

        answer_type_logits = self.type_head(
            self.type_norm(hidden_states[:, 0])
        )

        result = {
            "start_logits": start_logits,
            "end_logits": end_logits,
            "answer_type_logits": answer_type_logits,
        }

        if stats is not None:
            result["stats"] = stats

        return result