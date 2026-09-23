import torch.nn as nn
import config

from attention.models import Encoder, EncoderConfig
from data.tokenizer import Tokenizer


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        tokenizer = Tokenizer()

        self.encoder = Encoder(EncoderConfig(
            vocab_size=tokenizer.vocab_size,
            model_dim=config.MODEL_DIM,
            head_dim=config.HEAD_DIM,
            num_layers=config.NUM_LAYERS,
            dropout_rate=config.DROPOUT_RATE,
            device="cuda",
        ))

        self.qa_outputs = nn.Linear(config.MODEL_DIM, 2)
        self.answer_type = nn.Linear(config.MODEL_DIM, 3)

        self.to("cuda")

    def forward(
        self,
        input_ids,
        lengths,
    ):
        hidden_states = self.encoder(
            input_ids=input_ids,
            lengths=lengths,
        )

        qa_logits = self.qa_outputs(hidden_states)

        result = {
            "start_logits": qa_logits[..., 0],
            "end_logits": qa_logits[..., 1],
            "answer_type_logits": self.answer_type(hidden_states[:, 0]),
        }

        return result