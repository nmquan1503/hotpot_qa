import torch
from torch.utils.data import Dataset
import pandas as pd

import config


class QADataset(Dataset):
    def __init__(self, path, tokenizer):
        self.data = []

        for _, row in pd.read_csv(path).iterrows():
            question = str(row["question"])
            context = str(row["context"])
            answer = str(row["target"]).strip()

            enc = tokenizer.encode(question, context)

            sequence_ids = enc.sequence_ids()
            context_tokens = [
                i for i, sid in enumerate(sequence_ids) if sid == 1
            ]

            if not context_tokens:
                continue

            context_start = context_tokens[0]
            context_end = context_tokens[-1]

            if answer.lower() in ("yes", "no"):
                answer_type = 1 if answer.lower() == "yes" else 2
                start_positions = []
                end_positions = []

            else:
                answer_type = 0

                positions = []
                idx = context.find(answer)
                while idx != -1:
                    positions.append(idx)
                    idx = context.find(answer, idx + 1)

                if not positions:
                    continue

                start_positions = []
                end_positions = []

                for answer_start in positions:
                    answer_end = answer_start + len(answer)
                    s_pos = e_pos = None

                    for i, (start, end) in enumerate(enc["offset_mapping"]):
                        if sequence_ids[i] != 1:
                            continue
                        if start <= answer_start < end:
                            s_pos = i
                        if start < answer_end <= end:
                            e_pos = i

                    if s_pos is not None and e_pos is not None:
                        start_positions.append(s_pos)
                        end_positions.append(e_pos)

                if not start_positions:
                    continue

            input_ids = enc["input_ids"]

            if len(input_ids) > config.MAX_LEN:
                continue

            self.data.append({
                "input_ids": input_ids,
                "lengths": len(input_ids),
                "start_positions": start_positions,
                "end_positions": end_positions,
                "answer_type": answer_type,
                "target": answer,
                "context_start": context_start,
                "context_end": context_end,
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        x = self.data[i]

        return {
            "input_ids": torch.tensor(x["input_ids"], dtype=torch.long),
            "lengths": torch.tensor(x["lengths"], dtype=torch.long),
            "start_positions": x["start_positions"],
            "end_positions": x["end_positions"],
            "answer_type": torch.tensor(x["answer_type"], dtype=torch.long),
            "target": x["target"],
            "context_start": x["context_start"],
            "context_end": x["context_end"],
        }