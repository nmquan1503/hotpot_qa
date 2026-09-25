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
            input_ids = enc["input_ids"]

            if len(input_ids) > config.MAX_LEN:
                continue

            sequence_ids = enc.sequence_ids()
            offset_mapping = enc["offset_mapping"]

            context_tokens = [i for i, sid in enumerate(sequence_ids) if sid == 1]
            if not context_tokens:
                continue

            context_start = context_tokens[0]
            context_end = context_tokens[-1]

            if answer.lower() == "yes":
                answer_type = 1
                start_position = -100
                end_position = -100

            elif answer.lower() == "no":
                answer_type = 2
                start_position = -100
                end_position = -100

            else:
                answer_type = 0

                answer_start = context.find(answer)
                if answer_start == -1:
                    continue
                answer_end = answer_start + len(answer)

                start_position = None
                end_position = None

                for i in context_tokens:
                    s, e = offset_mapping[i]
                    if s == e:
                        continue
                    if start_position is None and s <= answer_start < e:
                        start_position = i
                    if end_position is None and s < answer_end <= e:
                        end_position = i
                    if start_position is not None and end_position is not None:
                        break

                if start_position is None or end_position is None:
                    continue

            self.data.append({
                "input_ids": input_ids,
                "lengths": len(input_ids),
                "start_position": start_position,
                "end_position": end_position,
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
            "start_position": torch.tensor(x["start_position"], dtype=torch.long),
            "end_position": torch.tensor(x["end_position"], dtype=torch.long),
            "answer_type": torch.tensor(x["answer_type"], dtype=torch.long),
            "target": x["target"],
            "context_start": x["context_start"],
            "context_end": x["context_end"],
        }