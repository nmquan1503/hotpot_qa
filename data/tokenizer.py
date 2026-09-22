from transformers import BertTokenizerFast


class Tokenizer:
    def __init__(self):
        self.hf = BertTokenizerFast.from_pretrained("bert-base-uncased")

        self.vocab_size = self.hf.vocab_size
        self.pad_id = self.hf.pad_token_id
        self.unk_id = self.hf.unk_token_id
        self.cls_id = self.hf.cls_token_id
        self.sep_id = self.hf.sep_token_id

    def encode(self, question, context):
        return self.hf(
            question,
            context,
            add_special_tokens=True,
            return_offsets_mapping=True,
        )

    def decode(self, ids):
        return self.hf.decode(ids, skip_special_tokens=True)