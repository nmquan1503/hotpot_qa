from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import os

SEED = 42

def build_source(x):
    context = "\n\n".join(
        f"[{t}]\n{' '.join(s)}"
        for t, s in zip(x["context"]["title"], x["context"]["sentences"])
    )
    return {
        "question": x["question"],
        "context": context,
        "target": x["answer"]
    }

def process(ds):
    return pd.DataFrame([build_source(x) for x in ds])

def main(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor")

    train, dev = train_test_split(
        list(ds["train"]),
        test_size=0.1,
        random_state=SEED,
        shuffle=True
    )

    train_df = process(train)
    dev_df = process(dev)
    eval_df = process(ds["validation"])

    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    dev_df.to_csv(os.path.join(output_dir, "dev.csv"), index=False)
    eval_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    print(f"Train: {len(train_df):,}")
    print(f"Dev:   {len(dev_df):,}")
    print(f"Eval:  {len(eval_df):,}")

    print("\nQUESTION:\n", train_df.iloc[0]["question"])
    print("\nCONTEXT:\n", train_df.iloc[0]["context"])
    print("\nTARGET:\n", train_df.iloc[0]["target"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    main(args.output_dir)