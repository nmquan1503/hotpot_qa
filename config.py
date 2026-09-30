# Dataset config
TRAIN_PATH = "dataset/train.csv"
DEV_PATH = "dataset/dev.csv"
EVAL_PATH = "dataset/test.csv"

# Training config
SEED = None
BATCH_SIZE = 4
NUM_EPOCHS = 30
LAST_CHECKPOINT_PATH = "last_checkpoint.pt"
BEST_MODEL_PATH = "best_model.pt"
LEARNING_RATE = 1e-3
DROPOUT_RATE = 0.2
RESUME_TRAINING = False
MAX_LEN = 4096

# Model config
MODEL_DIM = 512
HEAD_DIM = 64
NUM_LAYERS = 6

# Eval config
WARMUP_FULL = False