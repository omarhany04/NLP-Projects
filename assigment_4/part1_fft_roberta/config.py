PROJECT_NAME = "LLM_Alignment_Assignment_4"
RUN_NAME = "Part1_FFT_XLM_RoBERTa"

# Assignment-approved choices:
# - FacebookAI/xlm-roberta-base  (0.3B)
# - FacebookAI/xlm-roberta-large (0.6B)
BASE_MODEL_ID = "FacebookAI/xlm-roberta-base"
ALTERNATIVE_MODEL_ID = "FacebookAI/xlm-roberta-large"

DATASET_NAME = "flytech/python-codes-25k"
DATASET_SPLIT = "train"

# Part I may use the full dataset. Set an integer only for quick smoke runs.
DATASET_SAMPLE_SIZE = None
MAX_SEQ_LENGTH = 512

PER_DEVICE_TRAIN_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 1
NUM_TRAIN_EPOCHS = 2
LEARNING_RATE = 5e-5
OPTIMIZER = "adamw_torch"
BF16 = True
GRADIENT_CHECKPOINTING = True

LR_SCHEDULER_TYPE = "cosine"
WARMUP_RATIO = 0.1
LOGGING_STEPS = 25
SAVE_STRATEGY = "epoch"
SEED = 42

OUTPUT_DIR = "./results_fft_roberta"
FINAL_MODEL_DIR = "./fft_roberta_model"
REPORT_TO = "wandb"
