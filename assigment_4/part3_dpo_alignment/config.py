PROJECT_NAME = "LLM_Alignment_Assignment_4"
RUN_NAME_PREFIX = "Part3_DPO_Qwen"

BASE_MODEL_ID = "Qwen/Qwen2-1.5B-Instruct"
SFT_ADAPTER_CANDIDATES = [
    "../LLM  SFT  with  Q-LORA/sft_qwen_model",
    "../LLMSFT~1/sft_qwen_model",
    "../sft_qwen_model",
    "./sft_qwen_model",
]

DPO_DATASET_NAME = "jondurbin/truthy-dpo-v0.1"
DPO_DATASET_SPLIT = "train"
DPO_SAMPLE_SIZE = 4000
OVERSAMPLE_IF_NEEDED = True

BETAS = [0.1, 0.5, 0.8, 1.0]

BNB_4BIT_QUANT_TYPE = "nf4"
BNB_4BIT_USE_DOUBLE_QUANT = True

TRAIN_ADAPTER_NAME = "train"
REFERENCE_ADAPTER_NAME = "reference"

PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
NUM_TRAIN_EPOCHS = 3
LEARNING_RATE = 1e-5
MAX_PROMPT_LENGTH = 512
MAX_LENGTH = 1024
GRADIENT_CHECKPOINTING = True

OPTIMIZER = "paged_adamw_8bit"
LR_SCHEDULER_TYPE = "cosine"
WARMUP_RATIO = 0.1
LOGGING_STEPS = 10
SAVE_STRATEGY = "epoch"
SEED = 42

OUTPUT_ROOT = "./results_dpo_qwen"
REPORT_TO = "wandb"
