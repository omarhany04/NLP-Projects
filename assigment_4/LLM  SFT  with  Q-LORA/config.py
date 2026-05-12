MODEL_ID = "Qwen/Qwen2-1.5B-Instruct"  # Base model
BNB_4BIT_QUANT_TYPE = "nf4" # 4-bit quantization type

DATASET_NAME = "flytech/python-codes-25k" # SFT phase dataset
DATASET_SAMPLE_SIZE = 2500  #sample size above 2000

LORA_RANK = 16
TARGET_MODULES = "all-linear"
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
NUM_TRAIN_EPOCHS = 1
LEARNING_RATE = 2e-4
OPTIMIZER = "paged_adamw_8bit"
MAX_SEQ_LENGTH = 1024

