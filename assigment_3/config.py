import os

# --- Transformer hyperparameters (from assignment spec) ---
HIDDEN_SIZE        = 32
INTERMEDIATE_SIZE  = 32 * 4
NUM_HEADS          = 4
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
MAX_SEQ_LEN        = 32
DROPOUT            = 0.1
BATCH_SIZE         = 32
LEARNING_RATE      = 1e-3
MAX_EPOCHS         = 10

# --- Recurrent hyperparameters (Part 2: BiLSTM encoder + LSTM decoder) ---
LSTM_EMBED_SIZE    = 256
LSTM_HIDDEN_SIZE   = 512
LSTM_NUM_LAYERS    = 1
LSTM_DROPOUT       = 0.3
LSTM_BATCH_SIZE    = 32
LSTM_LEARNING_RATE = 1e-3
LSTM_MAX_EPOCHS    = 10

# --- Special token IDs (same for both tokenizers) ---
PAD_ID = 3
BOS_ID = 1
EOS_ID = 2
UNK_ID = 0

# --- Vocabulary sizes ---
SRC_VOCAB_SIZE = 3200   # French
TGT_VOCAB_SIZE = 3200   # English

# --- Paths ---
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH          = os.path.join(_HERE, "resources/parallel_en_fr_corpus")
TOKENIZER_FR_PATH  = os.path.join(_HERE, "resources/tokenizer_fr")
TOKENIZER_EN_PATH  = os.path.join(_HERE, "resources/tokenizer_en")
CHECKPOINT_DIR     = os.path.join(_HERE, "checkpoints")
