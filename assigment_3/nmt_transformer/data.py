import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from functools import partial
from transformers import PreTrainedTokenizerFast
from datasets import load_from_disk


# Part 2 data: stores French-English pairs and prepares encoder/decoder training examples.
class NMTDataset(Dataset):
    """Parallel French→English dataset.

    Encoder input : French tokens + [EOS]          (length ≤ MAX_SEQ_LEN)
    Decoder input : [BOS] + English tokens          (teacher-forcing input)
    Target        : English tokens + [EOS]          (what the decoder must predict)

    Decoder input and target are always the same length, shifted by 1.
    """

    # Part 2 data: keeps the dataset, tokenizers, max length, and special token IDs.
    def __init__(self, data, src_tokenizer, tgt_tokenizer,
                 max_seq_len, bos_id, eos_id, pad_id):
        self.data          = data
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_seq_len   = max_seq_len
        self.bos_id        = bos_id
        self.eos_id        = eos_id
        self.pad_id        = pad_id

    # Part 2 data: returns the number of parallel sentence pairs in this split.
    def __len__(self):
        return len(self.data)

    # Part 2 data: tokenizes one French-English pair into source IDs, decoder input, and target labels.
    def __getitem__(self, idx):
        src_text = self.data[idx]["text_fr"]   # French  → encoder
        tgt_text = self.data[idx]["text_en"]   # English → decoder

        # --- Source (French) ---
        src_ids = self.src_tokenizer.encode(src_text, add_special_tokens=False)
        src_ids = src_ids[: self.max_seq_len - 1] + [self.eos_id]
        src_ids = torch.tensor(src_ids, dtype=torch.long)

        # --- Target (English) ---
        tgt_ids = self.tgt_tokenizer.encode(tgt_text, add_special_tokens=False)
        tgt_ids = tgt_ids[: self.max_seq_len - 1]   # reserve 1 slot for BOS/EOS

        dec_input = torch.tensor([self.bos_id] + tgt_ids, dtype=torch.long) # example: [BOS] I love you
        target    = torch.tensor(tgt_ids + [self.eos_id], dtype=torch.long) # example: I love you [EOS]

        return src_ids, dec_input, target


# Part 2 data: pads variable-length sentences in a batch with PAD_ID so tensors have equal length.
def _collate_fn(batch, pad_id):
    src_ids, dec_inputs, targets = zip(*batch)
    src_ids    = pad_sequence(src_ids,    batch_first=True, padding_value=pad_id)
    dec_inputs = pad_sequence(dec_inputs, batch_first=True, padding_value=pad_id)
    targets    = pad_sequence(targets,    batch_first=True, padding_value=pad_id)
    return src_ids, dec_inputs, targets


# Part 2 data: loads tokenizers/dataset splits and returns DataLoaders for train, validation, and test.
def get_dataloaders(data_path, tokenizer_fr_path, tokenizer_en_path,
                    batch_size, max_seq_len,
                    bos_id=1, eos_id=2, pad_id=3,
                    pin_memory=False, num_workers=0):
    """Return (train_loader, val_loader, test_loader, src_tokenizer, tgt_tokenizer)."""
    dataset       = load_from_disk(data_path)
    src_tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_fr_path)
    tgt_tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_en_path)

    collate = partial(_collate_fn, pad_id=pad_id)

    # Part 2 data: creates one DataLoader for a specific split with the shared padding function.
    def make_loader(split, shuffle):
        ds = NMTDataset(dataset[split], src_tokenizer, tgt_tokenizer,
                        max_seq_len, bos_id, eos_id, pad_id)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate,
            pin_memory=pin_memory,
            num_workers=num_workers,
        )

    return (
        make_loader("train",      shuffle=True),
        make_loader("validation", shuffle=False),
        make_loader("test",       shuffle=False),
        src_tokenizer,
        tgt_tokenizer,
    )
