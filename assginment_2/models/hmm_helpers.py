def normalize_training_data(sentences, tag_lists=None):
    if tag_lists is None:
        normalized_sentences = []
        normalized_tags = []
        for sample in sentences:
            if isinstance(sample, dict):
                normalized_sentences.append(sample["tokens"])
                normalized_tags.append(sample["ner_tags"])
            else:
                raise ValueError(
                    "When tag_lists is None, each training sample must contain 'tokens' and 'ner_tags'."
                )
        return normalized_sentences, normalized_tags

    normalized_sentences = []
    for sentence in sentences:
        if isinstance(sentence, dict):
            normalized_sentences.append(sentence["tokens"])
        else:
            normalized_sentences.append(sentence)

    return normalized_sentences, tag_lists


def is_punct(word):
    return all(not ch.isalnum() for ch in word)


def split_bio_tag(tag):
    if isinstance(tag, str) and "-" in tag:
        prefix, entity_type = tag.split("-", 1)
        return prefix, entity_type
    return tag, None


def is_valid_bio_transition(prev_tag, current_tag, start_tag, end_tag):
    if prev_tag in {start_tag, end_tag}:
        prev_prefix, prev_type = prev_tag, None
    else:
        prev_prefix, prev_type = split_bio_tag(prev_tag)

    curr_prefix, curr_type = split_bio_tag(current_tag)

    if current_tag == end_tag:
        return True

    if prev_tag == start_tag:
        return curr_prefix != "I"

    if curr_prefix == "I":
        return prev_prefix in {"B", "I"} and prev_type == curr_type

    return True


def classify_unknown_word(word):
    has_digit = any(ch.isdigit() for ch in word)
    has_alpha = any(ch.isalpha() for ch in word)
    has_hyphen = "-" in word
    digit_count = sum(ch.isdigit() for ch in word)

    if is_punct(word):
        return "<UNK_PUNCT>"
    if len(word) == 4 and word.isdigit():
        return "<UNK_YEAR>"
    if has_digit and digit_count >= 4 and any(ch in "-/" for ch in word):
        return "<UNK_DATE>"
    if has_digit and has_hyphen:
        return "<UNK_DIGIT_HYPHEN>"
    if has_digit:
        return "<UNK_DIGIT>"
    if word.istitle():
        if has_hyphen:
            return "<UNK_INITCAP_HYPHEN>"
        return "<UNK_TITLE>"
    if has_hyphen:
        if word.isupper() and has_alpha:
            return "<UNK_ALLCAPS_HYPHEN>"
        if len(word) > 0 and word[0].isupper():
            return "<UNK_INITCAP_HYPHEN>"
        return "<UNK_HYPHEN>"
    if word.isupper() and has_alpha:
        return "<UNK_ALLCAPS>"
    if len(word) > 0 and word[0].isupper() and word[1:].islower():
        return "<UNK_INITCAP>"
    if word.islower():
        return "<UNK_LOWER>"
    if any(ch.islower() for ch in word) and any(ch.isupper() for ch in word):
        return "<UNK_MIXEDCASE>"
    return "<UNK_OTHER>"
