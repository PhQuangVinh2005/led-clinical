"""
LED Corrector — Model wrapper for clinical factual error correction.

Wraps HuggingFace LEDForConditionalGeneration with clinical-specific
generation config.
"""

from transformers import (
    AutoTokenizer,
    LEDForConditionalGeneration,
    GenerationConfig,
)


def load_model_and_tokenizer(
    model_name: str = "allenai/led-base-16384",
    gradient_checkpointing: bool = True,
):
    """
    Load LED model and tokenizer for fine-tuning.

    Args:
        model_name: HuggingFace model identifier
        gradient_checkpointing: Enable gradient checkpointing to reduce memory

    Returns:
        (model, tokenizer) tuple
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = LEDForConditionalGeneration.from_pretrained(model_name)

    # ── Critical: align embedding matrix with tokenizer vocab size ────────────
    # LED's tokenizer may have a different len() than model.config.vocab_size
    # (e.g. after special token alignment at load time). If they differ, labels
    # can contain token IDs >= vocab_size, crashing the CUDA cross-entropy
    # kernel with "vectorized_gather_kernel index out of bounds".
    tokenizer_vocab_size = len(tokenizer)
    model_vocab_size = model.config.vocab_size
    if tokenizer_vocab_size != model_vocab_size:
        print(f"[Model] Vocab size mismatch — tokenizer={tokenizer_vocab_size}, "
              f"model={model_vocab_size}. Resizing embeddings.")
        model.resize_token_embeddings(tokenizer_vocab_size)
    else:
        print(f"[Model] Vocab size OK — {tokenizer_vocab_size} tokens")

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Set generation config for inference
    model.generation_config = GenerationConfig(
        max_length=2048,
        num_beams=4,
        length_penalty=1.0,
        no_repeat_ngram_size=3,
        early_stopping=True,
    )

    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Loaded {model_name}")
    print(f"  Total params:     {param_count:,}")
    print(f"  Trainable params: {trainable_count:,}")

    return model, tokenizer

