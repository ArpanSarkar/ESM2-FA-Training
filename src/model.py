import os
import torch
from transformers import AutoTokenizer, AutoConfig
from faesm.esm import FAEsmForMaskedLM
from config import get_model_config, load_config


def add_mlm_loss_forward(model: torch.nn.Module) -> torch.nn.Module:
    """
    Monkey-patch model.forward to compute masked LM loss when `labels` are provided.

    Loss:
      CrossEntropyLoss over vocab with ignore_index = -100 and reduction="mean",
      i.e., standard MLM loss: only positions with labels != -100 contribute.
    """
    original_forward = model.forward

    def forward_with_loss(
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs,
    ):
        # Let FAESM do its normal forward: returns a dict with "logits" (and others)
        outputs = original_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

        logits = outputs["logits"]
        loss = None

        if labels is not None:
            loss_fn = torch.nn.CrossEntropyLoss(
                ignore_index=-100,
                reduction="mean",
            )
            loss = loss_fn(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
            )
            outputs["loss"] = loss

        return outputs

    model.forward = forward_with_loss
    return model


def initialize_model_and_tokenizer():
    """
    Initialize FAESM ESM2 MaskedLM + tokenizer:

      * Uses FAESM's ESM2 architecture as given by `model_config.model_name`.
      * Dropout (hidden + attention) can be overridden via model_config:
            hidden_dropout_prob
            attention_probs_dropout_prob
        If not set, we keep the checkpoint's defaults.
      * Does NOT touch biases, layer norms, or other architecture details.
      * Only adds MLM loss to forward() so HuggingFace Trainer can use it.
    """
    model_config = get_model_config()
    model_name = model_config.model_name  # e.g., "fredzzp/esm2_t33_650M_UR50D_faesm"

    # Load tokenizer and base config from FAESM/ESM2 checkpoint
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)

    # Ensure FAESM flag exists; default to SDPA unless explicitly overridden.
    # This avoids AttributeError in faesm when config.use_fa is missing.
    if not hasattr(config, "use_fa"):
        config.use_fa = False

    # Make sure we're in encoder-only MLM mode (should already be true, but harmless to enforce)
    config.is_decoder = False
    config.add_cross_attention = False

    # --- Dropout control via config ---
    # Only override if the YAML explicitly contains these keys; otherwise keep the
    # checkpoint defaults. (Checking hasattr() is not sufficient because config
    # objects will often define these attrs even when the YAML didn't specify them.)
    yaml_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    try:
        raw_cfg = load_config(yaml_path)
    except Exception:
        # Fallback: try whatever config.py considers the default path.
        raw_cfg = load_config()

    model_cfg_raw = (raw_cfg or {}).get("model", {})
    if "hidden_dropout_prob" in model_cfg_raw:
        config.hidden_dropout_prob = float(model_cfg_raw["hidden_dropout_prob"])
    if "attention_probs_dropout_prob" in model_cfg_raw:
        config.attention_probs_dropout_prob = float(model_cfg_raw["attention_probs_dropout_prob"])

    # Fresh model initialized from config (training from scratch)
    model = FAEsmForMaskedLM(config)

    # Add MLM loss so Trainer sees `loss` in outputs
    model = add_mlm_loss_forward(model)

    return model, tokenizer