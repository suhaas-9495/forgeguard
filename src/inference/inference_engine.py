"""
ForgeGuard Production Inference Engine
Phase 2 - Session B
"""

import uuid
import platform
import logging
from typing import Dict, Any

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizerBase,
    PreTrainedModel,
)
from peft import PeftModel

from src.training.model_loader import load_config
from src.confidence.confidence_estimator import ConfidenceEstimator
from src.inference.inference_result import InferenceResult
from src.metrics.latency import LatencyTracker
from src.metrics.token_counter import TokenCounter
from src.metrics.cost_estimator import CostEstimator

from src.exceptions import (
    ModelLoadError,
    AdapterLoadError,
    TokenizationError,
    InferenceError,
    ConfidenceError,
    CostEstimationError,
)

from src.logging import (
    log_inference,
    log_error,
)

logger = logging.getLogger(__name__)


# ============================================================
# Load Inference Engine
# ============================================================

def load_inference_engine(
    adapter_path: str | None = None,
    config_path: str = "configs/training_config.yaml",
) -> Dict[str, Any]:

    cfg = load_config(config_path)

    model_name = cfg["model"]["name"]

    logger.info(f"Loading model : {model_name}")

    # ----------------------------------------------------------
    # Device
    # ----------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    logger.info(f"Device : {device}")

    # ----------------------------------------------------------
    # Tokenizer
    # ----------------------------------------------------------

    try:

        tokenizer: PreTrainedTokenizerBase = (
            AutoTokenizer.from_pretrained(
                model_name
            )
        )

    except Exception as e:

        raise ModelLoadError(
            f"Unable to load tokenizer : {e}"
        )

    # ----------------------------------------------------------
    # Base Model
    # ----------------------------------------------------------

    try:

        model: PreTrainedModel = (
            AutoModelForCausalLM.from_pretrained(
                model_name
            )
        )

    except Exception as e:

        raise ModelLoadError(
            f"Unable to load model : {e}"
        )

    # ----------------------------------------------------------
    # Adapter
    # ----------------------------------------------------------

    if adapter_path:

        try:

            logger.info(
                f"Loading Adapter : {adapter_path}"
            )

            model = PeftModel.from_pretrained(
                model,
                adapter_path,
            )

            logger.info("Adapter Loaded Successfully")

        except Exception as e:

            raise AdapterLoadError(
                f"Unable to load adapter : {e}"
            )

    # ----------------------------------------------------------
    # Evaluation Mode
    # ----------------------------------------------------------

    model.eval()

    model.to(device)

    logger.info("Model Ready")

    # ----------------------------------------------------------
    # Engine Metadata
    # ----------------------------------------------------------

    engine = {

        "model": model,

        "tokenizer": tokenizer,

        "device": device,

        "provider": "Local",

        "framework": "Transformers",

        "peft_method": "LoRA",

        "model_name": model_name,

        "model_version": "1.0",

        "adapter_name": adapter_path,

        "adapter_version": "Phase-1",

        "python_version": platform.python_version(),

        "torch_version": torch.__version__,

        "ask": lambda question: ask_question(
            engine=None,
            question=question,
        ),
    }

    engine["ask"] = lambda question: ask_question(
        engine,
        question,
    )

    return engine
# ============================================================
# Ask Question
# ============================================================

def ask_question(
    engine: Dict[str, Any],
    question: str,
) -> InferenceResult:

    request_id = str(uuid.uuid4())

    tokenizer: PreTrainedTokenizerBase = engine["tokenizer"]
    model: PreTrainedModel = engine["model"]
    device = engine["device"]

    prompt = f"""
You are ForgeGuard, a precise AI assistant.

Question:
{question}

Answer:
"""

    latency_tracker = LatencyTracker()

    try:

        # --------------------------------------------------
        # Tokenization
        # --------------------------------------------------

        try:

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
            )

            inputs = {
                k: v.to(device)
                for k, v in inputs.items()
            }

        except Exception as e:

            raise TokenizationError(
                f"Tokenization Failed : {e}"
            )

        # --------------------------------------------------
        # Generation
        # --------------------------------------------------

        latency_tracker.start()

        retry_count = 3

        last_exception = None

        outputs = None

        for attempt in range(retry_count):

            try:

                with torch.inference_mode():

                    outputs = model.generate(

                        **inputs,

                        max_new_tokens=100,

                        temperature=0.30,

                        top_p=0.95,

                        do_sample=True,

                        repetition_penalty=1.1,

                        pad_token_id=tokenizer.eos_token_id,

                        return_dict_in_generate=True,

                        output_scores=True,

                    )

                break

            except RuntimeError as e:

                last_exception = e

                logger.warning(
                    f"Retry {attempt+1}/{retry_count}"
                )

                if torch.cuda.is_available():

                    torch.cuda.empty_cache()

        if outputs is None:

            raise InferenceError(
                f"Generation Failed : {last_exception}"
            )

        latency = latency_tracker.stop()

        # --------------------------------------------------
        # Decode
        # --------------------------------------------------

        response = tokenizer.decode(

            outputs.sequences[0],

            skip_special_tokens=True,

        )

        # --------------------------------------------------
        # Confidence
        # --------------------------------------------------

        try:

            confidence = ConfidenceEstimator().compute(

                outputs.scores

            )

        except Exception as e:

            raise ConfidenceError(

                f"Confidence Failed : {e}"

            )

        # --------------------------------------------------
        # Token Counter
        # --------------------------------------------------

        token_usage = TokenCounter(

            tokenizer

        ).count(

            prompt,

            response,

        )

        # --------------------------------------------------
        # Cost Estimation
        # --------------------------------------------------

        try:

            cost = CostEstimator().estimate(

                provider="local",

                input_tokens=token_usage.input_tokens,

                output_tokens=token_usage.output_tokens,

            )

        except Exception as e:

            raise CostEstimationError(

                f"Cost Estimation Failed : {e}"

            )
        # --------------------------------------------------
        # Build Production Inference Result
        # --------------------------------------------------

        result = InferenceResult(

            # --------------------------------------------------
            # Request Metadata
            # --------------------------------------------------

            request_id=request_id,

            status="SUCCESS",

            timestamp=InferenceResult.create_timestamp(),

            # --------------------------------------------------
            # Model Metadata
            # --------------------------------------------------

            provider=engine["provider"],

            model_name=engine["model_name"],

            model_version=engine["model_version"],

            adapter_name=engine["adapter_name"],

            adapter_version=engine["adapter_version"],

            framework=engine["framework"],

            peft_method=engine["peft_method"],

            device=str(device),

            python_version=engine["python_version"],

            torch_version=engine["torch_version"],

            # --------------------------------------------------
            # Prompt
            # --------------------------------------------------

            prompt=question,

            temperature=0.30,

            max_new_tokens=100,

            finish_reason="completed",

            # --------------------------------------------------
            # Response
            # --------------------------------------------------

            response=response,

            # --------------------------------------------------
            # Confidence
            # --------------------------------------------------

            sequence_confidence=confidence.sequence_confidence,

            token_confidences=confidence.token_confidences,

            min_confidence=confidence.min_confidence,

            max_confidence=confidence.max_confidence,

            # --------------------------------------------------
            # Latency
            # --------------------------------------------------

            latency_ms=latency.latency_ms,

            # --------------------------------------------------
            # Tokens
            # --------------------------------------------------

            input_tokens=token_usage.input_tokens,

            output_tokens=token_usage.output_tokens,

            total_tokens=token_usage.total_tokens,

            # --------------------------------------------------
            # Cost
            # --------------------------------------------------

            input_cost=cost.input_cost,

            output_cost=cost.output_cost,

            total_cost=cost.total_cost,

            # --------------------------------------------------
            # Error
            # --------------------------------------------------

            error_message=None,

        )

        # --------------------------------------------------
        # Structured Logging
        # --------------------------------------------------

        log_inference(result)

        logger.info(

            f"[{request_id}] "

            f"{latency.latency_ms:.2f} ms | "

            f"Confidence={confidence.sequence_confidence:.4f} | "

            f"Tokens={token_usage.total_tokens}"

        )

        return result

    # ======================================================
    # Custom Exceptions
    # ======================================================

    except (

        ModelLoadError,

        AdapterLoadError,

        TokenizationError,

        InferenceError,

        ConfidenceError,

        CostEstimationError,

    ) as e:

        log_error(request_id, e)

        logger.exception(e)

        raise

    # ======================================================
    # Unknown Exception
    # ======================================================

    except Exception as e:

        log_error(request_id, e)

        logger.exception(e)

        raise InferenceError(str(e))


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    engine = load_inference_engine(

        adapter_path="models/adapters/phase1",

    )

    logger.info("Inference Engine Ready")

    result = ask_question(

        engine,

        "Explain LoRA Fine-Tuning."

    )

    print("\n")

    print("=" * 80)

    print("ForgeGuard Production Inference")

    print("=" * 80)

    print(result.to_dict())