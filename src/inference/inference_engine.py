from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from src.training.model_loader import load_config
from src.confidence.confidence_estimator import ConfidenceEstimator
from src.inference.inference_result import InferenceResult
from src.metrics.latency import LatencyTracker
from src.metrics.token_counter import TokenCounter
from src.metrics.cost_estimator import CostEstimator


def load_inference_engine(
    adapter_path=None,
    config_path="configs/training_config.yaml",
):

    cfg = load_config(config_path)

    model_name = cfg["model"]["name"]

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if adapter_path:
        print(f"Loading adapter: {adapter_path}")

        model = PeftModel.from_pretrained(
            model,
            adapter_path,
        )

        print("Adapter loaded successfully")

    model.eval()

    engine = {
        "tokenizer": tokenizer,
        "model": model,
        "model_name": model_name,
        "adapter_path": adapter_path,
        "ask": lambda q: ask_question(
            {
                "tokenizer": tokenizer,
                "model": model,
                "model_name": model_name,
                "adapter_path": adapter_path,
            },
            q,
        ),
    }

    return engine


def ask_question(engine, question):

    tokenizer = engine["tokenizer"]
    model = engine["model"]

    prompt = f"""
You are a helpful AI assistant.

Question:
{question}

Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    latency_tracker = LatencyTracker()
    latency_tracker.start()

    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.3,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,
    )

    latency = latency_tracker.stop()

    response = tokenizer.decode(
        outputs.sequences[0],
        skip_special_tokens=True,
    )

    confidence = ConfidenceEstimator().compute(
        outputs.scores
    )

    token_usage = TokenCounter(tokenizer).count(
        prompt,
        response,
    )

    cost = CostEstimator().estimate(
        provider="local",
        input_tokens=token_usage.input_tokens,
        output_tokens=token_usage.output_tokens,
    )

    result = InferenceResult(
        response=response,

        sequence_confidence=confidence.sequence_confidence,
        token_confidences=confidence.token_confidences,
        min_confidence=confidence.min_confidence,
        max_confidence=confidence.max_confidence,

        latency_ms=latency.latency_ms,

        input_tokens=token_usage.input_tokens,
        output_tokens=token_usage.output_tokens,
        total_tokens=token_usage.total_tokens,

        input_cost=cost.input_cost,
        output_cost=cost.output_cost,
        total_cost=cost.total_cost,

        model_name=engine["model_name"],
        adapter_name=engine["adapter_path"],

        timestamp=InferenceResult.create_timestamp(),
    )

    return result


if __name__ == "__main__":

    engine = load_inference_engine(
        adapter_path="models/adapters/phase1",
    )

    print("\nInference Engine Loaded Successfully")
    print(f"Model: {engine['model_name']}")
    print(f"Adapter: {engine['adapter_path']}")

    result = ask_question(
        engine,
        "How do you prevent overfitting?",
    )

    print("\nInference Result\n")

    print(result.to_dict())