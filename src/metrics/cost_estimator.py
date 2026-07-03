from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(slots=True)
class CostResult:
    """
    Stores the estimated cost of one inference request.
    """

    input_cost: float
    output_cost: float
    total_cost: float


class CostEstimator:
    """
    Estimates inference cost based on provider pricing.
    """

    def __init__(
        self,
        pricing_file: str = "configs/pricing.yaml",
    ):

        pricing_path = Path(pricing_file)

        if not pricing_path.exists():
            raise FileNotFoundError(
                f"Pricing file not found: {pricing_path}"
            )

        with pricing_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            self.pricing = yaml.safe_load(file)

    def estimate(
        self,
        provider: str,
        input_tokens: int,
        output_tokens: int,
    ) -> CostResult:

        providers = self.pricing.get("providers", {})

        if provider not in providers:
            raise ValueError(
                f"Unknown provider: {provider}"
            )

        config = providers[provider]

        input_price = config["input_cost_per_1k_tokens"]
        output_price = config["output_cost_per_1k_tokens"]

        input_cost = (
            input_tokens / 1000
        ) * input_price

        output_cost = (
            output_tokens / 1000
        ) * output_price

        total_cost = input_cost + output_cost

        return CostResult(
            input_cost=round(input_cost, 6),
            output_cost=round(output_cost, 6),
            total_cost=round(total_cost, 6),
        )