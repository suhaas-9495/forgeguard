from dataclasses import dataclass

from transformers import PreTrainedTokenizerBase


@dataclass(slots=True)
class TokenUsage:
    """
    Stores token usage for a single inference.
    """

    input_tokens: int
    output_tokens: int
    total_tokens: int


class TokenCounter:
    """
    Counts input and output tokens.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
    ):
        self.tokenizer = tokenizer

    def count(
        self,
        prompt: str,
        response: str,
    ) -> TokenUsage:

        input_tokens = len(
            self.tokenizer.encode(
                prompt,
                add_special_tokens=True,
            )
        )

        output_tokens = len(
            self.tokenizer.encode(
                response,
                add_special_tokens=False,
            )
        )

        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )