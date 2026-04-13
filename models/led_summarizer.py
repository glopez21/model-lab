"""LED Summarizer Agent - Long document summaries using allenai/led-base-16384"""

from agents.base import BaseAgent, AgentConfig, AgentType, AgentResult


class LedSummarizerAgent(BaseAgent):
    """
    Summarization agent using LED (Longformer Encoder-Decoder).

    Best for: Long documents (>1024 tokens) that other models truncate
    Tradeoff: Slower due to long context, but handles full articles
    """

    def __init__(self, max_length: int = 200, min_length: int = 50):
        config = AgentConfig(
            name="led-base-16384",
            agent_type=AgentType.SUMMARIZER,
            model_name="allenai/led-base-16384",
            description="LED summarizer - handles long documents up to 16K tokens",
            max_length=max_length,
            min_length=min_length,
            timeout=60,
            priority=3,
            tags=["summarizer", "long-document", "extended-context"],
        )
        super().__init__(config)

    def load_model(self) -> None:
        from transformers import pipeline

        self._model = pipeline(
            "summarization",
            model=self.config.model_name,
            device_map="auto",
        )

    def process(self, text: str, **kwargs) -> AgentResult:
        max_length = kwargs.get("max_length", self.config.max_length)
        min_length = kwargs.get("min_length", self.config.min_length)

        result = self._model(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
        )

        summary = result[0]["summary_text"]

        return AgentResult(
            agent_id=self.id,
            agent_name=self.config.name,
            agent_type=self.config.agent_type,
            input_text=text[:200],
            output_text=summary,
            confidence=0.80,
            tokens_in=len(text.split()),
            tokens_out=len(summary.split()),
        )

    def health_check(self) -> bool:
        try:
            result = self._model(
                "Short health check article text.", max_length=20, min_length=5
            )
            return len(result) > 0 and len(result[0]["summary_text"]) > 0
        except Exception:
            return False
