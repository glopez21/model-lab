"""Pegasus Summarizer Agent - Abstractive summaries using google/pegasus-cnn_dailymail"""

from agents.base import BaseAgent, AgentConfig, AgentType, AgentResult


class PegasusSummarizerAgent(BaseAgent):
    """
    Summarization agent using Pegasus-CNN/DailyMail.

    Best for: High-quality abstractive summaries with strong ROUGE scores
    Tradeoff: Slower inference, larger model than DistilBART
    """

    def __init__(self, max_length: int = 150, min_length: int = 40):
        config = AgentConfig(
            name="pegasus-cnn_dailymail",
            agent_type=AgentType.SUMMARIZER,
            model_name="google/pegasus-cnn_dailymail",
            description="Pegasus summarizer - high quality abstractive summaries",
            max_length=max_length,
            min_length=min_length,
            timeout=30,
            priority=7,
            tags=["summarizer", "abstractive", "high-quality"],
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

        truncated = text[:4000] if len(text) > 4000 else text

        result = self._model(
            truncated,
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
            confidence=0.88,
            tokens_in=len(truncated.split()),
            tokens_out=len(summary.split()),
        )

    def health_check(self) -> bool:
        try:
            result = self._model(
                "Test health check article.", max_length=20, min_length=5
            )
            return len(result) > 0 and len(result[0]["summary_text"]) > 0
        except Exception:
            return False
