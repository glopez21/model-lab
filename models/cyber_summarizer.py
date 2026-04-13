"""Cybersecurity-Domain Summarizer - Fine-tuned BART for cyber news"""

from agents.base import BaseAgent, AgentConfig, AgentType, AgentResult


class CyberSummarizerAgent(BaseAgent):
    """
    Summarization agent fine-tuned on cybersecurity news articles.

    Best for: Cybersecurity and threat intelligence summaries
    Tradeoff: Domain-specific - may not generalize well to other topics
    Model path: Point MODEL_PATH to your fine-tuned model directory or HuggingFace repo
    """

    def __init__(
        self,
        model_path: str = "facebook/bart-large-cnn",
        max_length: int = 150,
        min_length: int = 50,
    ):
        config = AgentConfig(
            name="cyber-summarizer",
            agent_type=AgentType.SUMMARIZER,
            model_name=model_path,
            description="Cybersecurity-domain fine-tuned summarizer",
            max_length=max_length,
            min_length=min_length,
            timeout=30,
            priority=12,
            tags=["summarizer", "cybersecurity", "fine-tuned", "domain-specific"],
        )
        super().__init__(config)
        self._model_path = model_path

    def load_model(self) -> None:
        from transformers import pipeline

        self._model = pipeline(
            "summarization",
            model=self._model_path,
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
            confidence=0.90,
            tokens_in=len(truncated.split()),
            tokens_out=len(summary.split()),
        )

    def health_check(self) -> bool:
        try:
            result = self._model(
                "A new zero-day vulnerability was discovered.",
                max_length=20,
                min_length=5,
            )
            return len(result) > 0 and len(result[0]["summary_text"]) > 0
        except Exception:
            return False
