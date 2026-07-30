"""Minimal cross-encoder relevance scorer (transformers, no sentence-transformers).

Shared by the BEIR CE-negative miner and the NQ gold-quality filter. Default
model is cross-encoder/ms-marco-MiniLM-L-6-v2: higher logit = more relevant
(query, passage) pair.
"""

DEFAULT_CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderScorer:
    """Score (query, passage) pairs for relevance with a HF cross-encoder."""

    def __init__(self, model_name=DEFAULT_CE_MODEL, batch_size=128, max_length=512):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = (AutoModelForSequenceClassification
                      .from_pretrained(model_name).to(self.device).eval())
        self.bs = batch_size
        self.max_length = max_length
        print(f"  CE model {model_name} on {self.device}")

    def score(self, pairs):
        """pairs: list[(query, passage)] -> list[float] relevance logits."""
        out = []
        for i in range(0, len(pairs), self.bs):
            chunk = pairs[i:i + self.bs]
            inp = self.tok([p[0] for p in chunk], [p[1] for p in chunk],
                           padding=True, truncation=True,
                           max_length=self.max_length, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                logits = self.model(**inp).logits.squeeze(-1)
            # squeeze can collapse a length-1 batch to 0-d; re-expand
            if logits.ndim == 0:
                logits = logits.unsqueeze(0)
            out.extend(logits.float().cpu().tolist())
        return out
