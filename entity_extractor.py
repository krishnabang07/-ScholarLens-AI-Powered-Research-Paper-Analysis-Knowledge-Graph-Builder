"""
entity_extractor.py — NER + custom rule-based extraction for academic text.

Extracts:
  - ML methods (BERT, attention, diffusion models, etc.)
  - Datasets (ImageNet, GLUE, SQuAD, etc.)
  - Metrics (BLEU, F1, accuracy, perplexity, etc.)
  - Tasks (image classification, NER, summarization, etc.)
  - Author names (cross-referenced with arXiv metadata)
"""

import re
import spacy
from spacy.matcher import PhraseMatcher, Matcher
from typing import NamedTuple
from dataclasses import dataclass, field

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class ExtractedEntity:
    text: str
    label: str          # METHOD | DATASET | METRIC | TASK | AUTHOR
    start: int
    end: int
    confidence: float = 1.0
    normalized: str = ""

    def __post_init__(self):
        if not self.normalized:
            self.normalized = self.text.lower().strip()


@dataclass
class ExtractionResult:
    entities: list[ExtractedEntity] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)
    datasets: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    tasks: list[str] = field(default_factory=list)
    
    def summary(self) -> dict:
        return {
            "total": len(self.entities),
            "methods": len(self.methods),
            "datasets": len(self.datasets),
            "metrics": len(self.metrics),
            "tasks": len(self.tasks),
        }


# ── Seed vocabulary (expandable) ─────────────────────────────────────────────

KNOWN_METHODS = [
    "BERT", "GPT", "GPT-2", "GPT-3", "GPT-4", "T5", "RoBERTa", "ALBERT",
    "attention mechanism", "self-attention", "multi-head attention",
    "transformer", "LSTM", "GRU", "CNN", "ResNet", "VGG", "ViT",
    "diffusion model", "DDPM", "stable diffusion", "GAN", "VAE",
    "reinforcement learning", "RLHF", "PPO", "DQN",
    "contrastive learning", "SimCLR", "MoCo", "CLIP",
    "LoRA", "PEFT", "prompt tuning", "chain-of-thought",
    "knowledge distillation", "quantization", "pruning",
]

KNOWN_DATASETS = [
    "ImageNet", "CIFAR-10", "CIFAR-100", "MNIST", "COCO",
    "GLUE", "SuperGLUE", "SQuAD", "SQuAD 2.0", "RACE", "MMLU",
    "WMT14", "WMT16", "WMT19", "Common Crawl", "C4",
    "MS MARCO", "Natural Questions", "TriviaQA",
    "HumanEval", "MBPP", "GSM8K", "MATH",
    "LibriSpeech", "VoxCeleb", "AudioSet",
]

KNOWN_METRICS = [
    "BLEU", "ROUGE", "METEOR", "BERTScore", "CIDEr",
    "accuracy", "F1", "F1 score", "precision", "recall",
    "perplexity", "MMLU score", "pass@k",
    "FID", "IS", "CLIP score", "LPIPS",
    "WER", "CER", "MOS",
    "mAP", "IoU", "AP50",
]

KNOWN_TASKS = [
    "image classification", "object detection", "semantic segmentation",
    "machine translation", "text summarization", "question answering",
    "natural language inference", "sentiment analysis", "named entity recognition",
    "code generation", "text generation", "image generation",
    "speech recognition", "text-to-speech", "speaker verification",
    "visual question answering", "image captioning", "video understanding",
]


class EntityExtractor:
    """
    Two-pass entity extractor:
      1. PhraseMatcher for known terms (high precision)
      2. spaCy NER for unknown entities (higher recall)
    """

    def __init__(self, model_name: str = "en_core_web_trf"):
        logger.info(f"Loading spaCy model: {model_name}")
        self.nlp = spacy.load(model_name)
        self._build_matchers()

    def _build_matchers(self):
        """Build PhraseMatcher for known vocabulary."""
        self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        
        def add_phrases(label, phrases):
            patterns = [self.nlp.make_doc(p.lower()) for p in phrases]
            self.matcher.add(label, patterns)
        
        add_phrases("METHOD", KNOWN_METHODS)
        add_phrases("DATASET", KNOWN_DATASETS)
        add_phrases("METRIC", KNOWN_METRICS)
        add_phrases("TASK", KNOWN_TASKS)

        logger.info("PhraseMatcher built with %d categories", 4)

    def extract(self, text: str, max_chars: int = 50_000) -> ExtractionResult:
        """
        Run full extraction pipeline on text.
        Truncates at max_chars to avoid OOM on huge papers.
        """
        text = text[:max_chars]
        doc = self.nlp(text)
        
        entities: list[ExtractedEntity] = []
        seen_spans = set()

        # Pass 1: known phrase matching
        matches = self.matcher(doc)
        for match_id, start, end in matches:
            span = doc[start:end]
            key = (start, end)
            if key in seen_spans:
                continue
            seen_spans.add(key)
            label = self.nlp.vocab.strings[match_id]
            entities.append(ExtractedEntity(
                text=span.text,
                label=label,
                start=span.start_char,
                end=span.end_char,
                confidence=0.95,
            ))

        # Pass 2: spaCy NER for ORG / PRODUCT that weren't matched
        for ent in doc.ents:
            if ent.label_ in ("ORG", "PRODUCT", "WORK_OF_ART"):
                key = (ent.start, ent.end)
                if key not in seen_spans:
                    seen_spans.add(key)
                    entities.append(ExtractedEntity(
                        text=ent.text,
                        label="METHOD",  # Conservative mapping
                        start=ent.start_char,
                        end=ent.end_char,
                        confidence=0.7,
                    ))

        # Deduplicate by normalized text + label
        seen_norm = set()
        deduped = []
        for e in entities:
            key = (e.normalized, e.label)
            if key not in seen_norm:
                seen_norm.add(key)
                deduped.append(e)

        result = ExtractionResult(entities=deduped)
        result.methods  = [e.text for e in deduped if e.label == "METHOD"]
        result.datasets = [e.text for e in deduped if e.label == "DATASET"]
        result.metrics  = [e.text for e in deduped if e.label == "METRIC"]
        result.tasks    = [e.text for e in deduped if e.label == "TASK"]

        logger.debug("Extracted %d entities: %s", len(deduped), result.summary())
        return result
