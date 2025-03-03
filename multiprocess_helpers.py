import spacy
import logging
import torch
from typing import Dict

logger = logging.getLogger(__name__)

# Global variables for worker processes
nlp = None
worker_config = None


def init_worker(config: Dict, use_gpu: bool = False):
    """Initialize worker process with spaCy model.

    Args:
        config: Configuration dictionary
        use_gpu: Whether to use GPU acceleration
    """
    global nlp, worker_config
    worker_config = config

    try:
        # Configure GPU if requested and available
        if use_gpu and torch.cuda.is_available():
            spacy.prefer_gpu()
            logger.info("Worker using GPU acceleration")
        else:
            spacy.require_cpu()

        # Get model name and disabled components from config
        model_name = config["text_processing"]["spacy_model"]
        disabled = config["text_processing"]["spacy_pipeline"].get(
            "disabled_components", []
        )

        # Load the spaCy model
        logger.info(f"Worker loading spaCy model: {model_name}")
        nlp = spacy.load(model_name, disable=disabled)

        # Add essential components if needed
        if "sentencizer" not in nlp.pipe_names and "sentencizer" not in disabled:
            nlp.add_pipe("sentencizer")

        if "lemmatizer" not in nlp.pipe_names and "lemmatizer" not in disabled:
            nlp.add_pipe("lemmatizer")

        if "entity_ruler" in config["text_processing"]["spacy_pipeline"].get(
            "enabled_components", []
        ):
            _add_entity_ruler(nlp, config)

        logger.info(f"Worker initialized with pipeline: {nlp.pipe_names}")
    except Exception as e:
        logger.error(f"Failed to load spaCy model in worker: {e}")
        raise


def _add_entity_ruler(nlp, config):
    """Add entity ruler with skill patterns to the pipeline."""
    if "entity_ruler" not in nlp.pipe_names:
        ruler_config = {"phrase_matcher_attr": "LOWER", "validate": True}
        position = "before" if "ner" in nlp.pipe_names else "last"
        target = "ner" if position == "before" else None

        ruler = nlp.add_pipe(
            "entity_ruler",
            config=ruler_config,
            before=target if position == "before" else None,
        )
    else:
        ruler = nlp.get_pipe("entity_ruler")

    # Add skill patterns from config
    patterns = []
    for category, terms in config["keyword_categories"].items():
        for skill in terms:
            patterns.append({"label": "SKILL", "pattern": skill.lower()})

    ruler.add_patterns(patterns)


def process_chunk(texts):
    """Process a chunk of texts using the initialized model.

    Args:
        texts: List of text strings to process

    Returns:
        Processed results using the global nlp model
    """
    global nlp, worker_config
    if nlp is None:
        raise RuntimeError("Worker not properly initialized with spaCy model")

    # Create keyword extractor using the global model
    from keywords4cv import AdvancedKeywordExtractor

    keyword_extractor = AdvancedKeywordExtractor(worker_config, nlp)

    # Process texts and return results
    return keyword_extractor.extract_keywords(texts)
