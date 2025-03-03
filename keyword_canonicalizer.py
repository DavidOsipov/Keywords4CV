import logging
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import re
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cosine

logger = logging.getLogger(__name__)


class KeywordCanonicalizer:
    """
    Handles deduplication and canonicalization of keywords extracted from text.
    Implements clustering of similar terms and selection of canonical representatives.
    """

    def __init__(self, nlp, config: Dict):
        """
        Initialize the canonicalizer with language model and configuration.

        Args:
            nlp: spaCy language model with word vectors
            config: Configuration dictionary
        """
        self.nlp = nlp
        self.config = config
        self.abbreviation_map = self._load_abbreviation_map()
        self.canonical_cache = {}  # Cache for canonical forms

        # Extract parameters from config
        canonicalization_config = config.get("canonicalization", {})
        self.similarity_threshold = canonicalization_config.get(
            "similarity_threshold", 0.85
        )
        self.embedding_batch_size = canonicalization_config.get(
            "embedding_batch_size", 64
        )
        self.cluster_min_samples = canonicalization_config.get("cluster_min_samples", 2)
        self.cluster_eps = canonicalization_config.get("cluster_eps", 0.25)
        self.enable_abbreviation_handling = canonicalization_config.get(
            "enable_abbreviation_handling", True
        )
        self.enable_embedding_clustering = canonicalization_config.get(
            "enable_embedding_clustering", True
        )
        self.prioritize_whitelist = canonicalization_config.get(
            "prioritize_whitelist", True
        )
        self.max_ngram_overlap = canonicalization_config.get("max_ngram_overlap", 0.8)

        # Track statistics for monitoring
        self.stats = {
            "duplicates_found": 0,
            "terms_canonicalized": 0,
            "clusters_formed": 0,
        }

    def _load_abbreviation_map(self) -> Dict[str, str]:
        """
        Load abbreviation mappings from configuration or use defaults.

        Returns:
            Dict mapping abbreviations to their expanded forms
        """
        default_abbrev = {
            "ml": "machine learning",
            "ai": "artificial intelligence",
            "nlp": "natural language processing",
            "dl": "deep learning",
            "cv": "computer vision",
            "rl": "reinforcement learning",
            "gan": "generative adversarial network",
            "cnn": "convolutional neural network",
            "rnn": "recurrent neural network",
            "lstm": "long short-term memory",
            "api": "application programming interface",
            "ui": "user interface",
            "ux": "user experience",
            "db": "database",
            "oop": "object oriented programming",
            "js": "javascript",
            "ts": "typescript",
            "py": "python",
            "aws": "amazon web services",
            "gcp": "google cloud platform",
            "iot": "internet of things",
            "sde": "software development engineer",
            "swe": "software engineer",
        }

        # Merge with any abbreviations defined in config
        custom_abbrev = self.config.get("canonicalization", {}).get(
            "abbreviation_map", {}
        )
        return {**default_abbrev, **custom_abbrev}

    def canonicalize_keywords(
        self, keywords: List[str], all_skills: Set[str] = None
    ) -> List[str]:
        """
        Process keywords to remove duplicates and canonicalize forms.

        Args:
            keywords: List of extracted keywords
            all_skills: Optional set of known skills for prioritization

        Returns:
            List of canonicalized keywords
        """
        if not keywords:
            return []

        # 1. Normalize case and whitespace
        normalized_keywords = [self._normalize_keyword(kw) for kw in keywords]

        # 2. Handle abbreviations if enabled
        if self.enable_abbreviation_handling:
            normalized_keywords = self._expand_abbreviations(normalized_keywords)

        # 3. Handle overlapping n-grams
        normalized_keywords = self._resolve_ngram_overlaps(normalized_keywords)

        # 4. Cluster using embeddings if enabled
        if self.enable_embedding_clustering and len(normalized_keywords) > 1:
            canonicalized_keywords = self._cluster_similar_terms(
                normalized_keywords, all_skills
            )
        else:
            # Just remove exact duplicates
            canonicalized_keywords = list(dict.fromkeys(normalized_keywords))

        self.stats["terms_canonicalized"] += len(keywords) - len(canonicalized_keywords)

        return canonicalized_keywords

    def _normalize_keyword(self, keyword: str) -> str:
        """Normalize case and whitespace in a keyword."""
        return " ".join(keyword.lower().split())

    def _expand_abbreviations(self, keywords: List[str]) -> List[str]:
        """Replace abbreviations with their expanded forms."""
        result = []
        for kw in keywords:
            kw_lower = kw.lower()
            # Check if this is a known abbreviation
            if kw_lower in self.abbreviation_map:
                expanded = self.abbreviation_map[kw_lower]
                # Add the expanded form instead of the abbreviation
                result.append(expanded)
                self.stats["duplicates_found"] += 1
                logger.debug(f"Expanded abbreviation: '{kw}' -> '{expanded}'")
            else:
                # Check if this keyword IS the expansion of an abbreviation that's also present
                for abbrev, expansion in self.abbreviation_map.items():
                    if kw_lower == expansion and abbrev in [
                        k.lower() for k in keywords
                    ]:
                        # Skip this expansion since we'll include it when we process the abbreviation
                        break
                else:
                    # Not an expansion of an abbreviation that's also in the list
                    result.append(kw)

        return result

    def _resolve_ngram_overlaps(self, keywords: List[str]) -> List[str]:
        """
        Detect and resolve overlapping n-grams (e.g., "machine" + "learning" vs "machine learning").
        Prioritizes longer n-grams over constituent parts when significant overlap is detected.
        """
        if len(keywords) <= 1:
            return keywords

        # Group keywords by length (in tokens)
        keywords_by_tokens = defaultdict(list)
        for kw in keywords:
            tokens = kw.split()
            keywords_by_tokens[len(tokens)].append(kw)

        # Sort by decreasing n-gram length
        lengths = sorted(keywords_by_tokens.keys(), reverse=True)

        # Track which keywords to keep
        to_keep = set()

        # Process longer n-grams first to identify subsumed shorter n-grams
        for length in lengths:
            for kw in keywords_by_tokens[length]:
                # If this keyword is already marked for removal, skip
                if kw not in to_keep:
                    # Check if this keyword should be kept
                    kw_tokens = set(kw.split())
                    is_subsumed = False

                    # Check if it's subsumed by a longer keyword that we're keeping
                    for longer_length in [l for l in lengths if l > length]:
                        for longer_kw in keywords_by_tokens[longer_length]:
                            if longer_kw in to_keep:
                                longer_tokens = set(longer_kw.split())
                                # Calculate token overlap
                                overlap_ratio = len(kw_tokens & longer_tokens) / len(
                                    kw_tokens
                                )
                                if overlap_ratio >= self.max_ngram_overlap:
                                    is_subsumed = True
                                    self.stats["duplicates_found"] += 1
                                    logger.debug(
                                        f"N-gram overlap: '{kw}' subsumed by '{longer_kw}'"
                                    )
                                    break
                        if is_subsumed:
                            break

                    if not is_subsumed:
                        to_keep.add(kw)

                        # Remove any shorter n-grams that are fully contained in this one
                        kw_tokens = set(kw.split())
                        for shorter_length in [l for l in lengths if l < length]:
                            for shorter_kw in keywords_by_tokens[shorter_length]:
                                if shorter_kw in to_keep:
                                    shorter_tokens = set(shorter_kw.split())
                                    if shorter_tokens.issubset(kw_tokens):
                                        to_keep.remove(shorter_kw)
                                        self.stats["duplicates_found"] += 1
                                        logger.debug(
                                            f"Removing subset n-gram: '{shorter_kw}' contained in '{kw}'"
                                        )

        return list(to_keep)

    def _cluster_similar_terms(
        self, keywords: List[str], all_skills: Set[str] = None
    ) -> List[str]:
        """
        Use embeddings to cluster similar terms and select a representative for each cluster.

        Args:
            keywords: List of normalized keywords
            all_skills: Set of known valid skills (for prioritizing representatives)

        Returns:
            List of canonical representatives
        """
        if len(keywords) <= 1:
            return keywords

        # 1. Get embeddings for all keywords
        vectors = []
        valid_keywords = []

        # Process in batches to avoid memory issues
        for i in range(0, len(keywords), self.embedding_batch_size):
            batch = keywords[i : i + self.embedding_batch_size]
            docs = list(self.nlp.pipe(batch))

            for kw, doc in zip(batch, docs):
                if doc.has_vector:
                    vectors.append(doc.vector)
                    valid_keywords.append(kw)
                else:
                    # Keep keywords without valid vectors as-is
                    logger.debug(f"No vector for keyword: '{kw}'")

        if len(valid_keywords) <= 1:
            return keywords  # Not enough valid vectors for clustering

        # 2. Convert to numpy array
        X = np.array(vectors)

        # 3. Normalize vectors
        norms = np.linalg.norm(X, axis=1)
        norms[norms == 0] = 1  # Avoid division by zero
        X = X / norms[:, np.newaxis]

        # 4. Perform DBSCAN clustering
        try:
            clustering = DBSCAN(
                eps=self.cluster_eps,
                min_samples=self.cluster_min_samples,
                metric="cosine",
            ).fit(X)

            # 5. Extract clusters
            labels = clustering.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            self.stats["clusters_formed"] += n_clusters

            # 6. Get representative for each cluster
            canonical_keywords = []

            # Add unclustered terms as-is
            for i, label in enumerate(labels):
                if label == -1:  # Noise points
                    canonical_keywords.append(valid_keywords[i])

            # Process each cluster to find a representative
            for cluster_id in range(n_clusters):
                cluster_indices = [
                    i for i, label in enumerate(labels) if label == cluster_id
                ]
                cluster_terms = [valid_keywords[i] for i in cluster_indices]

                # Select representative
                representative = self._select_cluster_representative(
                    cluster_terms, all_skills, [vectors[i] for i in cluster_indices]
                )
                canonical_keywords.append(representative)

                # Log the cluster if it has more than one term
                if len(cluster_terms) > 1:
                    logger.debug(
                        f"Cluster formed: {cluster_terms} -> '{representative}'"
                    )
                    self.stats["duplicates_found"] += len(cluster_terms) - 1

            # 7. Add keywords not included in clustering
            not_included = set(keywords) - set(valid_keywords)
            canonical_keywords.extend(not_included)

            return canonical_keywords

        except Exception as e:
            logger.error(f"Error during keyword clustering: {str(e)}")
            return list(set(keywords))  # Fallback to simple deduplication

    def _select_cluster_representative(
        self,
        cluster_terms: List[str],
        all_skills: Optional[Set[str]],
        vectors: List[np.ndarray],
    ) -> str:
        """
        Select the best representative term for a cluster.

        Selection criteria (in order):
        1. Prefer terms from the whitelist (if prioritize_whitelist is True)
        2. Prefer the longest term (which is often more specific)
        3. Prefer the term closest to the cluster centroid

        Args:
            cluster_terms: List of terms in the cluster
            all_skills: Set of known valid skills (optional)
            vectors: List of vector embeddings for the terms

        Returns:
            The selected representative term
        """
        # If only one term, it's the representative
        if len(cluster_terms) == 1:
            return cluster_terms[0]

        # Prioritize whitelist terms if enabled
        if self.prioritize_whitelist and all_skills:
            whitelist_terms = [t for t in cluster_terms if t.lower() in all_skills]
            if whitelist_terms:
                cluster_terms = whitelist_terms
                # Use only vectors for whitelist terms
                indices = [
                    i
                    for i, term in enumerate(cluster_terms)
                    if term.lower() in all_skills
                ]
                vectors = [vectors[i] for i in indices]

        # If still multiple candidates, prefer longer terms
        max_length = max(len(term) for term in cluster_terms)
        longest_terms = [term for term in cluster_terms if len(term) == max_length]

        if len(longest_terms) == 1:
            return longest_terms[0]

        # If still ties, select term closest to centroid
        try:
            centroid = np.mean(vectors, axis=0)
            distances = [cosine(v, centroid) for v in vectors]
            closest_idx = np.argmin(distances)
            return cluster_terms[closest_idx]
        except Exception as e:
            logger.warning(f"Error selecting cluster representative: {str(e)}")
            # Fallback to first term
            return cluster_terms[0]

    def get_stats(self) -> Dict[str, int]:
        """Return statistics about the canonicalization process."""
        return self.stats.copy()

    def reset_stats(self) -> None:
        """Reset the statistics counters."""
        for key in self.stats:
            self.stats[key] = 0
