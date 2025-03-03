"""
Metrics Evaluation Module for Keywords4CV

This module provides additional evaluation metrics and utilities for assessing
keyword extraction performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, Set, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score


class KeywordMetricsEvaluator:
    """
    Evaluates keyword extraction performance with multiple metrics.

    This class provides methods to calculate and visualize various metrics
    for keyword extraction, including precision, recall, F1-score, and
    more advanced metrics like Mean Average Precision.
    """

    def __init__(self, original_skills: Set[str], expanded_skills: Set[str]):
        """
        Initialize with reference skill sets.

        Args:
            original_skills: Set of original skills from the configuration
            expanded_skills: Set of expanded skills (including synonyms and variants)
        """
        self.original_skills = set(s.lower() for s in original_skills)
        self.expanded_skills = set(s.lower() for s in expanded_skills)

    def calculate_basic_metrics(self, extracted_keywords: Set[str]) -> Dict:
        """
        Calculate basic keyword extraction metrics.

        Args:
            extracted_keywords: Set of keywords extracted by the system

        Returns:
            Dictionary containing precision, recall, and F1 metrics
        """
        # Convert to lowercase for case-insensitive comparison
        extracted = set(k.lower() for k in extracted_keywords)

        # Metrics against original skills
        original_recall = (
            len(extracted & self.original_skills) / len(self.original_skills)
            if self.original_skills
            else 0
        )

        # Metrics against expanded skills
        expanded_recall = (
            len(extracted & self.expanded_skills) / len(self.expanded_skills)
            if self.expanded_skills
            else 0
        )

        precision = (
            len(extracted & self.expanded_skills) / len(extracted) if extracted else 0
        )

        # Calculate F1 scores
        original_f1 = (
            2 * precision * original_recall / (precision + original_recall)
            if precision + original_recall > 0
            else 0
        )

        expanded_f1 = (
            2 * precision * expanded_recall / (precision + expanded_recall)
            if precision + expanded_recall > 0
            else 0
        )

        # Add coverage metric: how many categories are represented in the extracted keywords
        coverage = self._calculate_category_coverage(extracted)

        return {
            "original_recall": original_recall,
            "expanded_recall": expanded_recall,
            "precision": precision,
            "original_f1": original_f1,
            "expanded_f1": expanded_f1,
            "category_coverage": coverage,
        }

    def _calculate_category_coverage(self, extracted_keywords: Set[str]) -> float:
        """
        Calculate what percentage of skill categories are represented in extracted keywords.

        This is a placeholder implementation that should be customized based on your
        specific category structure.

        Args:
            extracted_keywords: Set of extracted keywords

        Returns:
            Coverage percentage (0.0 to 1.0)
        """
        # Implement based on your category structure
        # This is just a placeholder - implement according to your needs
        return 0.0

    def visualize_metrics(self, metrics_history: List[Dict], output_file: str = None):
        """
        Visualize metrics over time or across different configurations.

        Args:
            metrics_history: List of metric dictionaries from different runs
            output_file: Optional path to save the visualization
        """
        if not metrics_history:
            return

        # Convert metrics history to DataFrame
        df = pd.DataFrame(metrics_history)

        # Create visualization
        plt.figure(figsize=(10, 6))

        # Plot metrics
        for metric in [
            "precision",
            "original_recall",
            "expanded_recall",
            "expanded_f1",
        ]:
            if metric in df.columns:
                plt.plot(df.index, df[metric], label=metric)

        plt.title("Keyword Extraction Performance Metrics")
        plt.xlabel("Run")
        plt.ylabel("Score")
        plt.legend()
        plt.grid(True, alpha=0.3)

        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
        else:
            plt.show()

    def calculate_advanced_metrics(
        self, extracted_keywords: List[Tuple[str, float]]
    ) -> Dict:
        """
        Calculate advanced metrics for ranked keyword extraction.

        Args:
            extracted_keywords: List of tuples with (keyword, confidence_score)

        Returns:
            Dictionary of advanced metrics
        """
        # Sort by confidence score
        extracted_keywords = sorted(
            extracted_keywords, key=lambda x: x[1], reverse=True
        )

        # Calculate Mean Average Precision
        y_true = [
            1 if k[0].lower() in self.expanded_skills else 0 for k in extracted_keywords
        ]
        y_scores = [k[1] for k in extracted_keywords]

        if sum(y_true) == 0:  # No relevant keywords found
            return {"mean_avg_precision": 0.0}

        try:
            mean_avg_precision = average_precision_score(y_true, y_scores)
        except:
            mean_avg_precision = 0.0

        return {"mean_avg_precision": mean_avg_precision}


def compare_metrics_across_runs(run_metrics: Dict[str, Dict]) -> pd.DataFrame:
    """
    Compare metrics across different runs or configurations.

    Args:
        run_metrics: Dictionary mapping run_id to metrics dictionary

    Returns:
        DataFrame with comparative metrics
    """
    # Create DataFrame with metrics from all runs
    data = []
    for run_id, metrics in run_metrics.items():
        metrics_copy = metrics.copy()
        metrics_copy["run_id"] = run_id
        data.append(metrics_copy)

    return pd.DataFrame(data)


def plot_metrics_comparison(metrics_df: pd.DataFrame, output_file: str = None):
    """
    Create a comparison plot of metrics across runs.

    Args:
        metrics_df: DataFrame with metrics from compare_metrics_across_runs
        output_file: Optional path to save the plot
    """
    # Select numeric columns for plotting
    numeric_cols = metrics_df.select_dtypes(
        include=["float64", "int64"]
    ).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != "run_id"]

    if not numeric_cols:
        return

    # Melt the DataFrame for easier plotting
    melted_df = pd.melt(
        metrics_df,
        id_vars=["run_id"],
        value_vars=numeric_cols,
        var_name="Metric",
        value_name="Value",
    )

    # Create plot
    plt.figure(figsize=(12, 8))

    sns.barplot(x="run_id", y="Value", hue="Metric", data=melted_df)

    plt.title("Metrics Comparison Across Runs")
    plt.xlabel("Run")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.legend(title="Metric")
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
    else:
        plt.show()
