"""
Metrics Reporter for Keywords4CV

This module provides functionality to generate comprehensive metrics reports
for keyword extraction performance.
"""

import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set
import matplotlib.pyplot as plt
import seaborn as sns

from metrics_evaluation import (
    KeywordMetricsEvaluator,
    compare_metrics_across_runs,
    plot_metrics_comparison,
)


class MetricsReporter:
    """
    Generates comprehensive metrics reports for keyword extraction performance.
    """

    def __init__(self, output_dir: str = None):
        """
        Initialize the metrics reporter.

        Args:
            output_dir: Directory where reports will be stored
        """
        self.output_dir = output_dir or "metrics_reports"
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_report(
        self,
        run_id: str,
        metrics: Dict,
        summary_df: pd.DataFrame,
        expanded_skills: Set[str],
        original_skills: Set[str],
    ) -> str:
        """
        Generate a comprehensive metrics report.

        Args:
            run_id: Identifier for this analysis run
            metrics: Dictionary of calculated metrics
            summary_df: DataFrame with keyword summary
            expanded_skills: Set of expanded skills used
            original_skills: Set of original skills

        Returns:
            Path to the generated report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = Path(self.output_dir) / f"report_{run_id}_{timestamp}"
        report_dir.mkdir(exist_ok=True)

        # Save metrics to JSON
        metrics_path = report_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        # Create plots directory
        plots_dir = report_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        # Generate keyword distribution plots
        self._generate_keyword_distribution_plot(
            summary_df, plots_dir / "keyword_distribution.png"
        )

        # Generate category distribution plot if category information is available
        if "Category" in summary_df.columns:
            self._generate_category_distribution_plot(
                summary_df, plots_dir / "category_distribution.png"
            )

        # Generate skill coverage plot
        self._generate_skill_coverage_plot(
            summary_df.index.to_list(),
            original_skills,
            expanded_skills,
            plots_dir / "skill_coverage.png",
        )

        # Generate HTML report
        report_html_path = report_dir / "report.html"
        self._generate_html_report(
            report_html_path, run_id, metrics, summary_df, plots_dir
        )

        return str(report_html_path)

    def _generate_keyword_distribution_plot(
        self, summary_df: pd.DataFrame, output_path: str
    ):
        """Generate plot showing distribution of keyword scores"""
        plt.figure(figsize=(10, 6))

        # Plot distribution of Total_Score
        if "Total_Score" in summary_df.columns:
            sns.histplot(summary_df["Total_Score"], bins=30, kde=True)
            plt.title("Distribution of Keyword Scores")
            plt.xlabel("Score")
            plt.ylabel("Count")
            plt.grid(True, alpha=0.3)
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

    def _generate_category_distribution_plot(
        self, summary_df: pd.DataFrame, output_path: str
    ):
        """Generate plot showing distribution of keywords across categories"""
        if "Category" not in summary_df.columns:
            return

        # Get top 10 categories by count
        category_counts = summary_df["Category"].value_counts().nlargest(10)

        plt.figure(figsize=(10, 6))
        sns.barplot(x=category_counts.values, y=category_counts.index)
        plt.title("Top 10 Keyword Categories")
        plt.xlabel("Count")
        plt.ylabel("Category")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def _generate_skill_coverage_plot(
        self,
        extracted_keywords: List[str],
        original_skills: Set[str],
        expanded_skills: Set[str],
        output_path: str,
    ):
        """Generate plot showing coverage of skills"""
        extracted_lower = [k.lower() for k in extracted_keywords]

        # Calculate overlaps
        original_overlap = len(set(extracted_lower) & original_skills)
        expanded_overlap = len(set(extracted_lower) & expanded_skills)
        expanded_only = len(expanded_skills - original_skills)

        # Create data
        labels = ["Original Skills", "Expanded Only", "Extracted Skills"]
        values = [len(original_skills), expanded_only, len(extracted_keywords)]
        overlaps = [original_overlap, expanded_overlap - original_overlap, 0]

        plt.figure(figsize=(10, 6))

        # Create bar chart with overlap indicators
        bars = plt.bar(labels, values)
        for i, (bar, overlap) in enumerate(zip(bars, overlaps)):
            if overlap > 0:
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 5,
                    f"Overlap: {overlap}",
                    ha="center",
                )

        plt.title("Skill Coverage Analysis")
        plt.ylabel("Count")
        plt.grid(axis="y", alpha=0.3)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def _generate_html_report(
        self,
        output_path: str,
        run_id: str,
        metrics: Dict,
        summary_df: pd.DataFrame,
        plots_dir: Path,
    ):
        """Generate HTML report with all metrics and visualizations"""
        # Create relative paths for plots
        plots_relative = os.path.relpath(plots_dir, os.path.dirname(output_path))

        # Start building HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Keyword Extraction Metrics Report: {run_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #2c3e50; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ text-align: left; padding: 8px; }}
                tr:nth-child(even) {{ background-color: #f2f2f2 }}
                th {{ background-color: #2c3e50; color: white; }}
                .metrics-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; }}
                .metric-card {{ background-color: #f8f9fa; border-radius: 5px; padding: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; margin: 10px 0; }}
                .plot-container {{ margin: 20px 0; }}
                .plot-container img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>Keyword Extraction Metrics Report</h1>
            <p><strong>Run ID:</strong> {run_id}</p>
            <p><strong>Date:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <h2>Performance Metrics</h2>
            <div class="metrics-grid">
        """

        # Add metric cards
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                formatted_value = (
                    f"{value:.3f}" if isinstance(value, float) else str(value)
                )
                html_content += f"""
                <div class="metric-card">
                    <h3>{key.replace("_", " ").title()}</h3>
                    <div class="metric-value">{formatted_value}</div>
                </div>
                """

        html_content += """
            </div>
            
            <h2>Visualizations</h2>
        """

        # Add plots
        plots = [
            ("Keyword Score Distribution", "keyword_distribution.png"),
            ("Category Distribution", "category_distribution.png"),
            ("Skill Coverage", "skill_coverage.png"),
        ]

        for title, filename in plots:
            plot_path = os.path.join(plots_relative, filename)
            if os.path.exists(os.path.join(plots_dir, filename)):
                html_content += f"""
                <div class="plot-container">
                    <h3>{title}</h3>
                    <img src="{plot_path}" alt="{title}">
                </div>
                """

        # Add top keywords table
        top_n = min(50, len(summary_df))
        if not summary_df.empty:
            html_content += f"""
            <h2>Top {top_n} Keywords</h2>
            <table>
                <tr>
            """

            # Add table headers
            for col in summary_df.reset_index().columns[:5]:  # Limit to first 5 columns
                html_content += f"<th>{col}</th>"

            html_content += "</tr>"

            # Add table rows
            for _, row in summary_df.reset_index().head(top_n).iterrows():
                html_content += "<tr>"
                for col in row.index[:5]:  # Limit to first 5 columns
                    value = row[col]
                    if isinstance(value, float):
                        formatted_value = f"{value:.3f}"
                    else:
                        formatted_value = str(value)
                    html_content += f"<td>{formatted_value}</td>"
                html_content += "</tr>"

            html_content += "</table>"

        # Complete HTML document
        html_content += """
        </body>
        </html>
        """

        # Write HTML file
        with open(output_path, "w") as f:
            f.write(html_content)


def generate_metrics_report(
    analyzer, summary_df: pd.DataFrame, details_df: pd.DataFrame, output_dir: str = None
) -> str:
    """
    Generate a comprehensive metrics report for a keyword extraction run.

    Args:
        analyzer: The OptimizedATS instance
        summary_df: DataFrame with keyword summary
        details_df: DataFrame with detailed keyword scores
        output_dir: Optional directory to store the report

    Returns:
        Path to the generated report
    """
    # Create metrics evaluator
    evaluator = KeywordMetricsEvaluator(
        original_skills=set().union(*analyzer.config["keyword_categories"].values()),
        expanded_skills=analyzer.keyword_extractor.all_skills,
    )

    # Calculate comprehensive metrics
    extracted_keywords = set(summary_df.index)
    metrics = evaluator.calculate_basic_metrics(extracted_keywords)

    # Add advanced metrics if scores are available
    if "Total_Score" in summary_df.columns:
        keyword_scores = [
            (k, s) for k, s in zip(summary_df.index, summary_df["Total_Score"])
        ]
        advanced_metrics = evaluator.calculate_advanced_metrics(keyword_scores)
        metrics.update(advanced_metrics)

    # Generate report
    reporter = MetricsReporter(output_dir)
    report_path = reporter.generate_report(
        run_id=analyzer.run_id,
        metrics=metrics,
        summary_df=summary_df,
        expanded_skills=analyzer.keyword_extractor.all_skills,
        original_skills=set().union(*analyzer.config["keyword_categories"].values()),
    )

    return report_path
