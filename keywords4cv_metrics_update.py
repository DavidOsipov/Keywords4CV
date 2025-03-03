"""
Enhanced version of run_analysis with improved metrics reporting
"""

from pathlib import Path
from typing import Dict
import structlog
import sys

from keywords4cv import initialize_analyzer, save_results, load_job_data
from metrics_reporter import generate_metrics_report
from exceptions import ConfigError, InputValidationError, DataIntegrityError

logger = structlog.get_logger()


def run_analysis_with_metrics(args):
    analyzer = initialize_analyzer(args.config)
    jobs = load_job_data(args.input)
    report_path = None

    try:
        # 1. Analyze jobs (this will save intermediate files and checksums if enabled)
        analyzer.analyze_jobs(jobs)

        # 2. Determine the number of batches
        batch_count = 0
        if analyzer.config["intermediate_save"]["enabled"]:
            format_type = analyzer.config["intermediate_save"]["format"]
            suffix = {"feather": ".feather", "jsonl": ".jsonl", "json": ".json"}.get(
                format_type, ".json"
            )
            while (
                analyzer.working_dir
                / f"{analyzer.run_id}_chunk_summary_{batch_count}{suffix}"
            ).exists():
                batch_count += 1

            # 3. Verify checksums BEFORE loading
            analyzer._verify_intermediate_checksums()

        # 4. Load all intermediate results as a generator
        loaded_results_generator = analyzer._load_all_intermediate(batch_count)

        # 5. Aggregate the results using streaming aggregation
        final_summary, final_details = analyzer._aggregate_results(
            loaded_results_generator
        )

        # 6. Save to Excel
        save_results(final_summary, final_details, args.output)

        # 7. Generate comprehensive metrics report
        metrics_dir = Path(args.output).parent / "metrics_reports"
        report_path = generate_metrics_report(
            analyzer=analyzer,
            summary_df=final_summary,
            details_df=final_details,
            output_dir=str(metrics_dir),
        )

        logger.info(
            "Analysis complete with enhanced metrics",
            excel_output=args.output,
            metrics_report=report_path,
        )

    except DataIntegrityError as e:
        logger.error("Data integrity error: %s", e)
        sys.exit(75)  # Use a specific exit code for data integrity issues
    finally:
        analyzer._cleanup_intermediate()

    return final_summary, final_details, report_path


def main():
    """
    Entry point function for the program with enhanced metrics reporting.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="ATS Keyword Optimizer with Enhanced Metrics"
    )
    parser.add_argument(
        "-i", "--input", default="job_descriptions.json", help="Input JSON file"
    )
    parser.add_argument("-c", "--config", default="config.yaml", help="Config file")
    parser.add_argument("-o", "--output", default="results.xlsx", help="Output file")
    parser.add_argument(
        "--metrics-report",
        action="store_true",
        help="Generate comprehensive metrics report",
    )
    args = parser.parse_args()

    try:
        if args.metrics_report:
            summary, details, report_path = run_analysis_with_metrics(args)
            if report_path:
                print(f"\nComprehensive metrics report generated at: {report_path}")
        else:
            from keywords4cv import run_analysis

            run_analysis(args)

    except ConfigError as e:
        logger.error("Configuration error: %s", e)
        sys.exit(78)
    except InputValidationError as e:
        logger.error("Input validation error: %s", e)
        sys.exit(77)
    except MemoryError as e:
        logger.error("Memory error: %s", e)
        sys.exit(70)
    except Exception as e:
        logger.exception("Unhandled exception: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
