# Keywords4CV

[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

## Description

Keywords4CV is a Python-based tool designed to help job seekers optimize their resumes and LinkedIn profiles for Applicant Tracking Systems (ATS) and human recruiters. It analyzes a collection of job descriptions and extracts the most important and relevant keywords, enabling users to tailor their application materials to specific job roles. By incorporating these keywords, users can significantly increase their chances of getting noticed by both ATS and recruiters, leading to more interview opportunities.

## State of Production

**UNDER ACTIVE DEVELOPMENT**

Currently at version `0.26.0 (Alpha)`. This is a pre-release version with significant architectural changes, performance improvements, and new features. While this version is a substantial improvement over 0.24, it is still under active development and may contain critical bugs. Haven't been tested whether it's functional at this point of time.

## Features

*   **Enhanced Keyword Extraction:** Identifies key skills, qualifications, and terminology from job descriptions using advanced NLP techniques (spaCy, NLTK, scikit-learn, rapidfuzz).
    *   Preserves multi-word skills (e.g., "machine learning") through entity recognition.
    *   **Fuzzy Matching:** Integrates `rapidfuzz` for flexible keyword matching, handling variations in spelling and phrasing.  Uses an enhanced BK-Tree implementation with caching for improved performance.
    *   **Phrase-Level Synonyms:** Supports synonyms for multi-word phrases, loaded from a static file or a REST API (with caching, retries, and a circuit breaker).
    *   **Configurable Processing Order:** Option to apply fuzzy matching before or after semantic validation.
    *   **Trigram Optimization:** Uses a trigram cache to improve n-gram generation efficiency.
    *   **Dynamic N-gram Generation:** Improved n-gram handling with robust error handling.
    *   **Keyword Canonicalization:** Deduplicates and canonicalizes keywords using abbreviation expansion and embedding-based clustering.
*   **TF-IDF Analysis with Advanced Weighting:** Computes Term Frequency-Inverse Document Frequency (TF-IDF) scores, combined with *whitelist boosting*, *keyword frequency*, and *section weighting*. Configurable weights allow for fine-tuning keyword importance.
*   **Synonym Expansion:** Leverages WordNet and, optionally, a REST API or static file to suggest synonyms, expanding keyword coverage and improving semantic matching.
*   **Semantic Keyword Categorization:** Assigns keywords to categories using a hybrid approach:
    *   Direct matching against pre-defined category terms.
    *   Semantic similarity (cosine similarity between word vectors) for terms not directly matched.
    *   Configurable `default_category` for uncategorized terms.
    *   Caching for improved performance.
*   **Contextual Validation:** Uses a configurable context window to determine if a keyword is used in a relevant context within the job description, reducing false positives. Improved sentence splitting handles bullet points and numbered lists.
*   **Highly Configurable:** Uses a `config.yaml` file for extensive customization:
    *   **Validation:** Settings for input validation (e.g., allowing numeric titles, handling empty descriptions).
    *   **Dataset:** Parameters for controlling dataset processing (e.g., minimum/maximum job descriptions, short description threshold).
    *   **Text Processing:**
        *   spaCy model selection and pipeline component configuration.
        *   N-gram ranges (for general keywords and the whitelist).
        *   POS tag filtering.
        *   Semantic validation settings (enabled/disabled, similarity threshold).
        *   Synonym source (static file or API) and related settings.
        *   Context window size.
        *   Fuzzy matching order (before/after semantic validation).
        *   WordNet similarity threshold
    *   **Categorization:** Default category, categorization cache size, direct match threshold.
    *   **Whitelist:** Whitelist recall threshold, caching options, fuzzy matching settings (algorithm, minimum similarity, allowed POS tags).
    *   **Weighting:** Weights for TF-IDF, frequency, whitelist boost, and section-specific weights.
    *   **Hardware Limits:** GPU usage, batch size, memory thresholds, maximum workers, memory scaling factor.
    *   **Optimization:** Settings for reinforcement learning (Q-learning) based parameter tuning, including reward weights, learning rate, and complexity factors.
    *   **Caching:** Cache sizes for various components (general cache, TF-IDF max features, trigram cache, term cache, BK-tree cache). Includes a `cache_salt` for cache invalidation.
    *   **Intermediate Save:** Options for saving intermediate results (enabled/disabled, save interval, format, working directory, cleanup).
    *   **Advanced:** Dask integration (currently disabled), success rate threshold, checksum relative tolerance, negative keywords, section headings.
    *   Stop words (with options to add and exclude specific words).
    *   Extensive and customizable whitelist of technical and soft skills.
    *   Keyword categories.
*   **Detailed Output Reports:** Generates comprehensive Excel reports:
    *   **Keyword Summary:** Aggregated keyword scores, job counts, average scores, and assigned categories for a high-level overview.
    *   **Job Specific Details:** Detailed table showing keyword scores, TF-IDF, frequency, category, whitelist status, and other relevant information per job title.
*   **Comprehensive Metrics Reports:** Generates detailed reports with visualizations, including:
    * Precision, recall, F1-score (original and expanded).
    * Category coverage.
    * Mean Average Precision (mAP).
    * Keyword and category distribution plots.
    * Skill coverage plots.
    * HTML report summarizing metrics and visualizations.
*   **Robust Input Validation:** Rigorous validation of job descriptions and configuration parameters, handling empty titles, descriptions, incorrect data types, encoding issues, and invalid configuration values. Clear error messages and logging.
*   **User-Friendly Command-Line Interface:** `argparse` provides a clear and easy-to-use interface.  Includes an option to generate metrics reports.
*   **Comprehensive Error Handling and Logging:** Detailed logging to `Keywords4CV.log` with improved error handling for configuration, input, memory issues, API calls, data integrity, and other potential problems. Custom exceptions are used for specific error types.
*   **Multiprocessing for Parallel Processing:** Core analysis uses `concurrent.futures.ProcessPoolExecutor` for parallel processing, significantly improving performance.  Optimized spaCy model loading in worker processes.
*   **Efficient Caching:** Uses `functools.lru_cache`, `cachetools.LRUCache`, and a custom `CacheManager` to optimize performance, with cache invalidation on configuration changes.
*   **SpaCy Pipeline Optimization:** Dynamically enables and disables spaCy pipeline components based on configuration, improving efficiency.
*   **Automatic NLTK Resource Management:** Ensures WordNet and other NLTK resources are downloaded if missing.
*   **Memory Management and Adaptive Chunking:**
    *   **Smart Chunker:** Uses a Q-learning algorithm to dynamically adjust the chunk size based on dataset statistics and system resource usage.
    *   **Auto Tuner:** Automatically adjusts parameters (e.g., `chunk_size`, `pos_processing`) based on metrics and trigram cache hit rate.
    *   **Memory Monitoring:** Monitors memory usage and adapts processing accordingly.
    *   **Explicit Garbage Collection:** Releases memory proactively.
*   **Intermediate Result Saving and Checkpointing:** Saves intermediate results to disk in configurable formats (feather, jsonl, or json) with checksum verification to ensure data integrity. This allows for resuming processing and prevents data loss.
* **Streaming Data Aggregation:** Uses a generator-based approach to aggregate results from intermediate files, enabling processing of very large datasets.
* **Modular Design:** The codebase is split into well-defined modules for improved organization and maintainability.
* **Configuration Validation:** Uses `schema` and `pydantic` for robust configuration validation.

## Changelog (v0.26.0)

See the separate `CHANGELOG.md` file for a detailed list of changes.  This release includes a *major* refactoring and many new features.

## How it Works

1.  **Input:** Accepts a JSON file (e.g., `job_descriptions.json`) with job titles as keys and descriptions as values. Also uses a `config.yaml` file for configuration.
2.  **Configuration Validation:** Validates the structure and content of the `config.yaml` file using `schema` and Pydantic.
3.  **Preprocessing:** Cleans text (lowercasing, URL/email removal), tokenizes, lemmatizes, and caches results.
4.  **Keyword Extraction:**
    *   Matches `keyword_categories` phrases as `SKILL` entities using spaCy's `entity_ruler`.
    *   Generates n-grams (configurable range).
    *   Filters out noisy n-grams (containing stop words or single characters).
    *   Expands keywords with synonyms (from WordNet, a static file, or a REST API).
    *   Applies fuzzy matching (using `rapidfuzz` and an enhanced BK-tree) against the expanded set of skills.
    *   Performs semantic validation, checking if keywords are used in context.
    *   Canonicalizes keywords to remove duplicates and handle variations.
5.  **Keyword Weighting and Scoring:** Combines TF-IDF, frequency, whitelist boost, and section weighting (configurable weights).
6.  **Keyword Categorization:** Assigns categories using a hybrid approach (direct match + semantic similarity).
7.  **Intermediate Saving (Optional):** Saves intermediate results (summary and detailed scores) to disk in a configurable format (feather, jsonl, or json) with checksum verification.
8.  **Adaptive Parameter Tuning:** Uses reinforcement learning (Q-learning) to dynamically adjust parameters (e.g., chunk size, POS processing strategy) based on performance metrics.
9.  **Result Aggregation:** Combines intermediate results (if any) into final summary and detailed DataFrames.
10. **Output Generation:** Produces Excel reports and, optionally, a comprehensive metrics report (HTML with visualizations).

[Much more detailed description here](https://github.com/DavidOsipov/Keywords4CV/wiki/How-script-v0.26-works)

## Getting Started

### Prerequisites

*   Python 3.8+
*   Required libraries:
    ```bash
    pip install pandas nltk spacy scikit-learn pyyaml psutil requests rapidfuzz srsly xxhash cachetools pydantic schema pyarrow numpy structlog tenacity
    ```
*   SpaCy model:
    ```bash
    python -m spacy download en_core_web_lg
    ```
    (or another suitable spaCy model, specified in `config.yaml`)

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/DavidOsipov/Keywords4Cv.git
    cd Keywords4Cv
    ```
2.  (Recommended) Create a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # Linux/macOS
    venv\Scripts\activate     # Windows
    ```
3.  Install dependencies (see Prerequisites).

### Usage

1.  **Prepare Input:** Create `job_descriptions.json`:
    ```json
    {
      "Data Scientist": "Experience in Python, machine learning, and SQL...",
      "Product Manager": "Agile methodologies, product roadmapping...",
      "Software Engineer": "Proficient in Java, Spring, REST APIs..."
    }
    ```
2.  **Customize `config.yaml`:** **Thoroughly review and adjust the `config.yaml` file.** This is crucial for the script's behavior. Pay particular attention to:
    *   `keyword_categories`: Define your skill categories and list relevant keywords for each.
    *   `text_processing`: Configure spaCy model, n-gram ranges, synonym settings, fuzzy matching, and contextual validation.
    *   `whitelist`: Adjust fuzzy matching parameters.
    *   `weighting`: Set weights for TF-IDF, frequency, whitelist boost, and section weights.
    *   `hardware_limits`: Configure memory usage and processing limits.
    *   `optimization`: Tune reinforcement learning parameters (if desired).
    *   `intermediate_save`: Enable/disable intermediate saving and configure related options.
3.  **(Optional) Create `synonyms.json`:** If using static phrase synonyms, create a `synonyms.json` file:
    ```json
    {
      "product management": ["product leadership", "product ownership"],
      "machine learning": ["ml", "ai"]
    }
    ```
4.  **Run the Script:**
    ```bash
    python keywords4cv.py -i job_descriptions.json -c config.yaml -o results.xlsx
    ```
   To generate a comprehensive metrics report, use the `--metrics-report` flag:
    ```bash
   python keywords4cv_metrics_update.py -i job_descriptions.json -c config.yaml -o results.xlsx --metrics-report
    ```
5.  **Review Output:** Check `results.xlsx` and `Keywords4CV.log`. If `--metrics-report` was used, a `metrics_reports` directory will be created containing the HTML report and visualizations.

## Repository Structure

*   `keywords4cv.py`: Main script.
*   `config.yaml`: Configuration file.
*   `README.md`: This documentation.
*   `exceptions.py`: Custom exception definitions.
*   `config_validation.py`: Configuration validation logic.
*   `keyword_canonicalizer.py`: Keyword canonicalization logic.
*   `cache_manager.py`: Cache management utilities.
*   `validation_utils.py`: Semantic validation utilities.
*   `bk_tree_enhancement.py`: Enhanced BK-tree implementation.
*   `multiprocess_helpers.py`: Multiprocessing helper functions.
*   `metrics_evaluation.py`: Metrics evaluation utilities.
*   `metrics_reporter.py`: Metrics reporting utilities.
*   `keywords4cv_metrics_update.py`: Entry point for running analysis with metrics reporting.
*   `requirements.txt`: Dependency list.
*   `job_descriptions.json`: Sample input.
*   `synonyms.json`: (Optional) Static phrase synonyms.
*   `working_dir`: (Created by the script) Stores intermediate files.
*   `metrics_reports`: (Created by the script when `--metrics-report` is used) Stores metrics reports.

## Contributing

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes.
4.  Add or update tests as needed.
5.  Run the tests and ensure they pass.
6.  Commit your changes with clear commit messages.
7.  Submit a pull request.

## Licensing

Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

**Important Note about Software Licensing:**

While the CC BY-NC-SA 4.0 license is generally used for creative works, it is being used here to specifically restrict the commercial use of this software by others while allowing non-commercial use, modification, and sharing. The author chose this license to foster a community of non-commercial users and contributors while retaining the right to commercialize the software themselves.

**Please be aware that this license is not a standard software license and may not address all software-specific legal concerns.** Specifically:

*   **No Patent Grant:** This license does *not* grant any patent rights related to the software.
*   **Disclaimer of Warranties:** THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*   **Consult Legal Counsel:** If you have any concerns about the legal implications of using this software under the CC BY-NC-SA 4.0 license, especially regarding patent rights or commercial use, you should consult with legal counsel.
