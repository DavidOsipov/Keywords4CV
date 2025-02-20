# Keywords4CV

[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

## Description

Keywords4CV is a Python-based tool designed to help job seekers optimize their resumes and LinkedIn profiles for Applicant Tracking Systems (ATS) and human recruiters. It analyzes a collection of job descriptions and extracts the most important and relevant keywords, enabling users to tailor their application materials to specific job roles. By incorporating these keywords, users can significantly increase their chances of getting noticed by both ATS and recruiters, leading to more interview opportunities.


## State of production

**Not ready!  Under active development.**

## Features

*   **Enhanced Keyword Extraction:** Identifies key skills, qualifications, and terminology from job descriptions using advanced NLP techniques (spaCy, NLTK, scikit-learn).  Improved accuracy and handling of edge cases.
*   **TF-IDF Analysis with Advanced Weighting:** Computes Term Frequency-Inverse Document Frequency (TF-IDF) scores, combined with *whitelist boosting* and *keyword frequency*.  Configurable weights allow for fine-tuning keyword importance.
*   **Synonym Expansion:**  Leverages WordNet to suggest synonyms, expanding keyword coverage and improving semantic matching.
*   **Highly Configurable:** Uses a `config.yaml` file for extensive customization:
    *   Stop words (with options to add and exclude specific words).  More comprehensive default stop word list.
    *   Skills whitelist:  Extensive and customizable whitelist of technical and soft skills.
    *   Keyword categories and semantic categorization: Group keywords into meaningful categories for better report organization. Semantic categorization using word vectors for improved accuracy.
    *   Output directory: Control where reports and logs are saved.
    *   N-gram range for keyword extraction: Adjust for unigrams, bigrams, trigrams, and more.
    *   Similarity threshold for categorization: Fine-tune semantic category assignment.
    *   Weighting for TF-IDF, frequency, and whitelist boost:  Customize the scoring mechanism.
    *   Description length limits and job count thresholds: Configure input validation rules.
*   **Detailed Output Reports:** Generates comprehensive Excel reports:
    *   **Keyword Summary:**  Aggregated keyword scores, job counts, average scores, and assigned categories, providing a high-level overview.
    *   **Job Specific Details:** Pivot table showing keyword scores for each job title, enabling targeted resume customization.
*   **Robust Input Validation:**  Rigorous validation of job descriptions, handling empty titles, descriptions, incorrect data types, and encoding issues.  Clear error messages and logging for invalid inputs.
*   **User-Friendly Command-Line Interface:**  `argparse` provides a clear and easy-to-use command-line interface for script execution.
*   **Comprehensive Error Handling and Logging:**  Detailed logging using `logging` module, capturing informational messages, warnings, and errors to `ats_optimizer.log`.  Improved error handling for configuration issues, input validation failures, and unexpected exceptions with informative exit codes.
*   **Multiprocessing for Robust Analysis:** Core analysis runs in a separate process using `multiprocessing`, enhancing stability and enabling timeout control, especially for large datasets.
*   **Efficient Caching:**  Extensive caching mechanisms using `functools.lru_cache` and `OrderedDict` to minimize redundant computations during text preprocessing and keyword categorization, significantly improving performance. Cache invalidation on configuration changes.
*   **SpaCy Pipeline Optimization:** Disables unnecessary spaCy pipeline components (`parser`, `ner`) for increased efficiency and reduced memory footprint.
*   **Automatic NLTK Resource Management:**  Ensures required NLTK resources (WordNet) are automatically downloaded if missing, simplifying setup.
*   **Configurable Timeout Mechanism:** Prevents script hangs and resource exhaustion with a configurable timeout, ensuring stability even with problematic inputs or large datasets.
*   **Memory Management and Chunking:** Implements memory monitoring and adaptive chunking for processing large job description datasets, preventing memory errors.  Caches are cleared proactively to manage memory usage.
*   **Extensive Unit Testing:**  Includes a comprehensive suite of unit tests using `unittest` and `pytest` in the `tests/` directory, ensuring code quality, reliability, and facilitating future development.  Tests cover core functionalities, edge cases, and configuration scenarios.

## How it Works

1.  **Input:** The script accepts a JSON file (e.g., `job_descriptions.json`) as input via the command line. The JSON should contain:
    *   **Keys:** Job titles (strings).
    *   **Values:** Full text of the corresponding job descriptions (strings).
2.  **Preprocessing:** Input job descriptions undergo thorough preprocessing:
    *   Text cleaning: Lowercasing, removal of URLs, emails, special characters, and extra whitespace using regular expressions.
    *   Tokenization and Lemmatization:  spaCy and NLTK are used for tokenization and lemmatization, reducing words to their base forms.
    *   Stop word removal: Configurable stop word list is used to remove common and irrelevant words.
    *   Caching: Preprocessed text is cached for efficiency using an LRU cache.
3.  **Keyword Extraction:**  Advanced keyword extraction techniques are employed:
    *   Whitelist Matching:  Exact and lemmatized phrase matching against the extensive `skills_whitelist` in `config.yaml`.
    *   N-gram Generation: Generates n-grams (unigrams, bigrams, trigrams, etc.) based on the `ngram_range` configuration.
    *   Synonym Expansion:  WordNet is used to generate synonyms for whitelist skills, expanding keyword detection.
4.  **Keyword Weighting and Scoring:** A sophisticated scoring system determines keyword relevance:
    *   TF-IDF Calculation:  `TfidfVectorizer` from scikit-learn calculates TF-IDF scores, reflecting keyword importance across job descriptions.
    *   Whitelist Boosting: Keywords from the `skills_whitelist` receive a configurable boost, prioritizing essential skills.
    *   Frequency Weighting: Keyword frequency within job descriptions is factored into the score.
    *   Configurable Weights:  The contribution of TF-IDF, frequency, and whitelist boost is controlled by weights in `config.yaml`.
5.  **Semantic Keyword Categorization:** Keywords are intelligently categorized:
    *   Category Definitions:  `config.yaml` defines keyword categories (e.g., "Technical Skills", "Soft Skills") and associated terms.
    *   Semantic Similarity: Word vectors from spaCy and cosine similarity are used to measure the semantic similarity between extracted keywords and category terms.
    *   Category Assignment: Keywords are assigned to categories based on the highest semantic similarity score above a configurable `similarity_threshold`.
6.  **Output Generation:** Results are organized and presented in user-friendly Excel reports:
    *   Keyword Summary Sheet:  Provides a ranked list of keywords with aggregated "Total_Score", "Avg_Score", "Job_Count", and assigned "Category".
    *   Detailed Scores Sheet:  A pivot table showing the "Score" for each "Keyword" across each "Job Title", enabling detailed analysis and targeted resume tailoring.

## Getting Started

### Prerequisites

*   Python 3.8+
*   Required libraries (install via pip):
    ```bash
    pip install pandas nltk spacy scikit-learn pyyaml psutil pytest
    ```
*   Download the spaCy English language model:
    ```bash
    python -m spacy download en_core_web_sm
    ```

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/DavidOsipov/Keywords4Cv.git
    cd Keywords4Cv
    ```
2.  (Optional but highly recommended) Create and activate a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```
3.  Install the required libraries (as shown in Prerequisites).

### Usage

1.  **Prepare your job descriptions:** Create a JSON file (e.g., `job_descriptions.json`) containing your job descriptions.  Example:

    ```json
    {
      "Data Scientist": "We are looking for a data scientist with experience in Python, machine learning, and SQL...",
      "Product Manager": "The ideal candidate will have experience with Agile methodologies, product roadmapping, and user research...",
      "Software Engineer": "Seeking a software engineer proficient in Java, Spring, and REST APIs..."
    }
    ```

2.  **Customize `config.yaml`:**  Modify the `config.yaml` file to adjust:
    *   `stop_words`, `stop_words_add`, `stop_words_exclude`: Fine-tune the stop word list for your specific needs.
    *   `skills_whitelist`:  Expand and customize the skills whitelist to include technologies and skills relevant to your target roles and industry.  The more comprehensive this list, the more accurate keyword extraction will be.
    *   `keyword_categories`: Review and customize keyword categories and their associated terms to align with your desired output organization.
    *   `output_dir`:  Specify a directory to save output reports and log files, or leave it to default to the current directory.
    *   `ngram_range`:  Experiment with different n-gram ranges to optimize keyword extraction for your job descriptions. `[1, 3]` (unigrams, bigrams, trigrams) is a good starting point.
    *   `similarity_threshold`:  Adjust the `similarity_threshold` if you want to fine-tune how keywords are assigned to categories based on semantic similarity.
    *   `weighting`:  Customize the `weighting` parameters (`tfidf_weight`, `frequency_weight`, `whitelist_boost`) to adjust the keyword scoring mechanism.
    *   `min_desc_length`, `min_jobs`, `max_desc_length`: Adjust input validation parameters as needed.
    *   Explore other advanced configuration options in `config.yaml` to further tailor the tool to your specific requirements.

3.  **Run the script:**

    ```bash
    python keywords4cv.py -i job_descriptions.json -c config.yaml -o results.xlsx
    ```
   * `-i` or `--input`:  Path to the input JSON file (required).
   * `-c` or `--config`: Path to the configuration file (optional, defaults to `config.yaml`).
   * `-o` or `--output`: Path to the output Excel file (optional, defaults to `results.xlsx`).

4.  **Review the output:** The Excel file (e.g., `results.xlsx`) will be generated in the same directory where you run the script. Examine the "Summary" and "Detailed Scores" sheets to identify top keywords and tailor your resume accordingly.  Check `ats_optimizer.log` for any warnings or errors during processing.

### Running Unit Tests

To ensure the script's functionality and stability, a suite of unit tests is included in the `tests/` directory.  It is highly recommended to run these tests after making any modifications to the code.

1.  **Navigate to the `tests/` directory:**
    ```bash
    cd tests
    ```

2.  **Run the tests using `pytest`:**
    ```bash
    pytest
    ```
    Ensure `pytest` is installed (`pip install pytest`).  `pytest` will automatically discover and run all test files (`test_*.py`) in the `tests/` directory.  Review the test output to confirm all tests pass.


## Repository Structure

*   `keywords4cv.py`: The main Python script containing the core ATS optimization logic.
*   `config.yaml`:  YAML configuration file for customizing stop words, skills whitelist, keyword categories, weighting, and other parameters.
*   `README.md`: This file, providing documentation and usage instructions.
*   `output/`: (Created automatically)  Directory where Excel output reports (`results.xlsx`) and log files (`ats_optimizer.log`) are saved by default.
*   `requirements.txt`:  List of Python package dependencies.  It is recommended to update this file using `pip freeze > requirements.txt` after installing required packages.
*   `tests/`: Directory containing unit tests:
    *   `test_keywords4cv.py`: Python file containing unit tests written using `unittest` and `pytest` frameworks to verify the functionality of `keywords4cv.py`.
*   `job_descriptions.json`: Example input JSON file demonstrating the expected input format for job descriptions.
*   `ats_optimizer.log`: Log file where the script records informational messages, warnings, and errors during execution.

## Contributing

Contributions are welcome!  If you wish to contribute to the project, please follow these steps:

1.  Fork the repository to your GitHub account.
2.  Create a new branch for your feature or bug fix.  Choose a descriptive branch name.
3.  Make your code changes, ensuring they are well-commented and follow coding best practices.
4.  Add or update unit tests in the `tests/` directory to cover your changes and ensure code reliability.
5.  Run the unit tests locally using `pytest` from the `tests/` directory to confirm that all tests pass, including the ones you added.
6.  Commit your changes to your branch with clear and informative commit messages.
7.  Push your branch to your forked repository on GitHub.
8.  Submit a pull request to the main repository, describing your changes and their purpose in detail.

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

**Please be aware that this license is not a standard software license and may not address all software-specific legal concerns.**  Specifically:

*   **No Patent Grant:** This license does *not* grant any patent rights related to the software.
*   **Disclaimer of Warranties:**  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*   **Consult Legal Counsel:** If you have any concerns about the legal implications of using this software under the CC BY-NC-SA 4.0 license, especially regarding patent rights or commercial use, you should consult with legal counsel.

By using, modifying, or distributing this software, you agree to the terms of the CC BY-NC-SA 4.0 license, including the limitations and disclaimers stated above.
