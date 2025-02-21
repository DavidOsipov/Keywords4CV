# Keywords4CV

[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

## Description

Keywords4CV is a Python-based tool designed to help job seekers optimize their resumes and LinkedIn profiles for Applicant Tracking Systems (ATS) and human recruiters. It analyzes a collection of job descriptions and extracts the most important and relevant keywords, enabling users to tailor their application materials to specific job roles. By incorporating these keywords, users can significantly increase their chances of getting noticed by both ATS and recruiters, leading to more interview opportunities.

## State of Production

**Not ready! Under active development.**  
Currently at version `0.09 (Alpha)`. While significant improvements have been made, critical issues persist (see Known Issues below), and the tool is not yet production-ready.

## Features

*   **Enhanced Keyword Extraction:** Identifies key skills, qualifications, and terminology from job descriptions using advanced NLP techniques (spaCy, NLTK, scikit-learn). Now preserves multi-word skills (e.g., "machine learning") through entity recognition.
*   **TF-IDF Analysis with Advanced Weighting:** Computes Term Frequency-Inverse Document Frequency (TF-IDF) scores, combined with *whitelist boosting* and *keyword frequency*. Configurable weights allow for fine-tuning keyword importance.
*   **Synonym Expansion:** Leverages WordNet to suggest synonyms, expanding keyword coverage and improving semantic matching.
*   **Highly Configurable:** Uses a `config.yaml` file for extensive customization:
    *   Stop words (with options to add and exclude specific words). More comprehensive default stop word list.
    *   Skills whitelist: Extensive and customizable whitelist of technical and soft skills, now integrated into spaCy entity ruler for better preservation.
    *   Keyword categories and semantic categorization: Group keywords into meaningful categories (e.g., "Technical Skills", "Soft Skills") with improved semantic accuracy via word vectors.
    *   N-gram range: Adjusted to `[1, 2]` in `0.09` for concise, actionable phrases (configurable).
    *   Similarity threshold: Increased to `0.65` in `0.09` for stricter semantic categorization (configurable).
    *   Weighting for TF-IDF, frequency, and whitelist boost: Customize the scoring mechanism.
    *   Description length limits and job count thresholds: Configure input validation rules.
    *   Semantic validation: Enabled by default in `0.09` to filter keywords by context.
*   **Detailed Output Reports:** Generates comprehensive Excel reports:
    *   **Keyword Summary:** Aggregated keyword scores, job counts, average scores, and assigned categories for a high-level overview.
    *   **Job Specific Details:** Detailed table (formerly pivot) showing keyword scores, TF-IDF, frequency, category, and whitelist status per job title.
*   **Robust Input Validation:** Rigorous validation of job descriptions, handling empty titles, descriptions, incorrect data types, and encoding issues. Clear error messages and logging.
*   **User-Friendly Command-Line Interface:** `argparse` provides a clear and easy-to-use interface.
*   **Comprehensive Error Handling and Logging:** Detailed logging to `ats_optimizer.log` with improved error handling for configuration, input, and memory issues.
*   **Multiprocessing for Robust Analysis:** Core analysis uses multiprocessing for stability and timeout control.
*   **Efficient Caching:** Uses `functools.lru_cache` and `OrderedDict` to optimize performance, with cache invalidation on config changes.
*   **SpaCy Pipeline Optimization:** Disables unnecessary components (`parser`, `ner`) and adds `sentencizer` for efficiency and consistency.
*   **Automatic NLTK Resource Management:** Ensures WordNet and other resources are downloaded if missing.
*   **Memory Management and Chunking:** Adaptive chunking and memory monitoring prevent memory errors for large datasets.
*   **Extensive Unit Testing:** Includes a test suite in `tests/`, though currently unreliable (see Known Issues).

## How it Works

1.  **Input:** Accepts a JSON file (e.g., `job_descriptions.json`) with job titles as keys and descriptions as values.
2.  **Preprocessing:** Cleans text (lowercasing, URL/email removal), tokenizes, lemmatizes, and caches results.
3.  **Keyword Extraction:** 
    *   Matches `skills_whitelist` phrases as `SKILL` entities using spaCy.
    *   Generates n-grams (default `[1, 2]` in `0.09`) and filters out noise (single-letter tokens, stop words).
    *   Expands with WordNet synonyms.
4.  **Keyword Weighting and Scoring:** Combines TF-IDF, frequency, and whitelist boost (configurable weights).
5.  **Semantic Keyword Categorization:** Assigns categories using substring matching and semantic similarity (threshold `0.65` in `0.09`).
6.  **Output Generation:** Produces Excel reports:
    *   **Summary:** Ranked keywords with `Total_Score`, `Avg_Score`, `Job_Count`.
    *   **Detailed Scores:** Per-job details including `Score`, `TF-IDF`, `Frequency`, `Category`, `In Whitelist`.

## Getting Started

### Prerequisites

*   Python 3.8+
*   Required libraries:
    ```bash
    pip install pandas nltk spacy scikit-learn pyyaml psutil pytest
    ```
*   SpaCy model:
    ```bash
    python -m spacy download en_core_web_sm
    ```

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
2.  **Customize `config.yaml`:** Adjust `skills_whitelist`, `ngram_range`, `similarity_threshold`, etc.
3.  **Run the Script:**
    ```bash
    python keywords4cv.py -i job_descriptions.json -c config.yaml -o results.xlsx
    ```
4.  **Review Output:** Check `results.xlsx` and `ats_optimizer.log`.

### Running Unit Tests

```bash
cd tests
pytest
```
**Note:** Tests are currently unreliable (see Known Issues).

## Known Issues (Version 0.09)
- **Incorrect Keyword Display:** Summary shows single-word keywords with uniform scores (e.g., `1.42192903`), missing multi-word phrases from `skills_whitelist`.
- **Unreliable Unit Tests:** Test suite fails consistently and lacks coverage.
- **Whitelist Inconsistency:** Many whitelisted terms marked `FALSE` in output.
- **Low TF-IDF Variance:** Scores lack differentiation, possibly due to scoring bugs.

## Repository Structure

*   `keywords4cv.py`: Main script.
*   `config.yaml`: Configuration file.
*   `README.md`: This documentation.
*   `output/`: Auto-generated for `results.xlsx` and `ats_optimizer.log`.
*   `requirements.txt`: Dependency list (update with `pip freeze > requirements.txt`).
*   `tests/`: Unit tests (currently unreliable).
*   `job_descriptions.json`: Sample input.

## Contributing

1. Fork, create a branch, and make changes.
2. Update/add tests in `tests/`.
3. Run `pytest` (despite issues) and commit.
4. Submit a pull request with detailed description.

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
