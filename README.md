# Keywords4CV

[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

## Description

Keywords4CV is a Python-based tool designed to help job seekers optimize their resumes and LinkedIn profiles for Applicant Tracking Systems (ATS) and human recruiters. It analyzes a collection of job descriptions and extracts the most important and relevant keywords, enabling users to tailor their application materials to specific job roles. By incorporating these keywords, users can significantly increase their chances of getting noticed by both ATS and recruiters, leading to more interview opportunities.

## State of production

**Not ready!**

## Features

*   **Keyword Extraction:** Identifies key skills, qualifications, and terminology from job descriptions using NLP techniques (spaCy, NLTK, scikit-learn).
*   **TF-IDF Analysis:** Computes Term Frequency-Inverse Document Frequency (TF-IDF) scores, combined with *whitelist boosting* and *keyword frequency*, to determine the importance of keywords across multiple job descriptions.
*   **Synonym Suggestion:** Includes a synonym suggestion feature to help expand the keyword coverage.
*   **Configurable:** Uses a `config.yaml` file for easy customization of:
    *   Stop words (with options to add and exclude specific words).
    *   Skills whitelist.
    *   Keyword categories and their associated terms.
    *   Output directory.
    *   N-gram range for keyword extraction.
    *   Similarity threshold for categorization.
    *   Weighting for TF-IDF, frequency, and whitelist boost.
*   **Output:** Generates an Excel report with:
    *   **Keyword Summary:** A summary of keywords with their combined scores, job counts, and assigned categories.
    *   **Job Specific Details:** A pivot table showing the combined scores of keywords for each individual job title.
*   **Input Validation:** Includes robust input validation to handle various edge cases (e.g., non-string keys, empty descriptions, incorrect file formats).
*   **Command-Line Interface:** Uses `argparse` for a user-friendly command-line interface.
*   **Error Handling:** Includes comprehensive error handling and logging.
*   **Multiprocessing for Analysis:** Leverages multiprocessing to run the core analysis in a separate process, enhancing robustness and enabling timeout functionality.
*   **Caching:** Preprocessing results are cached to minimize redundant computations.
*   **SpaCy optimization:**  spaCy pipeline components that are not needed (parser, ner) are disabled for efficiency.
*   **NLTK resource management**: The script automatically manages required NLTK resources.
*   **Timeout Mechanism:** Implements a configurable timeout to prevent long-running analyses and ensure script stability, especially when processing large job description datasets or encountering unexpected issues.

## How it Works

1.  **Input:** The script takes a JSON file as input via the command line:
    *   **Keys:** Job titles (strings).
    *   **Values:** Full text of the corresponding job descriptions (strings).
2.  **Preprocessing:** The text is cleaned (lowercase, URL/email/special character removal) and tokenized/lemmatized using spaCy and NLTK. A cache is used for efficiency.
3.  **Keyword Extraction:**
    *   **Whitelist Matching:**  Identifies exact phrase matches for skills in the whitelist.
    *   **N-gram Generation:** Generates n-grams (multi-word phrases) based on the configured `ngram_range`.
    *   **Tokenization and Lemmatization:** Processes the text into individual tokens and reduces words to their base form (lemmas).
4.  **Keyword Weighting:**
    *   Calculates TF-IDF scores using scikit-learn's `TfidfVectorizer`.
    *   Applies a *whitelist boost* to keywords found in the whitelist.
    *   Factors in the *frequency* of each keyword.
    *   Combines these factors using configurable weights.
5.  **Keyword Categorization:**
    *   Calculates the semantic similarity between each keyword and user-defined categories using word embeddings and cosine similarity.
    *   Assigns keywords to categories based on a similarity threshold (configurable in `config.yaml`).
6.  **Output Generation:** Creates pandas DataFrames and saves them to an Excel file.

## Getting Started

### Prerequisites

*   Python 3.8+
*   Required libraries (install via pip):
    ```bash
    pip install pandas nltk spacy scikit-learn pyyaml psutil
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
2.  (Optional but recommended) Create and activate a virtual environment:
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
    *   `stop_words`, `stop_words_add`, `stop_words_exclude`: Fine-tune the stop word list.
    *   `skills_whitelist`:  Add skills and technologies relevant to your target roles.
    *   `keyword_categories`: Define categories and associated terms for grouping keywords.
    *   `output_dir`:  Specify the output directory.
    *   `ngram_range`:  Set the desired n-gram range (e.g., `[1, 3]` for unigrams, bigrams, and trigrams).
    *   `similarity_threshold`:  Adjust the threshold for category assignment.
    *   `weighting`: Configure the weights for TF-IDF, frequency, and whitelist boost.
    *   `min_desc_length`: Set minimum description length.
    *    `min_jobs`: Set minimum number of job descriptions.
3.  **Run the script:**

    ```bash
    python keywords4cv.py -i job_descriptions.json -c config.yaml -o results.xlsx
    ```
   * `-i` or `--input`:  Path to the input JSON file (required).
   * `-c` or `--config`: Path to the configuration file (optional, defaults to `config.yaml`).
   * `-o` or `--output`: Path to the output Excel file (optional, defaults to `results.xlsx`).

4.  **Review the output:** The Excel file (e.g., `results.xlsx`) will be saved in the same directory where you run the script.

## Repository Structure

*   `keywords4cv.py`: The main Python script.
*   `config.yaml`:  The configuration file.
*   `README.md`: This file.
*   `output/`: (Created automatically) The directory for storing Excel output and log files.
*   `requirements.txt`: List of required packages.  *(It's recommended to create one using `pip freeze > requirements.txt`)*.
*   `tests/`: Directory for unit tests.
*   `job_descriptions.json`: Example input file.
*   `ats_optimizer.log`: Log file.

## Contributing

Contributions are welcome!  Follow these steps:

1.  Fork the repository.
2.  Create a new branch.
3.  Make your changes and commit them.
4.  Push your branch.
5.  Submit a pull request.

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
