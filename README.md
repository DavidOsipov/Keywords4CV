# Keywords4CV

[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

## Description

Keywords4CV is a Python-based tool designed to help job seekers optimize their resumes and LinkedIn profiles for Applicant Tracking Systems (ATS) and human recruiters. It analyzes a collection of job descriptions and extracts the most important and relevant keywords, enabling users to tailor their application materials to specific job roles. By incorporating these keywords, users can significantly increase their chances of getting noticed by both ATS and recruiters, leading to more interview opportunities.

## Features

*   **Keyword Extraction:** Identifies key skills, qualifications, terminology, and named entities from job descriptions using NLP techniques (spaCy, NLTK, scikit-learn).
*   **TF-IDF Analysis:** Computes Term Frequency-Inverse Document Frequency (TF-IDF) scores, combined with *whitelist boosting* and *overall keyword frequency*, to determine the importance of keywords across multiple job descriptions.
*   **Curated Synonym Handling:**  Includes a *curated* synonym handling feature. The script allows users to manage a whitelist of skills and their synonyms via a "Whitelist Management Mode". This helps capture relevant keyword variations.
*   **Named Entity Recognition (NER):** Extracts relevant named entities (organizations, products, etc.) using spaCy to capture important industry-specific terms.
*   **Multi-Label Keyword Categorization:**  Categorizes extracted keywords into user-defined categories (e.g., "Technical Skills," "Soft Skills," "Industry Knowledge") using *semantic similarity* (word embeddings and cosine similarity). Keywords can belong to *multiple* categories.
*   **Configurable:** Uses a `config.yaml` file for easy customization of:
    *   Stop words (with options to add and exclude specific words).
    *   Skills whitelist (with synonym management).
    *   Allowed entity types for NER.
    *   Keyword categories and their associated terms.
    *   Output directory.
    *   N-gram range for keyword extraction.
    *   Similarity threshold for categorization.
*   **Output:** Generates two Excel reports:
    *   **Keyword Summary:** A summary of keywords with their *adjusted* TF-IDF scores (incorporating whitelist boost and frequency), job counts, and assigned categories.
    *   **Job Specific Details:** A pivot table showing the adjusted TF-IDF scores of keywords for each individual job title.
*   **Input Validation:** Includes robust input validation to handle various edge cases (e.g., non-string keys, empty descriptions).
*   **Whitelist Management Mode:** Provides a user-friendly way to manage the skills whitelist directly from the script, allowing users to add/remove skills and add/review curated synonyms.

## How it Works

1.  **Input:** The script takes a Python dictionary as input:
    *   **Keys:** Job titles (strings).
    *   **Values:** Full text of the corresponding job descriptions (strings).
2.  **Preprocessing:** The text is cleaned (lowercase, punctuation removal) and tokenized/lemmatized using spaCy and NLTK.
3.  **Keyword Extraction:**
    *   **Whitelist Matching:**  Identifies exact phrase matches for skills (and their curated synonyms) in the expanded whitelist.
    *   **N-gram Generation:** Generates n-grams (multi-word phrases) based on the configured `ngram_range`.
    *   **Named Entity Recognition:** Extracts relevant named entities.
    *   **Tokenization and Lemmatization:** Processes the text into individual tokens and reduces words to their base form (lemmas).
4.  **Keyword Weighting:**
    *   Calculates TF-IDF scores using scikit-learn's `TfidfVectorizer`.
    *   Applies a *whitelist boost* to keywords found in the (expanded) whitelist.
    *   Factors in the *overall frequency* of each keyword across all job descriptions.
    *   Combines these factors to produce an *adjusted* TF-IDF score.
5.  **Keyword Categorization:**
    *   Calculates the semantic similarity between each keyword and user-defined categories using word embeddings.
    *   Assigns keywords to categories based on a similarity threshold (configurable in `config.yaml`).
    *   Allows for *multi-label categorization* (a keyword can belong to multiple categories).
6.  **Output Generation:** Creates pandas DataFrames and saves them to an Excel file.

## Getting Started

### Prerequisites

*   Python 3.7+
*   Required libraries (install via pip):
    ```bash
    pip install pandas nltk spacy scikit-learn pyyaml openpyxl numpy
    ```
*   Download the spaCy English language model:
    ```bash
    python -m spacy download en_core_web_sm
    ```
*   NLTK resources: The script will automatically download necessary NLTK resources (punkt, wordnet, averaged_perceptron_tagger) on its first run.

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/Keywords4Cv.git  # Replace your-username
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

1.  **Prepare your job descriptions:** Create a Python dictionary containing your job descriptions.  Example:

    ```python
    job_descriptions = {
        "Data Scientist": "People deserve more from their money... (full job description)",
        "Product Manager - SaaS": "We are seeking a hands-on Product Manager... (full job description)",
        # ... more job descriptions ...
    }
    ```

2.  **Customize `config.yaml`:**  Modify the `config.yaml` file to adjust:
    *   `stop_words`, `stop_words_add`, `stop_words_exclude`: Fine-tune the stop word list.
    *   `skills_whitelist`:  Add initial skills and technologies relevant to your target roles.  *This is crucial for good results.* You'll manage synonyms via the script's "Whitelist Management Mode."
    *   `allowed_entity_types`: Select which named entity types to extract.
    *   `keyword_categories`: Define categories and associated terms for grouping keywords.
    *   `output_dir`:  Specify the output directory.
    *   `ngram_range`:  Set the desired n-gram range (e.g., `[1, 3]` for unigrams, bigrams, and trigrams).
    *   `similarity_threshold`:  Adjust the threshold for category assignment.

3.  **Run the script:** Execute the `main.py` script.  The script will present a menu:

    ```
    Choose an option:
    1. Analyze Job Descriptions
    2. Manage Whitelist
    Enter your choice (1 or 2):
    ```

    *   **Option 1 (Analyze Job Descriptions):**  Performs the keyword analysis and saves the results to an Excel file.
    *   **Option 2 (Manage Whitelist):**  Enters the "Whitelist Management Mode," allowing you to add/remove skills, add suggested synonyms, and save the changes to `config.yaml`.  It is recommended to run this *before* analyzing job descriptions to customize the whitelist.

4.  **Review the output:** The Excel file (e.g., `analysis_results.xlsx`) will be saved in the specified `output_dir`.

## Repository Structure

*   `main.py`: The main Python script.
*   `config.yaml`:  The configuration file.
*   `README.md`: This file.
*   `output/`: (Created automatically) The directory for storing Excel output.
*   `requirements.txt` (Optional, but highly recommended): List of required packages.  Create with `pip freeze > requirements.txt`.

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
