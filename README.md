# Keywords4Cv

[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

## Description

Keywords4Cv is a Python-based tool designed to help job seekers optimize their resumes and LinkedIn profiles for Applicant Tracking Systems (ATS) and human recruiters.  It analyzes a collection of job descriptions and extracts the most important and relevant keywords, enabling users to tailor their application materials to specific job roles.  By incorporating these keywords, users can significantly increase their chances of getting noticed by both ATS and recruiters, leading to more interview opportunities.

## Features

*   **Keyword Extraction:** Identifies key skills, qualifications, and terminology from job descriptions using NLP techniques (spaCy, NLTK, scikit-learn).
*   **TF-IDF Analysis:** Computes Term Frequency-Inverse Document Frequency (TF-IDF) scores to determine the importance of keywords across multiple job descriptions.
*   **Synonym Handling:** Includes synonym recognition (using WordNet) to broaden keyword coverage.
*   **NER (Named Entity Recognition):** Extracts relevant named entities (organizations, products) to capture important industry-specific terms.
*   **Configurable:** Uses a `config.yaml` file for easy customization of stop words, skills whitelist, keyword categories, and output directory.
*   **Output:** Generates two Excel reports:
    *   **Keyword Summary:**  A summary of keywords with their total TF-IDF scores, job counts, and average TF-IDF scores.
    *   **Job Specific Details:** A pivot table showing the TF-IDF scores of keywords for each individual job title.
* **Input Validation:** Includes robust input validation to handle various edge cases.
* **Dependency Parsing:** Utilizes spaCy for dependency parsing, which is used to extract the skills more accurately.

## How it Works

1.  **Input:** The script takes a Python dictionary as input, where:
    *   **Keys:** Job titles (strings).
    *   **Values:** Full text of the corresponding job descriptions (strings).
2.  **Preprocessing:** The text is cleaned and standardized (lowercase, punctuation removal, etc.).
3.  **Keyword Extraction:** The script uses spaCy and NLTK to:
    *   Tokenize the text.
    *   Identify parts of speech (nouns, verbs, adjectives, etc.).
    *   Lemmatize words (reduce to their base form).
    *   Generate n-grams (multi-word phrases).
    *   Extract named entities.
    *   Identify likely skills using dependency parsing.
4.  **TF-IDF Calculation:**  scikit-learn's `TfidfVectorizer` is used to calculate TF-IDF scores for each keyword.
5.  **Output Generation:**  Pandas DataFrames are created and saved to an Excel file.

## Getting Started

### Prerequisites

*   Python 3.7+
*   Required libraries (install via pip):
    ```bash
    pip install pandas nltk spacy scikit-learn pyyaml openpyxl
    ```
*   Download the spaCy English language model:
    ```bash
    python -m spacy download en_core_web_sm
    ```
* NLTK resources: The script will attempt to download necessary NLTK resources (punkt, wordnet, averaged_perceptron_tagger) if they are not already present.

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
        "Data Science Product Owner": "People deserve more from their money... (full job description)",
        "Product Manager - SaaS": "We are seeking a hands-on Product Manager... (full job description)",
        # ... more job descriptions ...
    }
    ```
2.  **Customize `config.yaml` (Optional):**  Modify the `config.yaml` file to adjust:
    *   `stop_words`:  Add or remove common words to be ignored.
    *   `skills_whitelist`:  Add or remove specific skills and technologies relevant to your target roles.  This is *crucial* for good results.
    *   `stop_words_add`, `stop_words_exclude`: Fine-tune the stop word list.
    *   `output_dir`:  Specify the directory where the Excel output will be saved.
    *   `keyword_categories`: Define categories for organizing keywords.

3.  **Run the script:** Execute the `main.py` script:
    ```bash
    python main.py
    ```
4.  **Review the output:** The results will be saved in an Excel file (e.g., `job_keywords_analysis.xlsx`) in the specified `output_dir`.

## Repository Structure

*   `main.py`: The main Python script containing the keyword extraction and analysis logic.
*   `config.yaml`:  The configuration file for customizing the script's behavior.
*   `README.md`: This file, providing information about the project.
*   `output/`: (Created automatically) The default directory for storing the Excel output.
*    `requirements.txt` (Optional, but highly recommended): A text file listing the required packages, which you could generate with `pip freeze > requirements.txt`.

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please follow these steps:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes and commit them with clear, descriptive messages.
4.  Push your branch to your forked repository.
5.  Submit a pull request to the main repository.

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
