import re
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
import yaml
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Download NLTK Resources ---
def download_nltk_resources():
    """Downloads necessary NLTK resources."""
    for resource in ['punkt', 'wordnet', 'averaged_perceptron_tagger']:
        try:
            nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}' if resource == 'wordnet' else f'taggers/{resource}')
        except LookupError:
            logging.info(f"Downloading NLTK resource: {resource}")
            nltk.download(resource, quiet=True)

download_nltk_resources()  # Call the download function at the start

# --- Load spaCy Model ---
try:
    nlp = spacy.load("en_core_web_sm", disable=["parser"])  # Disable parser for speed
except OSError:
    logger.warning("Downloading en_core_web_sm model...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=["parser"])


# --- Helper Functions ---
def get_synonyms(word: str, pos: str = None) -> Set[str]:
    """Gets synonyms of a word from WordNet, optionally filtering by POS."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        if pos is None or syn.pos() == pos:  # Optional POS filter
            for lemma in syn.lemmas():
                synonym = lemma.name().replace("_", " ").lower()
                synonyms.add(synonym)
    return synonyms

def _is_likely_skill(token: spacy.tokens.Token) -> bool:
    """Checks if a token is likely part of a skill phrase (simplified)."""
    if token.dep_ == "compound":  # Compound nouns (e.g., "data analysis")
        return True
    if token.pos_ == "VERB" and token.text.endswith("ing"): # Check verbs
        return True
    if token.pos_ == "NOUN": # Check nouns
        return True

    return False


# --- Classes ---
class TextPreprocessor:
    """Handles text preprocessing."""

    def __init__(self, config: Dict):
        """Initializes the TextPreprocessor with configuration."""
        self.config = config
        self.stop_words = self._load_stop_words()
        self.lemmatizer = WordNetLemmatizer() # Use WordNetLemmatizer

    def _load_stop_words(self) -> Set[str]:
        """Loads stop words from the configuration."""
        stop_words = set(self.config['stop_words'])
        stop_words.update(self.config.get('stop_words_add', []))
        stop_words.difference_update(self.config.get('stop_words_exclude', []))
        return stop_words

    def preprocess_text(self, text: str) -> str:
        """Cleans and standardizes text."""
        try:
            text = text.lower()
            text = re.sub(r'[\n\t]+', ' ', text)          # Remove newlines and tabs
            text = re.sub(r'http\S+|www.\S+', '', text)   # Remove URLs
            text = re.sub(r'\S+@\S+', '', text)           # Remove email addresses
            text = re.sub(r'[^\w\s-]', ' ', text)          # Remove punctuation except hyphens
            text = re.sub(r'\s+', ' ', text).strip()      # Remove extra whitespace
            return text
        except Exception as e:
            logger.error(f"Error in text preprocessing: {e}")
            raise

    def tokenize_and_lemmatize(self, text: str) -> List[str]:
        """Tokenizes, lemmatizes, and filters text."""
        doc = nlp(text)
        tokens = []
        for token in doc:
            if token.text in self.stop_words or len(token.text) <= 1 or token.text.isnumeric():
                continue # Skip stop words, short words, and numbers

            if token.pos_ in {"NOUN", "ADJ", "PROPN", "VERB"}:
                lemma = self.lemmatizer.lemmatize(token.text, pos='v' if token.pos_ == 'VERB' else 'n')
                tokens.append(lemma)
        return tokens



class KeywordExtractor:
    """Extracts keywords from preprocessed text."""

    def __init__(self, config: Dict):
        """Initializes the KeywordExtractor with configuration."""
        self.config = config
        self.skills_whitelist = set(self.config['skills_whitelist'])
        self.preprocessor = TextPreprocessor(self.config)
        self.allowed_entity_types = set(self.config.get('allowed_entity_types', ["ORG", "PRODUCT"])) # Configurable entity types

    def extract_ngrams(self, tokens: List[str], n: int) -> List[str]:
        """Generates n-grams from a list of tokens."""
        return [' '.join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

    def get_keywords(self, text: str) -> Counter:
        """Extracts keywords, including synonyms and named entities."""
        try:
            processed_text = self.preprocessor.preprocess_text(text)
            tokens = self.preprocessor.tokenize_and_lemmatize(processed_text)
            keywords = Counter()

            # Add whitelisted skills directly
            for skill in self.skills_whitelist:
                if skill in processed_text:
                    keywords[skill] += 1

            # Add lemmatized tokens
            keywords.update(tokens)

            # Noun chunks (simplified)
            doc = nlp(processed_text)  # Process with spaCy *after* preprocessing
            for chunk in doc.noun_chunks:
              chunk_text = chunk.text.lower()
              if any(word in self.skills_whitelist or word not in self.preprocessor.stop_words for word in chunk_text.split()):
                keywords[chunk_text] += 1


            # NER (using allowed entity types)
            for ent in doc.ents:
                if ent.label_ in self.allowed_entity_types:
                    entity_text = self.preprocessor.preprocess_text(ent.text.lower())
                    if entity_text: # and entity_text not in self.preprocessor.stop_words:  # Optional: re-check stop words
                        keywords[entity_text] += 1

            # Bigrams and Trigrams (filtered)
            tokens_text = [t for t in tokens]
            for n in [2, 3]:
                for ngram in self.extract_ngrams(tokens_text, n):
                    if any(word in self.skills_whitelist for word in ngram.split()) or all(word not in self.preprocessor.stop_words for word in ngram.split()):
                        keywords[ngram] += 1
            return keywords

        except Exception as e:
            logger.error(f"Error in keyword extraction: {e}")
            raise


class JobDescriptionAnalyzer:
    """Analyzes job descriptions, computes TF-IDF, and generates output."""

    def __init__(self, config: Dict):
        """Initializes the JobDescriptionAnalyzer with configuration."""
        self.config = config
        self.extractor = KeywordExtractor(self.config)
        self.keyword_categories = self.config.get('keyword_categories', {})

    def analyze_job_descriptions(self, job_descriptions: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Analyzes job descriptions and returns summary and pivot DataFrames."""
        # Input Validation
        if not isinstance(job_descriptions, dict):
            raise TypeError("job_descriptions must be a dictionary.")
        if not (2 <= len(job_descriptions) <= 100):
            raise ValueError("job_descriptions must contain between 2 and 100 entries.")
        for title, description in job_descriptions.items():
            if not isinstance(title, str):
                raise TypeError(f"Job title '{title}' must be a string.")
            if not isinstance(description, str):
                raise TypeError(f"Job description for '{title}' must be a string.")
            if not description.strip():
                raise ValueError(f"Job description for '{title}' is empty.")
            if len(description.split()) < 10:
                logger.warning(f"Job description for '{title}' is very short. Results may be unreliable.")

        job_texts = list(job_descriptions.values())

        # TF-IDF Vectorization (using custom tokenizer)
        vectorizer = TfidfVectorizer(
            analyzer='word',
            tokenizer=self.extractor.get_keywords,
            preprocessor=None, # Preprocessing handled within get_keywords
            ngram_range=(1, 3), # Consider unigrams, bigrams, and trigrams directly
            sublinear_tf=True,
            use_idf=True
        )


        try:

            tfidf_matrix = vectorizer.fit_transform(job_texts)
            feature_names = vectorizer.get_feature_names_out()

            keyword_data = []
            for i, job_title in enumerate(job_descriptions):
                doc_vector = tfidf_matrix[i].tocoo()
                for j, score in zip(doc_vector.col, doc_vector.data):
                    keyword = feature_names[j]
                    category = "Other"
                    # Categorize keywords (using 'in' for substring matching)
                    for cat, keywords_list in self.keyword_categories.items():
                        if any(kw in keyword for kw in keywords_list): # Check if extracted keyword contains category keyword
                            category = cat
                            break

                    keyword_data.append({
                        'keyword': keyword,
                        'job_title': job_title,
                        'tfidf': score,
                        'category': category
                    })

            df = pd.DataFrame(keyword_data)

            if df.empty:
                logger.warning("No keywords were extracted.")
                return pd.DataFrame(), pd.DataFrame()

            # Pivot table: keywords vs job titles
            pivot_df = df.pivot_table(index='keyword', columns='job_title', values='tfidf', aggfunc='sum', fill_value=0)

            # Summary statistics
            summary_stats = df.groupby('keyword').agg(
                total_tfidf=('tfidf', 'sum'),
                job_count=('job_title', 'nunique'),
                avg_tfidf=('tfidf', 'mean')
            ).reset_index()

            summary_stats = summary_stats.sort_values('total_tfidf', ascending=False)
            summary_stats.rename(columns={
                'total_tfidf': 'Total TF-IDF',
                'job_count': 'Job Count',
                'avg_tfidf': 'Average TF-IDF'
            }, inplace=True)

            return summary_stats, pivot_df

        except Exception as e:
            logger.error(f"Error in job description analysis: {e}")
            raise
    # ... (rest of the JobDescriptionAnalyzer class)

class JobKeywordAnalyzer:
    """Main class to coordinate the analysis."""

    def __init__(self, config_file: str = "config.yaml"):
        """Initializes the JobKeywordAnalyzer with the configuration file."""
        try:
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_file}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML in {config_file}: {e}")
            raise

        self.analyzer = JobDescriptionAnalyzer(self.config)

    def analyze_jobs(self, job_descriptions: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Analyzes job descriptions and returns the results."""
        return self.analyzer.analyze_job_descriptions(job_descriptions)



def main():
    """Main function to perform analysis and save results."""

    try:
        analyzer = JobKeywordAnalyzer()

        job_descriptions = {
            "Data Science Product Owner": (
                "People deserve more from their money.  We are looking for a Data Science Product Owner "
                "to join our team and lead the development of innovative data-driven products.  "
                "Responsibilities include defining product vision, prioritizing features, and working "
                "closely with data scientists and engineers.  The ideal candidate will have strong "
                "experience in product management, a deep understanding of data science techniques (like "
                "machine learning and statistical modeling), and excellent communication skills. "
                "Proficiency in Python, SQL, and data visualization tools (Tableau, Power BI) is required. "
                "Experience with Agile development methodologies is a plus. You will be managing data products."
            ),
            "Product Manager - SaaS": (
                "We are seeking a hands-on Product Manager to drive the strategy and execution of our "
                "SaaS platform.  This role requires a proven track record of successfully launching and "
                "scaling SaaS products.  Key responsibilities include conducting market research, gathering "
                "customer feedback, defining product requirements, and collaborating with engineering, "
                "design, and marketing teams.  Strong analytical and problem-solving skills are essential.  "
                "Experience with Agile methodologies (Scrum, Kanban) and product management tools (Jira, "
                "Confluence) is required.  Excellent communication, negotiation, and leadership skills are a must. "
                "Experience with data analysis is beneficial."
            ),
            "Data Analyst": (
                "Join our growing data analytics team! We need a Data Analyst proficient in SQL, Excel, and Python.  "
                "You will be responsible for analyzing large datasets, identifying trends, and creating reports "
                "to support business decisions. Experience with data visualization and machine learning is a plus. "
                "Strong communication and collaboration skills are important.  Experience managing databases is desirable."
            ),
            "Software Engineer (Backend)": (
                "We're hiring a talented Software Engineer to develop and maintain our backend systems.  "
                "This role requires strong proficiency in Java and experience with cloud platforms (AWS, Azure, or GCP). "
                "Experience with database technologies (SQL, NoSQL) and API development (REST, GraphQL) is essential.  "
                "Familiarity with DevOps practices (CI/CD, Docker, Kubernetes) is a plus. You must be able to deliver results."
            ),
            "Frontend Developer": (
                "We are looking for a skilled Frontend Developer to build user-friendly interfaces for our web applications.  "
                "The ideal candidate will have strong experience with JavaScript, React, and HTML/CSS.  "
                "Experience with testing frameworks and responsive design principles is required.  "
                "Good communication and teamwork skills are essential.  Experience with Node.js is a plus."
            )
        }
        summary_df, pivot_df = analyzer.analyze_jobs(job_descriptions)

        if summary_df.empty:
            print("No keywords were extracted.")
            return

        # Create output directory
        output_dir = Path(analyzer.config['output_dir'])
        output_dir.mkdir(exist_ok=True)

        # Save results to Excel
        output_file = output_dir / "job_keywords_analysis.xlsx"
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            summary_df.to_excel(writer, sheet_name='Keyword Summary', index=False)
            pivot_df.to_excel(writer, sheet_name='Job Specific Details')

        print("\nAnalysis complete! Results saved to:", output_file)
        print("\nTop Keywords (Sorted by Total TF-IDF):")
        print(summary_df[['keyword', 'Total TF-IDF']].head(20))


    except Exception as e:
        logger.exception("An error occurred during main execution:") # Use exception for full traceback



if __name__ == "__main__":
    main()