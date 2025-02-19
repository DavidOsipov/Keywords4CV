import re
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import yaml  # For configuration file
import spacy  # For NER and improved lemmatization
from sklearn.feature_extraction.text import TfidfVectorizer  # For TF-IDF

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Load spaCy Model (outside classes for efficiency) ---
try:
    nlp = spacy.load("en_core_web_sm", disable=["parser"])  # Disable parser
except OSError:
    print("Downloading en_core_web_sm model...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=["parser"])


# --- Helper Functions ---
def get_synonyms(word):
    """Gets synonyms of a word from WordNet.  Limits to the same part of speech."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            # Only add synonyms of the same part of speech
            if lemma.synset().pos() == syn.pos():
                synonym = lemma.name().replace("_", " ").lower()
                synonyms.add(synonym)
    return synonyms


def _is_likely_skill(token):
    """Checks if a verb (especially a gerund) or noun is likely describing a skill,
    using dependency parsing for improved context.
    """
    # Check if the token is part of a compound noun (e.g., "data analysis")
    if token.dep_ == "compound":
        return True

    # Check for verbs indicating skills (e.g., "managing", "developing")
    if token.pos_ == "VERB" and token.text.endswith("ing"):
        # Check for dependency relations that suggest a skill context
        if token.dep_ in ("xcomp", "advcl", "ROOT"):  # Open clausal complement, adverbial clause modifier
            return True

    # Check for nouns connected to skill-related verbs via prepositional phrases
    if token.pos_ == "NOUN":
        for child in token.children:
            if child.dep_ == "prep":  # Prepositional modifier
                for prep_child in child.children:
                    # Check if the prepositional phrase is connected to a skill-related verb
                    if prep_child.pos_ == "VERB" and _is_likely_skill(prep_child):  # Recursive check
                        return True
    return False


# --- Classes ---
class TextPreprocessor:
    """Handles text preprocessing."""

    def __init__(self, config):
        self.config = config
        self.stop_words = self._load_stop_words()

    def _load_stop_words(self) -> Set[str]:
        """Loads stop words from the configuration file, with additions and exclusions."""
        base_stop_words = set(self.config['stop_words'])
        additional_stop_words = set(self.config.get('stop_words_add', []))
        exclude_stop_words = set(self.config.get('stop_words_exclude', []))

        stop_words = (base_stop_words | additional_stop_words) - exclude_stop_words
        return stop_words

    def preprocess_text(self, text: str) -> str:
        """Cleans and standardizes text."""
        try:
            text = text.lower()
            text = re.sub(r'[\n\t]+', ' ', text)
            text = re.sub(r'http\S+|www.\S+', '', text)  # Remove URLs
            text = re.sub(r'\S+@\S+', '', text)  # Remove email addresses
            text = re.sub(r'[^\w\s-]', ' ', text)  # Remove punctuation except hyphens
            text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
            return text
        except Exception as e:
            logger.error(f"Error in text preprocessing: {str(e)}")
            raise


class KeywordExtractor:
    """Extracts keywords from preprocessed text."""

    def __init__(self, config):
        self.config = config
        self.skills_whitelist = set(self.config['skills_whitelist'])
        self.preprocessor = TextPreprocessor(self.config)  # Use composition

    def extract_ngrams(self, tokens: List[str], n: int) -> List[str]:
        """Generates n-grams from a list of tokens."""
        return [' '.join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

    def get_keywords(self, text: str) -> Counter:
        """Extracts keywords, including synonyms and named entities."""
        try:
            processed_text = self.preprocessor.preprocess_text(text)
            doc = nlp(processed_text)  # Use spaCy for tokenization and POS tagging
            keywords = []
            keyword_groups = {}  # For grouping synonyms

            for token in doc:
                lemma = token.lemma_.lower()

                # Check for hyphenated words in whitelist
                if '-' in token.text and token.text in self.skills_whitelist:
                    keywords.append(token.text)
                    continue

                # Process nouns, adjectives, proper nouns, and verbs
                if token.pos_ in {"NOUN", "ADJ", "PROPN", "VERB"}:
                    if _is_likely_skill(token) or lemma in self.skills_whitelist or (
                            lemma not in self.preprocessor.stop_words and len(lemma) > 1 and not lemma.isnumeric()):
                        # Handle Synonyms (with POS check)
                        synonyms = get_synonyms(lemma)
                        for synonym in synonyms:
                            if synonym in self.skills_whitelist or (
                                    synonym not in self.preprocessor.stop_words and len(synonym) > 1):
                                if lemma in keyword_groups:
                                    keyword_groups[lemma].add(synonym)
                                else:
                                    keyword_groups[lemma] = {synonym}  # Use set for synonyms
                                keywords.append(synonym)
                        # Always add the original lemma if it meets criteria
                        if lemma in self.skills_whitelist or (
                                lemma not in self.preprocessor.stop_words and len(lemma) > 1):
                            keywords.append(lemma)

            # Noun Phrases (improved handling)
            for chunk in doc.noun_chunks:
                chunk_text = self.preprocessor.preprocess_text(chunk.text.lower())
                # Check if any part of the chunk is in the whitelist or not a stop word
                if any(word in self.skills_whitelist or word not in self.preprocessor.stop_words for word in
                       chunk_text.split()) and len(chunk_text) > 1:
                    keywords.append(chunk_text)

            # NER (simplified and focused)
            doc_original = nlp(text)
            for ent in doc_original.ents:
                if ent.label_ in {"ORG", "PRODUCT"}:  # Focus on ORG and PRODUCT entities
                    entity_text = self.preprocessor.preprocess_text(ent.text.lower())
                    if entity_text and entity_text not in self.preprocessor.stop_words:
                        keywords.append(entity_text)

            # Extract bigrams and trigrams (prioritize whitelisted and non-stop word phrases)
            tokens_text = [t.text for t in doc]
            bigrams = self.extract_ngrams(tokens_text, 2)
            trigrams = self.extract_ngrams(tokens_text, 3)
            phrases = [phrase for phrase in bigrams + trigrams if
                       phrase in self.skills_whitelist or all(
                           word not in self.preprocessor.stop_words for word in phrase.split())]
            keywords.extend(phrases)

            return Counter(keywords)  # Use Counter for efficient counting

        except Exception as e:
            logger.error(f"Error in keyword extraction: {str(e)}")
            raise


class JobDescriptionAnalyzer:
    """Analyzes job descriptions, computes TF-IDF, and generates output."""

    def __init__(self, config):
        self.config = config
        self.extractor = KeywordExtractor(self.config)  # Use composition
        self.keyword_categories = self.config.get('keyword_categories', {})

    def analyze_job_descriptions(self, job_descriptions: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Analyzes job descriptions, computes TF-IDF, and returns summary and pivot DataFrames."""
        try:
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
                if len(description.split()) < 10:  # Check for minimum length
                    logger.warning(f"Job description for '{title}' is very short.  Results may be unreliable.")

            job_texts = list(job_descriptions.values())

            # Use TfidfVectorizer with custom tokenizer and preprocessor
            vectorizer = TfidfVectorizer(
                analyzer='word',
                tokenizer=self.extractor.get_keywords,
                preprocessor=self.extractor.preprocessor.preprocess_text,
                sublinear_tf=True,  # Apply sublinear scaling
                use_idf=True
            )

            tfidf_matrix = vectorizer.fit_transform(job_texts)
            feature_names = vectorizer.get_feature_names_out()

            keyword_data = []
            for i, job_title in enumerate(job_descriptions):
                doc_vector = tfidf_matrix[i].tocoo()
                for j, score in zip(doc_vector.col, doc_vector.data):
                    keyword = feature_names[j]
                    category = "Other"
                    # Categorize keywords (check if keyword is a substring of any category keyword)
                    for cat, keywords in self.keyword_categories.items():
                        if any(keyword in kw for kw in keywords):
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

            # Pivot table: keywords vs job titles, values = TF-IDF
            pivot_df = df.pivot_table(
                index='keyword',
                columns='job_title',
                values='tfidf',
                aggfunc='sum',
                fill_value=0
            )

            # Summary statistics: total TF-IDF, job count, average TF-IDF
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
            logger.error(f"Error in job description analysis: {str(e)}")
            raise


class JobKeywordAnalyzer:
    """Main class to coordinate analysis."""

    def __init__(self, config_file="config.yaml"):
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)

        self.analyzer = JobDescriptionAnalyzer(self.config)  # Use composition

    def analyze_jobs(self, job_descriptions: Dict[str, str]):
        """Analyzes job descriptions and returns results."""
        return self.analyzer.analyze_job_descriptions(job_descriptions)


def main():
    """Main function to perform analysis and save results."""
    try:
        analyzer = JobKeywordAnalyzer()  # Use the configuration file

        # Sample job descriptions (more varied and longer)
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
        logger.error(f"Error during main execution: {str(e)}")
        raise


if __name__ == "__main__":
    # Download required NLTK resources if not already present
    for resource in ['punkt', 'wordnet', 'averaged_perceptron_tagger']:
        try:
            nltk.data.find(
                f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}' if resource == 'wordnet' else f'taggers/{resource}')
        except LookupError:
            nltk.download(resource, quiet=True)
    main()