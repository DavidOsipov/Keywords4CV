import re
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import yaml
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm", disable=["parser"])
except OSError:
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=["parser"])


class TextPreprocessor:
    """Handles text preprocessing including stop words, lemmatization, and tokenization."""

    def __init__(self, config: Dict):
        self.config = config
        self.stop_words = self._load_stop_words()
        self.lemmatizer = WordNetLemmatizer()

    def _load_stop_words(self) -> Set[str]:
        stop_words = set(self.config['stop_words'])
        stop_words.update(self.config.get('stop_words_add', []))
        stop_words.difference_update(self.config.get('stop_words_exclude', []))
        return stop_words

    def preprocess(self, text: str) -> str:
        """Cleans text by removing noise and standardizing format."""
        text = text.lower()
        text = re.sub(r'[\n\t]+', ' ', text)
        text = re.sub(r'http\S+|www\.\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[^\w\s-]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize(self, text: str) -> List[str]:
        """Tokenizes and lemmatizes text, filtering stop words and short tokens."""
        doc = nlp(text)
        tokens = []
        for token in doc:
            if token.text in self.stop_words or len(token.text) <= 1 or token.text.isnumeric():
                continue
            pos = 'v' if token.pos_ == 'VERB' else 'n'
            lemma = self.lemmatizer.lemmatize(token.text, pos=pos)
            tokens.append(lemma)
        return tokens


class KeywordExtractor:
    """Extracts keywords using whitelist, n-grams, entities, and noun chunks."""

    def __init__(self, config: Dict):
        self.config = config
        self.preprocessor = TextPreprocessor(config)
        self.whitelist = self._create_expanded_whitelist(config['skills_whitelist'])
        self.entity_types = set(config.get('allowed_entity_types', []))

    def _create_expanded_whitelist(self, skills: List[str]) -> Set[str]:
        """
        Creates an expanded whitelist including curated synonyms.
        """
        expanded_whitelist = set()
        for skill in skills:
            processed_skill = self.preprocessor.preprocess(skill)
            tokens = self.preprocessor.tokenize(processed_skill)
            expanded_skill = ' '.join(tokens)  # Lemmatized skill
            expanded_whitelist.add(expanded_skill)

            # Get synonym suggestions (using WordNet and POS filtering)
            for token in tokens:
                for synset in wordnet.synsets(token):
                    for lemma in synset.lemmas():
                        synonym = lemma.name().replace("_", " ").lower()
                        if synonym != token:  # Avoid adding the word itself
                            # Add to suggestions, but DO NOT directly add to whitelist
                            # User will review these
                            # This part is handled in the Whitelist Management
                            pass  # Placeholder, suggestions are handled below
        return expanded_whitelist

    def _suggest_synonyms(self, skills: List[str]) -> Dict[str, List[str]]:
        """
        Suggests synonyms for each skill in the whitelist.
        Returns a dictionary where keys are original skills and values are lists of suggested synonyms.
        """

        suggestions = {}
        for skill in skills:
            processed_skill = self.preprocessor.preprocess(skill)
            tokens = self.preprocessor.tokenize(processed_skill)
            skill_synonyms = set()
            for token in tokens:
                for synset in wordnet.synsets(token):
                    for lemma in synset.lemmas():
                        synonym = lemma.name().replace("_", " ").lower()
                        if synonym != token:
                            skill_synonyms.add(synonym)
            suggestions[skill] = list(skill_synonyms)
        return suggestions

    def manage_whitelist(self):
        """Manages the skills whitelist, allowing users to add/remove skills and synonyms."""
        print("Current Whitelist:")
        for i, skill in enumerate(self.config['skills_whitelist']):
            print(f"{i + 1}. {skill}")

        suggestions = self._suggest_synonyms(self.config['skills_whitelist'])

        while True:
            print("\nOptions:")
            print("a. Add a new skill")
            print("b. Add synonyms to an existing skill")
            print("r. Remove a skill")
            print("q. Quit and save changes")
            choice = input("Enter your choice: ").lower()

            if choice == 'a':
                new_skill = input("Enter the new skill: ")
                self.config['skills_whitelist'].append(new_skill)
                suggestions = self._suggest_synonyms(self.config['skills_whitelist'])  # Update
            elif choice == 'b':
                try:
                    skill_index = int(input("Enter the number of the skill to modify: ")) - 1
                    skill = self.config['skills_whitelist'][skill_index]
                    print(f"Suggested synonyms for '{skill}': {', '.join(suggestions[skill])}")
                    synonyms_to_add = input("Enter synonyms to add, separated by commas (or press Enter to skip): ")
                    if synonyms_to_add:
                        for synonym in synonyms_to_add.split(','):
                            synonym = synonym.strip()
                            # Add to config, not only whitelist
                            self.config['skills_whitelist'].append(synonym)
                    suggestions = self._suggest_synonyms(self.config['skills_whitelist'])  # Update

                except (ValueError, IndexError):
                    print("Invalid skill number.")
            elif choice == 'r':
                try:
                    skill_index = int(input("Enter the number of the skill to remove: ")) - 1
                    removed_skill = self.config['skills_whitelist'].pop(skill_index)
                    print(f"Removed skill: {removed_skill}")
                    suggestions = self._suggest_synonyms(self.config['skills_whitelist'])  # Update
                except (ValueError, IndexError):
                    print("Invalid skill number.")

            elif choice == 'q':
                # Save changes to config.yaml
                with open("config.yaml", 'w') as f:
                    yaml.safe_dump(self.config, f)
                print("Whitelist updated and saved.")
                self.whitelist = self._create_expanded_whitelist(self.config['skills_whitelist'])
                break
            else:
                print("Invalid choice.")

    def _preprocess_whitelist(self, skills: List[str]) -> Set[str]:
        """Converts whitelist skills to lemmatized form for accurate matching."""
        processed = set()
        for skill in skills:
            cleaned = self.preprocessor.preprocess(skill)
            tokens = self.preprocessor.tokenize(cleaned)
            processed.add(' '.join(tokens))
        return processed

    def _extract_entities(self, text: str) -> Set[str]:
        """Extracts named entities of allowed types."""
        doc = nlp(text)
        return {ent.text.lower() for ent in doc.ents if ent.label_ in self.entity_types}

    def _extract_ngrams(self, tokens: List[str], n: int) -> Set[str]:
        """Generates n-grams from tokens."""
        return {' '.join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}

    def get_keywords(self, text: str) -> List[str]:
        """Extracts and combines keywords from multiple methods."""
        cleaned_text = self.preprocessor.preprocess(text)
        tokens = self.preprocessor.tokenize(cleaned_text)
        keywords = set()

        # Whitelist skills (exact n-gram matches)
        for skill in self.whitelist:
            skill_len = len(skill.split())
            for ngram in self._extract_ngrams(tokens, skill_len):
                if ngram == skill:
                    keywords.add(ngram)

        # Named entities
        keywords.update(self._extract_entities(cleaned_text))

        # Tokens and n-grams (using configured range)
        for n in range(self.config.get('ngram_range', [1, 1])[0], self.config.get('ngram_range', [1, 1])[1] + 1):  # Use config
            keywords.update(self._extract_ngrams(tokens, n))

        return list(keywords)


class JobAnalyzer:
    """Analyzes job descriptions to compute keyword importance using TF-IDF."""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.keyword_extractor = KeywordExtractor(self.config)
        self.categories = self.config.get('keyword_categories', {})

    def _cosine_similarity(self, vec1, vec2):
        """Calculates cosine similarity between two vectors."""
        return vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def _categorize(self, keyword: str) -> List[str]:
        """
        Assigns a keyword to one or more categories based on semantic similarity.
        """
        categories = []
        keyword_embedding = nlp(keyword).vector  # Get embedding

        for category, terms in self.categories.items():
            # Calculate centroid vector for the category (could be pre-calculated)
            category_embeddings = [nlp(term).vector for term in terms]
            centroid_vector = sum(category_embeddings) / len(category_embeddings)

            similarity = self._cosine_similarity(keyword_embedding, centroid_vector)

            if similarity > self.config.get('similarity_threshold', 0.7):  # Default threshold
                categories.append(category)

        if not categories:
            categories.append('Other')
        return categories

    def analyze(self, job_descriptions: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Main analysis method generating TF-IDF scores and categorization."""
        self._validate_input(job_descriptions)

        # --- Calculate Keyword Frequencies (BEFORE TF-IDF) ---
        all_keywords = {}
        for title, desc in job_descriptions.items():
            keywords = self.keyword_extractor.get_keywords(desc)
            for keyword in keywords:
                all_keywords[keyword] = all_keywords.get(keyword, 0) + 1

        # --- Prepare keyword strings for TF-IDF ---
        job_keywords = {
            title: ' '.join(self.keyword_extractor.get_keywords(desc))
            for title, desc in job_descriptions.items()
        }

        # --- Compute TF-IDF ---
        vectorizer = TfidfVectorizer(
            tokenizer=lambda x: x.split(),  # Treat space-separated n-grams as tokens
            lowercase=False,  # Already handled during preprocessing
            ngram_range=(1, 1)  # IMPORTANT:  Do NOT use the configurable ngram_range here.
        )
        tfidf_matrix = vectorizer.fit_transform(job_keywords.values())
        feature_names = vectorizer.get_feature_names_out()

        # --- Generate Results (with Adjusted Weights) ---
        results = []
        for i, (title, _) in enumerate(job_descriptions.items()):
            for col in tfidf_matrix[i].nonzero()[1]:
                keyword = feature_names[col]
                tfidf_score = tfidf_matrix[i, col]

                # --- Calculate Adjusted Weight ---
                whitelist_boost = 1.5 if keyword in self.keyword_extractor.whitelist else 1.0  # Example boost
                log_frequency = 1 + np.log(all_keywords.get(keyword, 1))  # Log frequency, handle 0
                adjusted_tfidf = tfidf_score * whitelist_boost * log_frequency

                results.append({
                    'Keyword': keyword,
                    'Job Title': title,
                    'TF-IDF': adjusted_tfidf,  # Use adjusted TF-IDF
                    'Category': ','.join(self._categorize(keyword))  # Join categories with commas
                })

        # --- (Rest of the method remains the same - DataFrame creation, etc.) ---
        df = pd.DataFrame(results)
        summary = df.groupby('Keyword').agg(
            Total_TFIDF=('TF-IDF', 'sum'),
            Job_Count=('Job Title', 'nunique'),
            Avg_TFIDF=('TF-IDF', 'mean')
        ).sort_values('Total_TFIDF', ascending=False).reset_index()

        pivot = df.pivot_table(index='Keyword', columns='Job Title', values='TF-IDF', aggfunc='sum', fill_value=0)
        return summary, pivot

    def _validate_input(self, jobs: Dict[str, str]):
        """Validates job description input format and content."""
        if not isinstance(jobs, dict):
            raise TypeError("Input must be a dictionary.")
        if len(jobs) < 2:
            raise ValueError("At least 2 job descriptions required.")
        for title, desc in jobs.items():
            if not isinstance(title, str) or not title.strip():
                raise ValueError(f"Invalid job title: {title}")
            if not isinstance(desc, str) or len(desc.split()) < 10:
                raise ValueError(f"Invalid description for: {title}")


# Example usage
if __name__ == "__main__":
    print("Choose an option:")
    print("1. Analyze Job Descriptions")
    print("2. Manage Whitelist")
    choice = input("Enter your choice (1 or 2): ")

    if choice == '1':
        analyzer = JobAnalyzer()
        sample_jobs = {
            "Data Scientist": "Expertise in Python, machine learning, and statistical modeling...",
            "Product Manager": "Experience in Agile, Scrum, and product lifecycle management..."
        }

        summary_df, pivot_df = analyzer.analyze(sample_jobs)

        # Save results
        output_dir = Path(analyzer.config['output_dir'])
        output_dir.mkdir(exist_ok=True)
        with pd.ExcelWriter(output_dir / "analysis_results.xlsx") as writer:
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
            pivot_df.to_excel(writer, sheet_name="Details")

        print("Top Keywords:")
        print(summary_df.head(10))
    elif choice == '2':
        analyzer = JobAnalyzer()  # Or just the KeywordExtractor
        analyzer.keyword_extractor.manage_whitelist()  # New method
    else:
        print("Invalid choice.")