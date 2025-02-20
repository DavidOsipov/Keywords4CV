import unittest
from unittest.mock import Mock, patch, MagicMock, mock_open
import pytest
import sys
import json
import yaml
import pandas as pd
import numpy as np
from keywords4cv import (
    EnhancedTextPreprocessor, AdvancedKeywordExtractor, ATSOptimizer,
    ConfigError, InputValidationError, ValidationResult,
    cosine_similarity, ensure_nltk_resources, load_job_data,
    parse_arguments, run_analysis, save_results, initialize_analyzer
)

# --- Fixtures using pytest ---
@pytest.fixture
def sample_config():
    """Fixture for a sample configuration matching the script's defaults."""
    return {
        "stop_words": ["the", "a"],
        "cache_size": 2,
        "ngram_range": [1, 3],
        "whitelist_ngram_range": [1, 3],
        "skills_whitelist": ["python", "coding"],
        "memory_per_process": 50 * 1024 * 1024,
        "section_headings": ["Skills", "Experience"],
        "weighting": {"tfidf_weight": 0.7, "frequency_weight": 0.3, "whitelist_boost": 1.5},
        "max_retries": 2,
        "strict_mode": True,
        "semantic_validation": False,
        "similarity_threshold": 0.6,
        "text_encoding": "utf-8",
        "min_jobs": 2,
        "max_desc_length": 100000,
        "memory_threshold": 70,
        "max_memory_percent": 85,
        "max_workers": 4,
        "min_chunk_size": 1,
        "max_chunk_size": 1000,
        "auto_chunk_threshold": 100,
        "spacy_model": "en_core_web_sm",
        "model_download_retries": 2,
        "timeout": 600,
        "keyword_categories": {"Tech": ["python"], "Other": ["misc"]}
    }

@pytest.fixture
def mock_spacy_doc():
    """Fixture for a mock spaCy Doc object with realistic attributes."""
    doc = Mock()
    doc.text = "test text"
    doc.sents = [Mock(text="test text")]
    doc.__iter__.return_value = [Mock(text="test", lemma_="test")]
    doc.vector = np.array([1.0, 0.0])
    return doc

# --- Utility Function Tests ---
class TestUtilityFunctions(unittest.TestCase):
    @patch('test_keywords4cv.nltk.data.find')
    def test_ensure_nltk_resources_present(self, mock_find):
        """Test when all NLTK resources are already downloaded."""
        mock_find.return_value = "/path/to/resource"
        ensure_nltk_resources()
        mock_find.assert_called()

    @patch('test_keywords4cv.nltk.data.find')
    @patch('test_keywords4cv.nltk.download')
    def test_ensure_nltk_resources_missing(self, mock_download, mock_find):
        """Test downloading missing NLTK resources."""
        mock_find.side_effect = LookupError
        mock_download.return_value = True
        ensure_nltk_resources()
        mock_download.assert_called_with("wordnet", quiet=True)

    @patch('test_keywords4cv.nltk.data.find')
    @patch('test_keywords4cv.nltk.download')
    @patch('test_keywords4cv.logger')
    def test_ensure_nltk_resources_network_error(self, mock_logger, mock_download, mock_find):
        """Test handling of network errors during NLTK download."""
        mock_find.side_effect = LookupError
        mock_download.side_effect = Exception("Network error")
        ensure_nltk_resources()
        mock_logger.warning.assert_called()

    def test_cosine_similarity(self):
        """Test cosine similarity across valid, zero-norm, and negative cases."""
        self.assertAlmostEqual(cosine_similarity(np.array([1, 2]), np.array([2, 1])), 0.8, places=1)
        self.assertEqual(cosine_similarity(np.array([0, 0]), np.array([1, 2])), 0.0)
        self.assertEqual(cosine_similarity(np.array([1, 0]), np.array([-1, 0])), -1.0)

    @patch('builtins.open', new_callable=mock_open, read_data='{"Job1": "desc1"}')
    def test_load_job_data_valid(self, mock_file):
        """Test loading valid JSON job data."""
        result = load_job_data("test.json")
        self.assertEqual(result, {"Job1": "desc1"})

    @patch('builtins.open', new_callable=mock_open, read_data='{invalid json')
    @patch('test_keywords4cv.logger')
    @patch('sys.exit')
    def test_load_job_data_invalid_json(self, mock_exit, mock_logger, mock_file):
        """Test handling of invalid JSON data."""
        load_job_data("test.json")
        mock_exit.assert_called_with(1)
        mock_logger.error.assert_called()

    @patch('builtins.open')
    @patch('test_keywords4cv.logger')
    @patch('sys.exit')
    def test_load_job_data_file_not_found(self, mock_exit, mock_logger, mock_file):
        """Test handling of file not found error."""
        mock_file.side_effect = FileNotFoundError
        load_job_data("test.json")
        mock_exit.assert_called_with(1)

# --- EnhancedTextPreprocessor Tests ---
class TestEnhancedTextPreprocessor(unittest.TestCase):
    def setUp(self):
        """Set up the preprocessor with mocked spaCy and sample config."""
        patcher = patch('test_keywords4cv.spacy.load', return_value=Mock())
        self.addCleanup(patcher.stop)
        self.mock_spacy_load = patcher.start()
        self.nlp = Mock()
        self.config = sample_config()
        self.preprocessor = EnhancedTextPreprocessor(self.config, self.nlp)

    def test_init_valid_config(self):
        """Test initialization with valid configuration."""
        self.assertEqual(self.preprocessor.stop_words, {"the", "a"})
        self.assertEqual(self.preprocessor._CACHE_SIZE, 2)

    @patch('test_keywords4cv.logger')
    def test_small_stop_words_warning(self, mock_logger):
        """Test warning for small stop words list."""
        config = {"stop_words": ["a"], "cache_size": 2}
        EnhancedTextPreprocessor(config, self.nlp)
        mock_logger.warning.assert_called()

    def test_preprocess_basic(self):
        """Test basic text preprocessing."""
        result = self.preprocessor.preprocess("Hello http://example.com World!")
        self.assertEqual(result, "hello world")

    def test_preprocess_cache(self):
        """Test caching of preprocessed text."""
        text = "Test text"
        result1 = self.preprocessor.preprocess(text)
        result2 = self.preprocessor.preprocess(text)
        self.assertEqual(result1, result2)
        self.assertIn(text, self.preprocessor._cache)

    def test_cache_eviction(self):
        """Test cache eviction when exceeding size limit."""
        self.preprocessor.preprocess("Text1")
        self.preprocessor.preprocess("Text2")
        self.preprocessor.preprocess("Text3")
        self.assertNotIn("Text1", self.preprocessor._cache)

    def test_config_change_clears_cache(self):
        """Test cache clearing on config change."""
        text = "Test text"
        self.preprocessor.preprocess(text)
        self.config["stop_words"].append("test")
        self.preprocessor.preprocess(text)
        self.assertNotIn(text, self.preprocessor._cache)

    def test_preprocess_batch(self):
        """Test batch preprocessing of multiple texts."""
        result = self.preprocessor.preprocess_batch(["Text 1 http://a.com", "Text 2"])
        self.assertEqual(result, ["text 1", "text 2"])

    @patch('test_keywords4cv.psutil.virtual_memory')
    def test_tokenize_batch_basic(self, mock_memory):
        """Test basic batch tokenization."""
        mock_memory.return_value.available = 100 * 1024 * 1024
        self.nlp.pipe.return_value = [Mock(__iter__=lambda x: ["test"])]
        result = self.preprocessor.tokenize_batch(["Test text"])
        self.assertEqual(result, [["test"]])

    @patch('test_keywords4cv.psutil.virtual_memory')
    @patch('test_keywords4cv.current_process')
    def test_tokenize_batch_daemon(self, mock_process, mock_memory):
        """Test tokenization in daemon process uses single thread."""
        mock_memory.return_value.available = 100 * 1024 * 1024
        mock_process.return_value.daemon = True
        self.nlp.pipe.return_value = [Mock(__iter__=lambda x: ["test"])]
        result = self.preprocessor.tokenize_batch(["Test text"])
        self.nlp.pipe.assert_called_with(["Test text"], batch_size=2, n_process=1)

    def test_calculate_config_hash(self):
        """Test config hash changes with configuration updates."""
        hash1 = self.preprocessor._calculate_config_hash()
        self.config["stop_words"].append("new")
        hash2 = self.preprocessor._calculate_config_hash()
        self.assertNotEqual(hash1, hash2)

    def test_load_stop_words(self):
        """Test loading stop words with additions and exclusions."""
        config = {"stop_words": ["the", "a"], "stop_words_add": ["and"], "stop_words_exclude": ["a"], "cache_size": 2}
        preprocessor = EnhancedTextPreprocessor(config, self.nlp)
        self.assertEqual(preprocessor.stop_words, {"the", "and"})

    def test_process_doc_tokens(self):
        """Test token processing from spaCy doc."""
        doc = Mock()
        doc.__iter__.return_value = [Mock(text="The", lemma_="the"), Mock(text="cat", lemma_="cat"), Mock(text="1", lemma_="1")]
        result = self.preprocessor._process_doc_tokens(doc)
        self.assertEqual(result, ["cat"])

# --- AdvancedKeywordExtractor Tests ---
class TestAdvancedKeywordExtractor(unittest.TestCase):
    def setUp(self):
        """Set up the keyword extractor with mocked spaCy and sample config."""
        patcher = patch('test_keywords4cv.spacy.load', return_value=Mock())
        self.addCleanup(patcher.stop)
        self.mock_spacy_load = patcher.start()
        self.nlp = Mock()
        self.config = sample_config()
        self.extractor = AdvancedKeywordExtractor(self.config, self.nlp)

    def test_init_valid(self):
        """Test initialization with valid configuration."""
        self.assertEqual(self.extractor.ngram_range, (1, 3))
        self.assertIn("python", self.extractor.whitelist)

    @patch('test_keywords4cv.wordnet.synsets')
    def test_create_expanded_whitelist(self, mock_synsets):
        """Test whitelist expansion with synonyms."""
        mock_synsets.return_value = [Mock(lemmas=lambda: [Mock(name=lambda: "code")])]
        whitelist = self.extractor._create_expanded_whitelist()
        self.assertIn("python", whitelist)
        self.assertIn("code", whitelist)

    def test_extract_keywords_basic(self):
        """Test basic keyword extraction without semantic validation."""
        self.extractor.preprocessor.tokenize_batch.return_value = [["python", "coding"]]
        result = self.extractor.extract_keywords(["Python coding"])
        self.assertIn("python", result[0])
        self.assertIn("coding", result[0])

    @patch('test_keywords4cv.logger')
    def test_generate_synonyms_empty(self, mock_logger):
        """Test synonym generation skips empty skills."""
        result = self.extractor._generate_synonyms([""])
        mock_logger.warning.assert_called_with("Skipping empty skill in whitelist")
        self.assertEqual(result, set())

    def test_extract_sections(self):
        """Test section extraction from text."""
        self.nlp.return_value.sents = [Mock(text="Skills: Python"), Mock(text="coding")]
        result = self.extractor._extract_sections("Skills: Python\ncoding")
        self.assertEqual(result["Skills"], " Python")
        self.assertEqual(result["General"], " coding")

    def test_generate_ngrams(self):
        """Test n-gram generation from tokens."""
        tokens = ["python", "coding", "skills"]
        result = self.extractor._generate_ngrams(tokens, 2)
        self.assertEqual(result, {"python coding", "coding skills"})

    def test_semantic_filter(self):
        """Test semantic filtering of keywords based on context."""
        keywords = [["python"], ["java"]]
        texts = ["I use python daily", "Coding in Python"]
        result = self.extractor._semantic_filter(keywords, texts)
        self.assertEqual(result, [["python"], []])

    def test_is_in_context(self):
        """Test checking if a keyword is in context within text."""
        self.assertTrue(self.extractor._is_in_context("python", "Skills: Python coding"))
        self.assertFalse(self.extractor._is_in_context("java", "Skills: Python coding"))

    @patch('test_keywords4cv.cosine_similarity')
    def test_semantic_categorization(self, mock_cosine):
        """Test semantic categorization using cosine similarity."""
        self.extractor.category_vectors = {"Tech": {"centroid": np.array([1, 0]), "terms": ["python"]}, "Other": {"centroid": None, "terms": []}}
        mock_cosine.return_value = 0.8
        category = self.extractor._semantic_categorization("coding")
        self.assertEqual(category, "Tech")

    def test_get_term_vector_failure(self):
        """Test handling of vectorization failure."""
        self.nlp.side_effect = ValueError("Vector error")
        result = self.extractor._get_term_vector("test")
        self.assertTrue(np.array_equal(result, np.array([])))

    def test_categorize_term(self):
        """Test term categorization with direct match and semantic fallback."""
        self.extractor.category_vectors = {"Tech": {"centroid": np.array([1, 0]), "terms": ["python"]}, "Other": {"centroid": None, "terms": []}}
        self.assertEqual(self.extractor._categorize_term("python"), "Tech")
        with patch.object(self.extractor, '_semantic_categorization', return_value="Other"):
            self.assertEqual(self.extractor._categorize_term("unknown"), "Other")

# --- ATSOptimizer Tests ---
class TestATSOptimizer(unittest.TestCase):
    def setUp(self):
        """Set up the optimizer with mocked dependencies."""
        patcher1 = patch('test_keywords4cv.spacy.load', return_value=Mock(pipe_names=["ner"]))
        patcher2 = patch('builtins.open', new_callable=mock_open, read_data="stop_words: [the]\nskills_whitelist: [python]")
        patcher3 = patch('test_keywords4cv.yaml.safe_load', return_value=sample_config())
        self.addCleanup(patcher1.stop)
        self.addCleanup(patcher2.stop)
        self.addCleanup(patcher3.stop)
        self.mock_spacy_load = patcher1.start()
        self.mock_file = patcher2.start()
        self.mock_yaml = patcher3.start()
        self.optimizer = ATSOptimizer("config.yaml")

    def test_init_valid_config(self):
        """Test initialization with valid configuration."""
        self.assertIsInstance(self.optimizer.keyword_extractor, AdvancedKeywordExtractor)

    @patch('sys.version_info', (3, 7))
    def test_init_old_python(self):
        """Test raising exception for unsupported Python version."""
        with self.assertRaises(Exception) as context:
            ATSOptimizer("config.yaml")
        self.assertIn("Requires Python 3.8+", str(context.exception))

    @patch('test_keywords4cv.yaml.safe_load')
    def test_load_config_invalid_yaml(self, mock_yaml):
        """Test handling of invalid YAML configuration."""
        mock_yaml.side_effect = yaml.YAMLError("Invalid YAML")
        with self.assertRaises(ConfigError):
            self.optimizer._load_config("config.yaml")

    def test_analyze_jobs_normal(self):
        """Test normal job analysis without chunking."""
        jobs = {"Job1": "Python coding"}
        with patch.object(self.optimizer, '_analyze_jobs_internal', return_value=(pd.DataFrame(), pd.DataFrame())):
            summary, details = self.optimizer.analyze_jobs(jobs)
            self.assertIsInstance(summary, pd.DataFrame)

    @patch('test_keywords4cv.psutil.virtual_memory')
    def test_needs_chunking(self):
        """Test chunking decision based on job count and memory."""
        mock_memory.return_value.percent = 80
        self.assertTrue(self.optimizer._needs_chunking({"Job1": "desc1"}))

    def test_validate_input_dict(self):
        """Test validation of dictionary input."""
        result = self.optimizer._validate_input({"Job1": "Python coding"})
        self.assertEqual(result, {"Job1": "Python coding"})

    def test_validate_input_list(self):
        """Test conversion and validation of list input."""
        result = self.optimizer._validate_input(["Python coding"])
        self.assertEqual(result, {"Job_1": "Python coding"})

    def test_add_entity_ruler(self):
        """Test adding entity ruler to spaCy pipeline."""
        self.optimizer._add_entity_ruler()
        self.optimizer.nlp.add_pipe.assert_called_with("entity_ruler", before="ner")

    def test_init_categories(self):
        """Test initialization of keyword categories."""
        self.optimizer._init_categories()
        self.assertIn("Tech", self.optimizer.keyword_extractor.category_vectors)

    def test_validate_config(self):
        """Test validation of configuration parameters."""
        self.optimizer.config.pop("skills_whitelist")
        with self.assertRaises(ConfigError):
            self.optimizer._validate_config()

    @patch('test_keywords4cv.spacy.load')
    def test_try_load_model(self, mock_spacy_load):
        """Test loading a spaCy model."""
        mock_spacy_load.return_value = Mock(pipe_names=["lemmatizer"])
        nlp = self.optimizer._try_load_model("en_core_web_sm")
        self.assertIsNotNone(nlp)

    @patch('test_keywords4cv.spacy.cli.download')
    def test_download_model(self, mock_download):
        """Test downloading a spaCy model."""
        mock_download.return_value = True
        result = self.optimizer._download_model("en_core_web_sm")
        self.assertTrue(result)

    def test_create_fallback_model(self):
        """Test creation of a fallback spaCy model."""
        nlp = self.optimizer._create_fallback_model()
        self.assertIn("sentencizer", nlp.pipe_names)

    @patch('test_keywords4cv.TfidfVectorizer')
    def test_create_tfidf_matrix(self, mock_vectorizer):
        """Test creation of TF-IDF matrix."""
        mock_vectorizer.return_value.fit_transform.return_value = Mock()
        mock_vectorizer.return_value.get_feature_names_out.return_value = ["python"]
        dtm, features = self.optimizer._create_tfidf_matrix(["Python coding"], [["python"]])
        self.assertEqual(features, ["python"])

    def test_calculate_scores(self):
        """Test calculation of keyword scores."""
        dtm = Mock(tocoo=lambda: Mock(row=[0], col=[0], data=[0.5]))
        results = self.optimizer._calculate_scores(dtm, ["python"], [["python"]], {"Job1": "desc"})
        self.assertEqual(results[0]["Keyword"], "python")
        self.assertGreater(results[0]["Score"], 0)

    def test_analyze_jobs_chunked(self):
        """Test chunked job analysis."""
        with patch.object(self.optimizer, '_process_chunk', return_value=(pd.DataFrame(), pd.DataFrame())):
            summary, details = self.optimizer._analyze_jobs_chunked({"Job1": "desc"})
            self.assertIsInstance(summary, pd.DataFrame)

    def test_process_chunk(self):
        """Test processing a single job chunk."""
        chunk = {"Job1": "Python coding"}
        with patch.object(self.optimizer, '_create_tfidf_matrix', return_value=(Mock(), ["python"])):
            summary, details = self.optimizer._process_chunk(chunk)
            self.assertIsInstance(summary, pd.DataFrame)

    def test_calculate_chunk_size(self):
        """Test calculation of chunk size based on memory."""
        jobs = {"Job1": "a" * 1000, "Job2": "b" * 1000}
        with patch('test_keywords4cv.psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.available = 10 * 1024 * 1024
            size = self.optimizer._calculate_chunk_size(jobs)
            self.assertGreater(size, 0)

    @patch('test_keywords4cv.psutil.virtual_memory')
    def test_memory_check(self, mock_memory):
        """Test memory check clears caches when usage is high."""
        mock_memory.return_value.percent = 80
        self.optimizer._memory_check()
        self.assertEqual(len(self.optimizer.keyword_extractor.preprocessor._cache), 0)

    def test_clear_caches(self):
        """Test clearing of all caches."""
        self.optimizer.keyword_extractor.preprocessor._cache["test"] = "test"
        self.optimizer._clear_caches()
        self.assertEqual(len(self.optimizer.keyword_extractor.preprocessor._cache), 0)

    def test_validate_title(self):
        """Test validation of job titles."""
        result = self.optimizer._validate_title("Software Engineer")
        self.assertTrue(result.valid)
        result = self.optimizer._validate_title("")
        self.assertFalse(result.valid)

    def test_validate_description(self):
        """Test validation of job descriptions."""
        result = self.optimizer._validate_description("Python coding experience")
        self.assertTrue(result.valid)
        result = self.optimizer._validate_description("a" * 100001)
        self.assertFalse(result.valid)

# --- CLI and End-to-End Tests ---
class TestCLI(unittest.TestCase):
    @patch('sys.argv', ['script.py', '-i', 'input.json'])
    def test_parse_arguments_custom(self):
        """Test parsing custom command-line arguments."""
        args = parse_arguments()
        self.assertEqual(args.input, "input.json")

    @patch('test_keywords4cv.initialize_analyzer')
    @patch('test_keywords4cv.load_job_data')
    @patch('test_keywords4cv.save_results')
    def test_run_analysis(self, mock_save, mock_load, mock_init):
        """Test running the full analysis workflow."""
        args = Mock(config="config.yaml", input="input.json", output="output.xlsx")
        mock_load.return_value = {"Job1": "desc1"}
        mock_analyzer = Mock()
        mock_analyzer.analyze_jobs.return_value = (pd.DataFrame(), pd.DataFrame())
        mock_init.return_value = mock_analyzer
        run_analysis(args)
        mock_save.assert_called()

    @patch('test_keywords4cv.pd.ExcelWriter')
    def test_save_results(self, mock_writer):
        """Test saving results to Excel."""
        summary = pd.DataFrame()
        details = pd.DataFrame()
        save_results(summary, details, "output.xlsx")
        mock_writer.return_value.__enter__.return_value.to_excel.assert_called()

# --- Integration Tests ---
class TestIntegration(unittest.TestCase):
    @patch('test_keywords4cv.spacy.load')
    def test_full_pipeline(self, mock_spacy_load):
        """Test the full pipeline with mocked spaCy."""
        nlp = Mock()
        nlp.pipe.return_value = [Mock(__iter__=lambda x: ["python", "coding"])]
        mock_spacy_load.return_value = nlp
        optimizer = ATSOptimizer("config.yaml")
        jobs = {"Job1": "Python coding experience"}
        summary, details = optimizer.analyze_jobs(jobs)
        self.assertFalse(summary.empty)

# --- Performance Tests ---
class TestPerformance(unittest.TestCase):
    def setUp(self):
        """Set up for performance testing."""
        patcher = patch('test_keywords4cv.spacy.load', return_value=Mock())
        self.addCleanup(patcher.stop)
        self.mock_spacy_load = patcher.start()
        self.nlp = Mock()
        self.config = sample_config()
        self.preprocessor = EnhancedTextPreprocessor(self.config, self.nlp)

    def test_tokenize_batch_performance(self):
        """Test performance of batch tokenization with large input."""
        import timeit
        texts = ["Python coding " * 100] * 1000
        self.nlp.pipe.return_value = [Mock(__iter__=lambda x: ["python", "coding"])] * 1000
        execution_time = timeit.timeit(lambda: self.preprocessor.tokenize_batch(texts), number=1)
        self.assertLess(execution_time, 5.0)  # Adjust threshold based on hardware

# --- Running Tests ---
if __name__ == '__main__':
    unittest.main()