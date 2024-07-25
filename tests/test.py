# tests/test_process_message.py
import unittest
from unittest.mock import patch, MagicMock
import sys

import re

# Insert path for custom modules
sys.path.insert(1, '/home/jabez/Documents/week_7/Precision-RAG/scripts')
from query_data import process_message

class TestProcessMessage(unittest.TestCase):

    @patch('query_data.ChatOpenAI.predict')
    @patch('query_data.Chroma.similarity_search_with_relevance_scores')
    @patch('query_data.OpenAIEmbeddings')
    @patch('query_data.load_dotenv')
    @patch('query_data.os.getenv')
    def test_process_message(self, mock_getenv, mock_load_dotenv, mock_openai_embeddings, mock_similarity_search, mock_predict):
        # Set up the mocks
        mock_getenv.return_value = 'fake_api_key'
        
        mock_openai_embeddings.return_value = MagicMock()
        mock_similarity_search.return_value = [
            (MagicMock(metadata={'source': 'Source1', 'article': '55'}, page_content="Content1"), 0.9),
            (MagicMock(metadata={'source': 'Source2', 'article': '60'}, page_content="Content2"), 0.8),
            (MagicMock(metadata={'source': 'Source3', 'article': '70'}, page_content="Content3"), 0.75),
        ]
        mock_predict.return_value = "The procedure for arrest is defined in Article 55 and Article 60."

        # Call the function
        query = "What is the procedure for arrest under Ethiopian criminal law?"
        response = process_message(query)

        # Assertions
        expected_response = (
            "Response: The procedure for arrest is defined in Article 55 and Article 60.\n"
            "Sources: 55, 60, 70"
        )
        self.assertEqual(response, expected_response)
        mock_load_dotenv.assert_called_once()
        mock_getenv.assert_called_with('OPENAI_API_KEY')
        mock_similarity_search.assert_called_once_with(query, k=3)
        mock_predict.assert_called_once()

    @patch('query_data.os.getenv')
    def test_process_message_no_api_key(self, mock_getenv):
        # Set up the mock
        mock_getenv.return_value = None

        # Call the function and check for ValueError
        query = "What is the procedure for arrest under Ethiopian criminal law?"
        with self.assertRaises(ValueError) as context:
            process_message(query)

        self.assertEqual(str(context.exception), "OpenAI API key is not set in the environment variables.")

    @patch('query_data.Chroma.similarity_search_with_relevance_scores')
    @patch('query_data.OpenAIEmbeddings')
    @patch('query_data.load_dotenv')
    @patch('query_data.os.getenv')
    def test_process_message_no_results(self, mock_getenv, mock_load_dotenv, mock_openai_embeddings, mock_similarity_search):
        # Set up the mocks
        mock_getenv.return_value = 'fake_api_key'
        mock_openai_embeddings.return_value = MagicMock()
        mock_similarity_search.return_value = []

        # Call the function
        query = "What is the procedure for arrest under Ethiopian criminal law?"
        response = process_message(query)

        # Assertions
        self.assertIsNone(response)
        mock_load_dotenv.assert_called_once()
        mock_getenv.assert_called_with('OPENAI_API_KEY')
        mock_similarity_search.assert_called_once_with(query, k=3)

if __name__ == "__main__":
    unittest.main()
