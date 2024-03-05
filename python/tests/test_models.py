import unittest
import pickle
from unittest.mock import patch, mock_open
from rational_linkages.models import bennett_ark24, collisions_free_6r


class TestModels(unittest.TestCase):
    @patch('importlib.resources.path')
    @patch('builtins.open', new_callable=mock_open, read_data=pickle.dumps("Bennett data"))
    def test_bennett_ark24_returns_correct_data(self, mock_open, mock_path):
        result = bennett_ark24()
        self.assertEqual(result, "Bennett data")

    @patch('importlib.resources.path')
    @patch('builtins.open', new_callable=mock_open, read_data=pickle.dumps("6R data"))
    def test_collisions_free_6r_returns_correct_data(self, mock_open, mock_path):
        result = collisions_free_6r()
        self.assertEqual(result, "6R data")

    @patch('importlib.resources.path')
    @patch('builtins.open', new_callable=mock_open)
    def test_bennett_ark24_raises_error_when_file_not_found(self, mock_open, mock_path):
        mock_path.side_effect = FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            bennett_ark24()

    @patch('importlib.resources.path')
    @patch('builtins.open', new_callable=mock_open)
    def test_collisions_free_6r_raises_error_when_file_not_found(self, mock_open, mock_path):
        mock_path.side_effect = FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            collisions_free_6r()