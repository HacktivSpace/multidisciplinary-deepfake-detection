import unittest
import os
from src.utils.helpers import some_helper_function
from src.utils.logger import setup_logger
from src.utils.data_utils import load_data, save_data
from src.utils.file_utils import create_directory, delete_file
import pandas as pd
import numpy as np

class TestUtils(unittest.TestCase):
    def setUp(self):
        """
        Setting up test variables and environment.
        """
        self.test_dir = 'tests/temp'
        os.makedirs(self.test_dir, exist_ok=True)
        self.test_file = os.path.join(self.test_dir, 'test_file.csv')
        self.data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })

    def test_some_helper_function(self):
        """
        To test helper function.
        """
        result = some_helper_function(self.data)
        self.assertTrue(result)

    def test_setup_logger(self):
        """
        To test logger setup.
        """
        logger = setup_logger('test_logger', log_file=os.path.join(self.test_dir, 'test_log.log'))
        logger.info('This is a test log message.')
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'test_log.log')))

    def test_load_data(self):
        """
        To test loading data.
        """
        self.data.to_csv(self.test_file, index=False)
        loaded_data = load_data(self.test_file)
        pd.testing.assert_frame_equal(loaded_data, self.data)

    def test_save_data(self):
        """
        To test saving data.
        """
        save_data(self.data, self.test_file)
        self.assertTrue(os.path.exists(self.test_file))
        loaded_data = pd.read_csv(self.test_file)
        pd.testing.assert_frame_equal(loaded_data, self.data)

    def test_create_directory(self):
        """
        To test creating directory.
        """
        new_dir = os.path.join(self.test_dir, 'new_dir')
        create_directory(new_dir)
        self.assertTrue(os.path.exists(new_dir))

    def test_delete_file(self):
        """
        To test deleting file.
        """
        self.data.to_csv(self.test_file, index=False)
        delete_file(self.test_file)
        self.assertFalse(os.path.exists(self.test_file))

    def tearDown(self):
        """
        Cleaning up after tests.
        """
        import shutil
        shutil.rmtree(self.test_dir)

if __name__ == "__main__":
    unittest.main()
