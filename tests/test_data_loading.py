import unittest
import os
import numpy as np
import cv2
from src.dataset.data_loader import create_dataloader

class TestDataLoading(unittest.TestCase):
    def setUp(self):
        """
        Setting up test variables and environment.
        """
        self.data_csv = 'tests/data/sample_data.csv'
        self.image_dir = 'tests/data/images'
        self.batch_size = 4
        self.num_workers = 2

        os.makedirs('tests/data/images', exist_ok=True)
        with open(self.data_csv, 'w') as f:
            f.write('image,label\n')
            for i in range(10):
                image_path = f'image_{i}.jpg'
                f.write(f'{image_path},{i % 2}\n')
                image = (255 * np.random.rand(224, 224, 3)).astype(np.uint8)
                cv2.imwrite(os.path.join(self.image_dir, image_path), image)

    def test_data_loading(self):
        """
        Testing data loading functionality.
        """
        dataloader = create_dataloader(self.data_csv, self.image_dir, batch_size=self.batch_size, num_workers=self.num_workers)
        
        batch_count = 0
        for images, labels in dataloader:
            self.assertEqual(len(images), self.batch_size)
            self.assertEqual(len(labels), self.batch_size)
            batch_count += 1
        
        self.assertGreater(batch_count, 0)

    def tearDown(self):
        """
        Cleaning up after tests.
        """
        import shutil
        shutil.rmtree('tests/data')

if __name__ == "__main__":
    unittest.main()
