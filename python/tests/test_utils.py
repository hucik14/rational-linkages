from unittest import TestCase

from rational_linkages.utils import is_package_installed


class TestUtils(TestCase):
    def test_is_package_installed(self):
        self.assertTrue(is_package_installed('numpy'))  # assuming numpy is installed

        self.assertFalse(is_package_installed(
            'some_non_existent_package'))  # assuming this package does not exist