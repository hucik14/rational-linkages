import pytest
from unittest.mock import MagicMock
from itertools import product
from rational_linkages.models import bennett_ark24
from rational_linkages import CollisionFreeOptimization
from rational_linkages.CollisionFreeOptimization import CombinatorialSearch


class TestCollisionFreeOptimization:
    def test_init(self):
        cfo = CollisionFreeOptimization(bennett_ark24())
        assert cfo
        assert isinstance(cfo, CollisionFreeOptimization)


class TestCombinatorialSearch:
    def test_init(self):
        cs = CombinatorialSearch(bennett_ark24(),
                                 linkage_length=1.322267221075116,
                                 step_length=10.0,
                                 min_joint_segment_length=0.001,
                                 max_iters=10)
        assert cs
        assert isinstance(cs, CombinatorialSearch)

    def test__get_combinations_sequences(self):
        mechanism = MagicMock()
        mechanism.num_joints = 3

        result = CombinatorialSearch(mechanism, linkage_length=1)._get_combinations_sequences(False)
        expected_result = list(product([0, 1, -1], repeat=mechanism.num_joints))
        expected_result.remove((0,) * mechanism.num_joints)
        assert result == expected_result
