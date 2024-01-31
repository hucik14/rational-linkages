from rational_linkages import RationalMechanism
import numpy as np

class ExudynAnalysis:
    """
    Class for dynamics analysis using Exudyn package.
    """
    def __init__(self, mechanism: RationalMechanism):
        self.mechanism = mechanism

    def get_links_length(self):
        lenghts = np.zeros(self.mechanism.num_joints)
        for i in range(self.mechanism.num_joints):
            pass


