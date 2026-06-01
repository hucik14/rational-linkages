# Define 6-bar mechanism from dual quaternions ijk representation
from rational_linkages.StaticMechanism import StaticMechanism
from sympy import symbols

epsilon, i, j, k = symbols('epsilon i j k')


linkage = [epsilon*k + i,
           epsilon*i + epsilon*k + j,
           epsilon*i + epsilon*j + k,
           -epsilon*k + i,
           epsilon*i - epsilon*k - j,
           epsilon*i - epsilon*j - k]

m = StaticMechanism.from_ijk_representation(linkage)
m.get_design(unit='deg')