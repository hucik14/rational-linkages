from rational_linkages.models import collisions_free_6r
from rational_linkages import SingularityAnalysis


m = collisions_free_6r()
m.update_segments()

sa = SingularityAnalysis()

r = sa.check_singularity(m)

print(r)

