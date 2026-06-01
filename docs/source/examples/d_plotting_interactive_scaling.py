"""
# Interactive plotting with a loaded mechanism model, adjusted scaling
"""

from rational_linkages import Plotter
from rational_linkages.models import bennett_ark24 as bennett


m = bennett()

plt = Plotter(mechanism=m, arrows_length=0.05, joint_sliders_lim=0.5)
plt.show()

