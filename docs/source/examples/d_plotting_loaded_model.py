from rational_linkages import Plotter, PointHomogeneous
from rational_linkages.models import bennett_ark24


# load the mechanism
m = bennett_ark24()

# create an interactive plotter object
plt = Plotter(mechanism=m, arrows_length=0.05)

# create a point with homogeneous coordinates w = 1, x = 2, y = -3, z = 1.5
point = PointHomogeneous([1, 0.5, -0.75, 0.25])

plt.plot(point, label='pt')
plt.show()