from rational_linkages import Plotter
from rational_linkages.models import bennett_ark24


m = bennett_ark24()

p = Plotter(backend="matplotlib", arrows_length=0.03)
p.plot(m.curve(), interval='closed', with_poses=True)
p.show()

