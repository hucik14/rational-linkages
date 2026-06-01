from rational_linkages import Plotter, TransfMatrix
from rational_linkages.models import bennett_ark24

# load the model of the Bennett's linkage
m = bennett_ark24()

# create an interactive plotter object, with 500 descrete steps
# for the input rational curves, and arrows scaled to 0.05 length
myplt = Plotter(mechanism=m, steps=500, arrows_length=0.05)

##### additional plotting options #####
# create a pose of the identity
base = TransfMatrix()
myplt.plot(base)

# create another pose
p0 = TransfMatrix.from_rpy_xyz([-90, 0, 0], [0.15, 0, 0], unit='deg')
myplt.plot(p0, label='p0')
######################################

# show the plot
myplt.show()
