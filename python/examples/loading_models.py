from rational_linkages.models import bennett_ark24, collisions_free_6r
from rational_linkages import Plotter

# load the model of the Bennett's linkage
bennett = bennett_ark24()

# load the model of the 6R mechanism
r6 = collisions_free_6r()


if __name__ == '__main__':

    m = r6

    p = Plotter(interactive=True, arrows_length=0.2, joint_range_lim=2, steps=500)
    p.plot(m, show_tool=True)

    p.show()
