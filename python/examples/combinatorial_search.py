# NOT TESTED

from rational_linkages.models import collisions_free_6r
from rational_linkages import Plotter


if __name__ == '__main__':
    m = collisions_free_6r()

    coll_free = m.collision_free_optimization(step_length=9,
                                              min_joint_segment_length=0.1,
                                              max_iters=30)

    plt = Plotter(mechanism=m, show_tool=False, joint_sliders_lim=4)
