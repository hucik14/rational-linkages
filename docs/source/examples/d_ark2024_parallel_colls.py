from rational_linkages.models import bennett_ark24

# on Windows, the script has to be run inside the if __name__ == '__main__'
# so the parallel processing can be used
if __name__ == '__main__':
    # load the mechanism
    m = bennett_ark24()

    # check for collisions
    m.collision_check(parallel=True)

    # generate the design
    dh, cp, _ = m.get_design(unit='deg', scale=200)