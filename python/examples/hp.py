from rational_linkages import DualQuaternion, Plotter, RationalMechanism, MotionFactorization


if __name__ == '__main__':
    # Create two DualQuaternion objects, h1 and h2, if they are rotational quaternions,
    # set is_rotation=True; by default, is_rotation=False
    h1 = DualQuaternion([0, 0, 0, 1, 0, 0, 0, 0], is_rotation=True)
    h2 = DualQuaternion([0, 0, 0, 2, 0, 0, -1, 0], is_rotation=True)

    # Create a MotionFactorization object, f1, with h1 and h2 as parameters
    f1 = MotionFactorization([h1, h2])

    # Create another MotionFactorization object, f2, with two new DualQuaternion objects as parameters
    f2 = MotionFactorization(
        [DualQuaternion([0, 0, 0, 2, 0, 0, -1 / 3, 0], is_rotation=True),
         DualQuaternion([0, 0, 0, 1, 0, 0, -2 / 3, 0], is_rotation=True)])

    # f2 can be also factorized using line bellow, it returns a list of MotionFactorization objects
    # f2 = f1.factorize()[0]  # or [1] if equal

    # Create a RationalMechanism object, with f1 and f2 as parameters
    m = RationalMechanism([f1, f2])
    m.curve()


    # Create a Plotter object, that is interactive, with 500 descrete steps for plotting
    # curves, and poses frame arrows will be 0.2 units long
    plt = Plotter(interactive=True, steps=500, arrows_length=0.2)

    # Plot the RationalMechanism object with the tool shown
    plt.plot(m, show_tool=True)

    # Display the plot
    plt.show()
