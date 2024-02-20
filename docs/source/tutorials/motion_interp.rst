Motion interpolation
====================

The package implements the method described in the paper by :footcite:t:`Hegeds2015`
for 4 poses interpolation using cubic rational function, that yields
6-revolute linkages, and the method by :footcite:t:`Brunnthaler2005` for
3 poses interpolation using quadratic rational functions, that yields 4-revolute
linkage, i.e. the Bennett mechanism.

.. testcode::

    # Cubic interpolation of 4 poses

    from rational_linkages import DualQuaternion, Plotter, FactorizationProvider, MotionInterpolation, RationalMechanism


    if __name__ == "__main__":
        # 4 poses
        p0 = DualQuaternion()  # identity
        p1 = DualQuaternion.as_rational([0, 0, 0, 1, 1, 0, 1, 0])
        p2 = DualQuaternion.as_rational([1, 2, 0, 0, -2, 1, 0, 0])
        p3 = DualQuaternion.as_rational([3, 0, 1, 0, 1, 0, -3, 0])

        # obtain the interpolated motion curve
        c = MotionInterpolation.interpolate([p0, p1, p2, p3])

        # factorize the motion curve
        fs = c.factorize()

        # create a mechanism from the factorization
        m = RationalMechanism(fs)

        # create an interactive plotter object, with 500 descrete steps
        # for the input rational curves, and arrows scaled to 0.05 length
        myplt = Plotter(interactive=True, steps=500, arrows_length=0.5)
        myplt.plot(m, show_tool=True)

        # plot the poses
        for pose in [p0, p1, p2, p3]:
            myplt.plot(pose)

        # show the plot
        myplt.show()


The following example applies the method by :footcite:t:`Brunnthaler2005`.
It is important to note that the method is providing a rational function that consists
of polynomials that are not monic. The implemented factorization method uses
the produced curve but returns factors that, if multiplied, will yield a monic
polynomial.
In practice, this means that the synthesized mechanism will be still able to perform a
desired motion and pass the given poses, but the visualization will however transform
the whole mechanism by the a static transformation :math:`p_2` (or the last pose if
named differently). To match the visualization with the originally given poses, the
easiest way is to pre-multiply the original poses with the :math:`p_2`.

.. testcode::

    # Quadratic interpolation of 3 poses

    from rational_linkages import DualQuaternion, Plotter, MotionInterpolation


    if __name__ == "__main__":
        p0 = DualQuaternion([0, 17, -33, -89, 0, -6, 5, -3])
        p1 = DualQuaternion([0, 84, -21, -287, 0, -30, 3, -9])
        p2 = DualQuaternion([0, 10, 37, -84, 0, -3, -6, -3])

        c = MotionInterpolation.interpolate([p0, p1, p2])

        plt = Plotter(interactive=False, steps=500, arrows_length=0.05)
        plt.plot(c, interval='closed')

        for i, pose in enumerate([p0, p1, p2]):
            plt.plot(pose, label='p{}'.format(i+1))
        plt.show()

**References**

.. footbibliography::

