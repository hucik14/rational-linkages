Motion interpolation
====================

The package implements the method described in the paper by Hegedüs et al.
[#hedegus2015]_.

.. code-block:: python
    :caption: Loading a model from the package

    from rational_linkages import DualQuaternion, Plotter, FactorizationProvider, MotionInterpolation, RationalMechanism


    if __name__ == "__main__":
        # 4 poses
        p0 = DualQuaternion()  # identity
        p1 = DualQuaternion.as_rational([0, 0, 0, 1, 1, 0, 1, 0])
        p2 = DualQuaternion.as_rational([1, 2, 0, 0, -2, 1, 0, 0])
        p3 = DualQuaternion.as_rational([3, 0, 1, 0, 1, 0, -3, 0])

        # obtain the interpolated motion curve
        c = MotionInterpolation.interpolate([p1, p2, p3])

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

.. [#hedegus2015] Hegedüs, G., Schicho, J., and Schröcker, H. Four-Pose Synthesis of
    Angle-Symmetric 6R Linkages. *ASME. J. Mechanisms Robotics*. 2015.
    https://doi.org/10.1115/1.4029186 arxiv: https://arxiv.org/abs/1309.4959
