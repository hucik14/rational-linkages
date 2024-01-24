Loading a prepared model
========================

The package can load a prepared model from its source. So far, only a Bennett mechanism
model from `ARK paper - extended info`_ is supported. The model is loaded from a file
with the following code:

.. testcode::

    # Loading a model from the package

    from rational_linkages import Plotter, TransfMatrix
    from rational_linkages.models import bennett_ark24


    if __name__ == "__main__":
        # load the model of the Bennett's linkage
        m = bennett_ark24()

        # create an interactive plotter object, with 500 descrete steps
        # for the input rational curves, and arrows scaled to 0.05 length
        myplt = Plotter(interactive=True, steps=500, arrows_length=0.05)

        # plot the model with tool frame
        myplt.plot(m, show_tool=True)

        ##### additional plotting options #####
        # create a pose of the identity
        base = TransfMatrix()
        myplt.plot(base)

        # create another pose
        p0 = TransfMatrix.from_rpy_xyz([-90, 0, 0], [0.15, 0, 0], units='deg')
        myplt.plot(p0)
        ######################################

        # show the plot
        myplt.show()

.. _ARK paper - extended info: ../tutorials/ark2024.rst