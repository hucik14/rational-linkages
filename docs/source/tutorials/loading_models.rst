Loading Prepared Models
=======================

.. include:: ../refs-weblinks.rst

The package can load a prepared model from its source. So far, a **Bennett mechanism**
model from :ref:`ARK 2024 paper - extended information<ark2024extended>` is supported,
and a **collision-free 6R mechanism**.

.. testcode:: [loading_models_example1]

    from rational_linkages.models import bennett_ark24, collisions_free_6r

    # load the model of the Bennett's linkage
    bennett = bennett_ark24()

    # load the model of the 6R mechanism
    r6 = collisions_free_6r()

.. testcleanup:: [loading_models_example1]

    del bennett_ark24, collisions_free_6r
    del bennett, r6

Plotting a model can be done as in the following example.

.. testcode:: [loading_models_example2]

    # Loading a model from the package

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
    myplt.plot(p0)
    ######################################

    # show the plot
    myplt.show()

.. testcleanup:: [loading_models_example2]

    del Plotter, TransfMatrix, bennett_ark24
    del m, myplt, base, p0
