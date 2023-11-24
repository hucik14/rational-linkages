Joint Angle to Curve Parameter
==============================


The packages uses rational motion parametrization, i.e. every curve :math:`C(t)` is
defined by a list of linear factors (parametric equations) in Dual Quaternion space :math:`\mathbb{DH}`
of the form:

.. math::
   C(t) = (t - h_1)(t - h_2) \ldots (t - h_n)

where :math:`t` is the parameter and :math:`h_i` are the dual quaternions representing
the joints.

Parameter t is a curve parameter, i.e. it is not a joint angle. To convert joint angle,
the method :meth:`.MotionFactorization.joint_angle_to_t_param` is used.

It takes the arguments of the **driving** joint angle in radians (or degrees if
specified) and returns the curve parameter :math:`t`.

The relation between the joint angle and the curve parameter is obtained from
the relation between the joint angle and rotational quaternion (primal quaternonion of
the dual quaternion representing the joint). In general, a quaternion with given
rotation angle :math:`\phi` around axis vector :math:`\vec{u}` is defined as:

.. math::
   q = \mathrm{cos}(\frac{\phi}{2}) + \mathrm{sin}(\frac{\phi}{2})\vec{u} =
   \mathrm{cos}(\frac{\phi}{2}) + \mathrm{sin}(\frac{\phi}{2})(q_1\mathbf{i} +
   q_2\mathbf{j} + q_3\mathbf{k})

where :math:`\mathbf{i}, \mathbf{j}, \mathbf{k}` are the basis vectors of the quaternion
space. The quaternion is represented as a vector with 4 elements :math:`q = [q_0, q_1, q_2, q_3]`.

We can view the :math:`t` parameter as a vector, too, and in the case of pure rotation
it will then be :math:`t = [t_0, 0, 0, 0]`. If we plug this in
the quaternion equation, we get:

.. math::
   t - q = t_0 - \mathrm{cos}(\frac{\phi}{2})
   - \mathrm{sin}(\frac{\phi}{2})(q_1\mathbf{i} + q_2\mathbf{j} + q_3\mathbf{k})

We put the right side equal to zero and solve for :math:`t_0`:

.. math::
   t_0 = \mathrm{cos}(\frac{\phi}{2}) + \mathrm{sin}(\frac{\phi}{2})(q_1\mathbf{i} +
   q_2\mathbf{j} + q_3\mathbf{k})

The right-hand side we can simplify to:

.. math::
   t_0 = \sqrt{q q^*} \mathrm{cotan}(\frac{\phi}{2}) + q_0

Which is the equation used in the function
:meth:`.MotionFactorization.joint_angle_to_t_param`. :math:`q q^*` is the norm of
the quaternion :math:`q` and :math:`q_0` is its real part.