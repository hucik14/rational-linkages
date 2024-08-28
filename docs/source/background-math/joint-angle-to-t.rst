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
rotation angle :math:`\theta` around axis vector :math:`\vec{u}` is defined as:

.. math::
   \mathbf{q} = \mathrm{cos}(\frac{\theta}{2}) + \mathrm{sin}(\frac{\theta}{2})\vec{u} =
   \mathrm{cos}(\frac{\theta}{2}) + \mathrm{sin}(\frac{\theta}{2})(q_1\mathbf{i} +
   q_2\mathbf{j} + q_3\mathbf{k})

where :math:`\mathbf{i}, \mathbf{j}, \mathbf{k}` are the basis vectors of the quaternion
space. The quaternion is represented as a vector with 4 elements
:math:`q = [q_0, q_1, q_2, q_3]`. The element :math:`q_0` is the real part
of the quaternion and equals to :math:`\mathrm{cos}(\frac{\theta}{2})`.

We can view the :math:`\mathbf{t}` parameter as a vector, too, and in the case of pure rotation
it will then be :math:`\mathbf{t} = [t, 0, 0, 0]`. If we plug this in
the quaternion equation, we get:

.. math::
   \mathbf{t} - \mathbf{q} = t - \mathrm{cos}(\frac{\theta}{2})
   - \mathrm{sin}(\frac{\theta}{2})(q_1\mathbf{i} + q_2\mathbf{j} + q_3\mathbf{k})

This equation can be derived into the form:

.. math::
    t = \frac{\sqrt{q_1^2 + q_2^2 + q_3^2}}{\mathrm{tan}(\frac{\theta}{2})} + q_0

Which is the equation used in the function
:meth:`.MotionFactorization.joint_angle_to_t_param`. The inverse
method :meth:`.MotionFactorization.t_param_to_joint_angle` which takes the curve
parameter as an argument and returns the joint angle

.. math::
   \theta = 2\mathrm{arctan}\Bigg(\frac{\sqrt{q_1^2 + q_2^2 + q_3^2}}{t - q_0}\Bigg)
