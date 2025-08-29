from typing import Union

import numpy as np
from scipy.optimize import minimize, differential_evolution

from .AffineMetric import AffineMetric
from .DualQuaternion import DualQuaternion
from .PointHomogeneous import PointHomogeneous
from .RationalCurve import RationalCurve

### NOT YET in the documentation ### TODO: add to docs


class MotionApproximation:
    """
    MotionApproximation class
    """
    def __init__(self):
        pass

    @staticmethod
    def approximate(init_curve,
                    poses_or_points: list[Union[DualQuaternion, PointHomogeneous]],
                    t_vals: Union[list[float], np.ndarray]
                    ) -> tuple[RationalCurve, dict]:
        """
        Approximate a motion curve that passes through the given poses

        :param RationalCurve init_curve: initial curve (guess), use interpolation
            algorithm from :class:`.MotionInterpolation.MotionInterpolation` to get
            a good initial guess
        :param list[Union[DualQuaternion, PointHomogeneous]] poses_or_points: poses
            or points to be approximated
        :param Union[list[float], np.ndarray] t_vals: parameter t values for the poses
            in the same order

        :return: Approximated curve and optimization result
        :rtype: tuple[RationalCurve, dict]
        """
        if init_curve.degree != 3:
            raise ValueError("So far, only cubic curves are supported")

        t_array = np.asarray(t_vals)

        if isinstance(poses_or_points[0], DualQuaternion):
            approx_curve, opt_result \
                = MotionApproximation._cubic_approximation(init_curve,
                                                           poses_or_points,
                                                           t_array)
        elif isinstance(poses_or_points[0], PointHomogeneous):
            approx_curve, opt_result \
                = MotionApproximation._cubic_approximation_for_points(init_curve,
                                                                      poses_or_points,
                                                                      t_array)
        else:
            raise TypeError("poses must be a list of DualQuaternion or PointHomogeneous objects")

        return approx_curve, opt_result

    @staticmethod
    def _construct_curve(flattended_coeffs) -> RationalCurve:
        """
        Construct a RationalCurve from the flattened coefficients

        :param flattended_coeffs: flattened coefficients

        :return: RationalCurve constructed from the coefficients
        :rtype: RationalCurve
        """
        coeffs = np.zeros((8, 4))  # Preallocate an array of shape (8, 4)
        coeffs[0, 0] = 1
        coeffs[:, 1:] = flattended_coeffs.reshape(8, 3)

        return RationalCurve.from_coeffs(coeffs)

    @staticmethod
    def _construct_curve_nonmonic(flattended_coeffs) -> RationalCurve:
        """
        Construct a RationalCurve from the flattened coefficients

        :param flattended_coeffs: flattened coefficients

        :return: RationalCurve constructed from the coefficients
        :rtype: RationalCurve
        """
        return RationalCurve.from_coeffs(flattended_coeffs.reshape(8, 4))

    @staticmethod
    def _cubic_approximation(init_curve,
                             poses,
                             t_vals) -> tuple[RationalCurve, dict]:
        """
        Get the curve of the cubic motion approximation

        :return: Approximated curve
        :rtype: tuple[RationalCurve, dict]
        """
        metric = AffineMetric(init_curve,
                              [PointHomogeneous.from_3d_point(pose.dq2point_via_matrix())
                               for pose in poses])

        num_added_poses = len(poses) - 4

        initial_guess = init_curve.coeffs[:,1:4].flatten()
        initial_guess = np.concatenate((initial_guess, t_vals[-num_added_poses:]), axis=None)

        def objective_function(params):
            """
            Objective function to minimize the sum of squared distances between
            the poses and the curve
            """
            curve = MotionApproximation._construct_curve(params[:24])

            for i in range(num_added_poses):
                val = i + 1
                t_vals[-val] = params[24:][i]

            sq_dist = 0.
            for i, pose in enumerate(poses):
                curve_pose = DualQuaternion(curve.evaluate(t_vals[i]))
                sq_dist += metric.squared_distance(pose, curve_pose)

            return sq_dist

        def constraint_func(params):
            curve = MotionApproximation._construct_curve(params[:24])
            sq_err = curve.study_quadric_check()

            if len(sq_err) != 8:  # expand if necessary to avoid index errors
                sq_err = np.concatenate((sq_err, np.zeros(8 - len(sq_err))), axis=None)

            return sq_err

        def callback(params):
            current_distance = objective_function(params)
            current_constraint = constraint_func(params)
            print(f"Objective function: {current_distance}, Constraints:")
            print(current_constraint)

        constraints = []
        for i in range(6):  # separate constraint functions for Study Quadric equation
            constraints.append({
                'type': 'eq',
                'fun': (lambda params, index=i: constraint_func(params)[index])
            })

        result = minimize(objective_function,
                          initial_guess,
                          constraints=constraints,
                          callback=callback,
                          options={'maxiter': 50,
                                   'ftol': 1e-16,
                                   },
                          )

        print(result)
        result_curve = MotionApproximation._construct_curve(result.x[:24])

        return result_curve, result

    @staticmethod
    def _cubic_approximation_for_points(init_curve,
                                        points,
                                        t_vals) -> tuple[RationalCurve, dict]:
        """
        Get the curve of the cubic motion approximation

        :return: Approximated curve
        :rtype: tuple[RationalCurve, dict]
        """
        t_vals_init = np.array([0, 1/6, 1/3, 1/2, 2/3, 5/6, 1])
        t_vals = np.concatenate((t_vals_init, t_vals), axis=None)

        num_added_points = len(points) - 7

        initial_guess = init_curve.coeffs.flatten()
        initial_guess = np.concatenate((initial_guess, t_vals[-num_added_points:]), axis=None)

        def objective_function(params):
            """
            Objective function to minimize the sum of squared distances between
            the poses and the curve
            """
            curve = MotionApproximation._construct_curve_nonmonic(params[:32])

            for i in range(num_added_points):
                val = i + 1
                t_vals[-val] = params[32:][i]

            sq_dist = 0.
            for i, pt in enumerate(points):
                # Get the 3D point from the curve
                curve_pt = DualQuaternion(
                    curve.evaluate(t_vals[i])).dq2point_via_matrix()
                target_pt = pt.normalized_in_3d()

                sq_dist += np.linalg.norm(curve_pt - target_pt) ** 2

            return sq_dist

        def constraint_func(params):
            curve = MotionApproximation._construct_curve_nonmonic(params[:32])
            sq_err = curve.study_quadric_check()

            if len(sq_err) != 8:  # expand if necessary to avoid index errors
                sq_err = np.concatenate((sq_err, np.zeros(8 - len(sq_err))), axis=None)

            return sq_err

        def callback(params):
            current_distance = objective_function(params)
            current_constraint = constraint_func(params)
            print(f"Objective function: {current_distance}, Constraints:")
            print(current_constraint)

        constraints = []
        for i in range(6):  # separate constraint functions for Study Quadric equation
            constraints.append({
                'type': 'eq',
                'fun': (lambda params, index=i: constraint_func(params)[index])
            })

        result = minimize(objective_function,
                          initial_guess,
                          constraints=constraints,
                          callback=callback,
                          options={'maxiter': 20,
                                   'ftol': 1e-14,
                                   },
                          )

        print(result)
        result_curve = MotionApproximation._construct_curve_nonmonic(result.x[:32])

        return result_curve, result

    @staticmethod
    def force_study_quadric(init_curve: RationalCurve):
        """
        For given curve, force it to be on the study quadric
        """
        initial_guess = init_curve.coeffs.flatten()

        def objective_func(params):
            curve = MotionApproximation._construct_curve_nonmonic(params[:32])
            sq_err = curve.study_quadric_check()

            # sum of squares of the errors
            return np.sum(sq_err**2)

        def callback(params):
            current_distance = objective_func(params)
            print(f"Objective function: {current_distance}")

        result = minimize(objective_func,
                          initial_guess,
                          method='Powell', # Powell, --TNC, --SLSQP
                          callback=callback,
                          tol=1e-14,
                          options={'maxiter': 100,
                                   'ftol': 1e-14,
                                   },
                          )

        print(result)
        result_curve = MotionApproximation._construct_curve_nonmonic(result.x[:32])

        return result_curve, result


    @staticmethod
    def optimize_for_t_and_study_quadric(init_curve: RationalCurve,
                                         points: list[PointHomogeneous],
                                         t_vals: np.ndarray):
        """
        Optimize the curve to be on the study quadric and find optimal parameter values

        :param RationalCurve init_curve:
        :param list[PointHomogeneous] points: points to be approximated
        :param np.ndarray t_vals: initial parameter t values for the points

        :return: Optimized curve and optimization result
        :rtype: tuple[RationalCurve, dict]
        """
        initial_guess = np.concatenate((init_curve.coeffs.flatten(), t_vals),
                                       axis=None)

        def objective_function(params):
            """
            Objective function to minimize the sum of squared distances between
            the poses and the curve
            """
            curve = MotionApproximation._construct_curve_nonmonic(params[:32])

            t_vals = params[32:]

            sq_dist = 0.
            for i, pt in enumerate(points):
                # Get the 3D point from the curve
                curve_pt = DualQuaternion(
                    curve.evaluate(t_vals[i])).dq2point_via_matrix()
                target_pt = pt.normalized_in_3d()

                sq_dist += np.linalg.norm(curve_pt - target_pt) ** 2

            return sq_dist

        def constraint_func(params):
            curve = MotionApproximation._construct_curve_nonmonic(params[:32])
            sq_err = curve.study_quadric_check()

            if len(sq_err) != 8:  # expand if necessary to avoid index errors
                sq_err = np.concatenate((sq_err, np.zeros(8 - len(sq_err))), axis=None)

            return sq_err

        def callback(params):
            current_distance = objective_function(params)
            current_constraint = constraint_func(params)
            print(f"Objective function: {current_distance}, Constraints:")
            print(current_constraint)

        constraints = []
        for i in range(6):  # separate constraint functions for Study Quadric equation
            constraints.append({
                'type': 'eq',
                'fun': (lambda params, index=i: constraint_func(params)[index])
            })

        result = minimize(objective_function,
                          initial_guess,
                          constraints=constraints,
                          callback=callback,
                          options={'maxiter': 20,
                                   'ftol': 1e-14,
                                   },
                          )

        print(result)
        result_curve = MotionApproximation._construct_curve_nonmonic(result.x[:32])
        print("Resulting t values:", t_vals)
        print(result.x[32:])

        return result_curve, result


    @staticmethod
    def optimize_for_t_and_study_quadric_global(init_curve: "RationalCurve",
                                                points: list["PointHomogeneous"],
                                                t_vals: np.ndarray):
        """
        Global+local optimization:
          • Global: differential evolution on a constraint-penalized objective
          • Local: SLSQP with explicit equality constraints (Study Quadric)
        """

        # ---- Helpers ---------------------------------------------------------
        def _pack(curve_coeffs: np.ndarray, tvals: np.ndarray) -> np.ndarray:
            return np.concatenate((curve_coeffs.flatten(), tvals), axis=None)

        def _unpack(params: np.ndarray):
            coeffs = params[:32]  # assumes 4x8 coeffs flattened
            tvals = params[32:]
            return coeffs, tvals

        def _curve_from_params(params: np.ndarray) -> "RationalCurve":
            coeffs, _ = _unpack(params)
            return MotionApproximation._construct_curve_nonmonic(coeffs)

        # Objective: sum of squared distances curve(t_i) ↔ target_i
        def _objective_raw(params: np.ndarray) -> float:
            curve = _curve_from_params(params)
            _, tvals = _unpack(params)

            sq_dist = 0.0
            for i, pt in enumerate(points):
                curve_pt = DualQuaternion(
                    curve.evaluate(tvals[i])).dq2point_via_matrix()
                target_pt = pt.normalized_in_3d()
                # squared L2 error
                diff = curve_pt - target_pt
                sq_dist += float(np.dot(diff, diff))
            return sq_dist

        # Constraint residuals (Study Quadric); shape (<=8,)
        def _constraint_residuals(params: np.ndarray) -> np.ndarray:
            curve = _curve_from_params(params)
            res = curve.study_quadric_check()
            # pad to length 8 for safety
            if len(res) != 8:
                res = np.concatenate((res, np.zeros(8 - len(res))), axis=None)
            return res

        # Penalized objective for global step
        def _objective_penalized(params: np.ndarray) -> float:
            f = _objective_raw(params)
            c = _constraint_residuals(params)
            # Smooth quadratic penalty + small L1 to help exploration
            penalty = np.dot(c, c) + 1e-3 * np.sum(np.abs(c))
            return f + mu * penalty

        # SLSQP equality constraints for local polish
        def _make_eq_constraints():
            cons = []
            for i in range(
                    6):  # keep first 6 residuals as hard equalities (like your code)
                cons.append({
                    'type': 'eq',
                    'fun': (lambda p, idx=i: _constraint_residuals(p)[idx])
                })
            return cons

        # ---- Initial guess ---------------------------------------------------
        initial_guess = _pack(init_curve.coeffs, t_vals)

        # ---- Bounds for global search ---------------------------------------
        # Differential evolution needs finite bounds; we derive them from the initial guess.
        coeff0 = init_curve.coeffs.flatten()
        # scale per-coefficient: generous range, but finite
        # (wider near zero; proportional when large)
        coeff_scale = np.maximum(1.0, 3.0 * np.abs(coeff0))
        coeff_lo = coeff0 - 5.0 * coeff_scale
        coeff_hi = coeff0 + 5.0 * coeff_scale

        # Robust t bounds (handle infinity tendency)
        t0 = np.asarray(t_vals, dtype=float)
        if t0.size == 0:
            raise ValueError("t_vals must be a non-empty array.")

        # Compute min/max ignoring zeros to avoid log issues
        t_min, t_max = float(np.min(t0)), float(np.max(t0))
        trange = t_max - t_min

        if trange < 1e-6:
            # Degenerate case: single t value; allow wide window
            t_lo_scalar = t_min - 100.0
            t_hi_scalar = t_min + 100.0
        else:
            # Expand by factor rather than additive margin
            factor = 10.0  # can tune (10x spread)
            t_lo_scalar = t_min - factor * abs(t_min if t_min != 0 else 1)
            t_hi_scalar = t_max + factor * abs(t_max if t_max != 0 else 1)

        # Clip to practical finite bounds for DE (avoid inf)
        t_lo_scalar = max(t_lo_scalar, -1e6)
        t_hi_scalar = min(t_hi_scalar, 1e6)

        t_lo = np.full_like(t0, t_lo_scalar)
        t_hi = np.full_like(t0, t_hi_scalar)

        bounds = list(zip(coeff_lo, coeff_hi)) + list(zip(t_lo, t_hi))

        # ---- Global step (DE) ------------------------------------------------
        # Penalty weight — large enough to bias toward feasibility, not so large
        # that it destroys exploration. You can tweak if needed.
        mu = 1e4

        # To make runs reproducible if desired, set seed here:
        rng_seed = 42

        de_result = differential_evolution(
            _objective_penalized,
            bounds=bounds,
            strategy='best1bin',
            maxiter=1000,  # global exploration budget
            popsize=20,  # larger pop helps in rough landscapes
            tol=1e-6,
            mutation=(0.5, 1.0),
            recombination=0.7,
            polish=False,  # we will do our own polish with SLSQP
            disp=True,
            seed=rng_seed,
            updating='deferred',  # good performance on larger dims
            workers=1,  # set >1 if your environment supports it
        )

        # # ---- Local polish (SLSQP with equality constraints) ------------------
        # slsqp_constraints = _make_eq_constraints()
        #
        # def _callback_local(p):
        #     # lightweight trace
        #     obj = _objective_raw(p)
        #     res = _constraint_residuals(p)
        #     print(f"[SLSQP] f={obj:.6e}, ||c||2={np.linalg.norm(res):.3e}")
        #
        # local_result = minimize(
        #     _objective_raw,
        #     de_result.x,
        #     method='SLSQP',
        #     constraints=slsqp_constraints,
        #     options={
        #         'maxiter': 200,
        #         'ftol': 1e-12,
        #         'eps': 1e-8,
        #         'disp': False
        #     },
        #     callback=_callback_local
        # )
        #
        # # ---- Output ----------------------------------------------------------
        # final_params = local_result.x if local_result.success else de_result.x
        final_params = de_result.x
        result_curve = MotionApproximation._construct_curve_nonmonic(final_params[:32])
        final_coeffs, final_tvals = _unpack(final_params)

        print("Global (DE) objective (penalized):", de_result.fun)
        # print("Local (SLSQP) success:", local_result.success)
        print("Final raw objective:", _objective_raw(final_params))
        print("Final constraint L2 norm:",
              np.linalg.norm(_constraint_residuals(final_params)))
        print("Resulting t values:", final_tvals)

        # Return a result dict that mirrors SciPy’s style and includes both stages
        result_dict = {
            "global_method": "differential_evolution",
            "global_result": de_result,
            "local_method": "SLSQP",
            # "local_result": local_result,
            "final_params": final_params,
            "final_t_values": final_tvals,
        }

        return result_curve, result_dict
