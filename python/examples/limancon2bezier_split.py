import sympy as sp

from rational_linkages import Plotter, BezierSegment, RationalCurve

t = sp.Symbol("t")

# Limancon of Pascal
a = 1
b = 0.5
l0 = sp.Poly((1 + t**2) ** 2, t)
l1 = sp.Poly(b * (1 - t**2) * (1 + t**2) + a * (1 - t**2) ** 2, t)
l2 = sp.Poly(2 * b * t * (1 + t**2) + 2 * a * t * (1 - t**2), t)

limancon = RationalCurve([l0, l1, l2, l0])

limancon_inv = limancon.inverse_curve()

bezier = BezierSegment(limancon.curve2bezier_control_points(reparametrization=True), t_param=(False, [-1, 1]))
bezier_inv = BezierSegment(limancon_inv.curve2bezier_control_points(reparametrization=True), t_param=(True, [-1, 1]))
b_left, b_right = bezier.split_de_casteljau(0.4)
b_left_inv, b_right_inv = bezier_inv.split_de_casteljau()

p = Plotter(interactive=False, arrows_length=0.5, joint_sliders_lim=2, steps=200)
bezier_segments = [b_left, b_right, b_left_inv, b_right_inv]

for bezier_curve in bezier_segments:
    p.plot(bezier_curve.curve, interval=(-1, 1), plot_control_points=True)
    p.plot(bezier_curve.ball)

p.show()
