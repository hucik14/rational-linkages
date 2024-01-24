import matplotlib.pyplot as plt
import sympy as sp
from rational_linkages import RationalBezier, RationalCurve, Plotter

t = sp.Symbol("t")

# Limancon of Pascal
a = 1
b = 0.5
l0 = sp.Poly((1 + t**2) ** 2, t)
l1 = sp.Poly(b * (1 - t**2) * (1 + t**2) + a * (1 - t**2) ** 2, t)
l2 = sp.Poly(2 * b * t * (1 + t**2) + 2 * a * t * (1 - t**2), t)

limancon = RationalCurve([l0, l1, l2, l0])

limancon_inv = limancon.inverse_curve()

bezier = RationalBezier(limancon.curve2bezier(reparametrization=True))
bezier_inv = RationalBezier(limancon_inv.curve2bezier(reparametrization=True))

# # prepare axes to plot
# ax = limancon.plot((-1, 1), line_style="yellow")
#
# # optional argument 'ax = ax' is passing axes from previous plot to the next one
# #ax = limancon_inv.plot((-1, 1), ax=ax, line_style="yellow")
#
# if bezier.check_for_control_points_at_infinity():
#     left, right = bezier.split_de_casteljau(
#         0.45
#     )  # optional ratio (0.45) is ratio of the parts, ie. one blue ball will be bigger
#
#     ax = left.plot((-1, 1), ax=ax, line_style="b--")
#     ax = left.ball.plot_ball(ax=ax, color="blue")
#     ax = right.plot((-1, 1), ax=ax, line_style="b--")
#     ax = right.ball.plot_ball(ax=ax, color="blue")
# else:
#     ax = bezier.plot((-1, 1), ax=ax, line_style="r--")
#     ax = bezier.ball.plot_ball(ax=ax)
#
# if bezier_inv.check_for_control_points_at_infinity():
#     left_inv, right_inv = bezier_inv.split_de_casteljau()
#
#     ax = left_inv.plot((-1, 1), ax=ax, line_style="g--")
#     ax = left_inv.ball.plot_ball(ax=ax, color="green")
#     ax = right_inv.plot((-1, 1), ax=ax, line_style="g--")
#     ax = right_inv.ball.plot_ball(ax=ax, color="green")
# else:
#     ax = bezier_inv.plot((-1, 1), ax=ax, line_style="r--")
#     ax = bezier_inv.ball.plot_ball(ax=ax)
#
#
# # Set labels and title for the plot
# ax.set_xlabel("X-axis")
# ax.set_ylabel("Y-axis")
# ax.set_zlabel("Z-axis")
#
# # set aspect ratio
# ax.set_aspect("equal")
# plt.show()
