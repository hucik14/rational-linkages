import biquaternion_py as bq
import sympy as sp

from FactorizationProvider import FactorizationProvider
from rational_linkages import DualQuaternion
from RationalDualQuaternion import RationalDualQuaternion
from rational_linkages import MotionFactorization
from RationalMechanism import RationalMechanism
from Plotter import Plotter


# Setting up the symbols
t = sp.Symbol("t")

# Given a Curve as a polynomial in the dual quaternions
# we first need to find the irreducible factors of the normpolynomial

# For this we construct a polynomial from linear factors (t-h)

h1bq = bq.BiQuaternion(
    [-1 / 4, 13 / 5, -213 / 5, -68 / 15, 0, -52 / 3, -28 / 15, 38 / 5]
)
h2bq = bq.BiQuaternion(
    [-3 / 10, 833 / 240, -451 / 160, 19 / 24, 0, -427 / 480, -1609 / 720, -1217 / 300]
)
h3bq = bq.BiQuaternion(
    [9 / 4, -96 / 385, -3 / 11, 12 / 121, 0, -9 / 22, 18 / 77, -27 / 70]
)

# h1 = bq.rand_line()
# h2 = bq.rand_line()
# h3 = bq.rand_line()

poly = bq.Poly((t - h1bq) * (t - h2bq) * (t - h3bq), t)

# Next we calculate the norm polynomial. To avoid numerical problems, we extract
# the scalar part, since the norm should be purely real anyhow.

norm_poly = poly.norm()
norm_poly = bq.Poly(norm_poly.poly.scal, *norm_poly.indets)

# From this we can calculate the irreducible factors, that then determine the different
# factorizations

_, factors = bq.irreducible_factors(norm_poly)

# The different permutations of the irreducible factors then generate the different
# factorizations of the motion.

factorization1 = bq.factorize_from_list(poly, factors)
factorization2 = bq.factorize_from_list(poly, [factors[2], factors[1], factors[0]])
# factorization1[0].poly.coeffs

# We can now calculate the direct kinematics of the motion

rdq = [sp.Rational(-1 / 4), sp.Rational(13 / 5), sp.Rational(-213 / 5), sp.Rational(-68/ 15), 0, sp.Rational(-52 / 3), sp.Rational(-28 / 15), sp.Rational(38 / 5)]
h1r = RationalDualQuaternion(rdq)
rdq = [sp.Rational(-3 / 10), sp.Rational(833 / 240), sp.Rational(-451 / 160), sp.Rational(19 / 24), 0, sp.Rational(-427 / 480), sp.Rational(-1609 / 720), sp.Rational(-1217 / 300)]
h2r = RationalDualQuaternion(rdq)
rdq = [sp.Rational(9 / 4), sp.Rational(-96 / 385), sp.Rational(-3 / 11), sp.Rational(12 / 121), 0, sp.Rational(-9 / 22), sp.Rational(18 / 77), sp.Rational(-27 / 70)]
h3r = RationalDualQuaternion(rdq)

h1 = DualQuaternion([-1 / 4, 13 / 5, -213 / 5, -68 / 15, 0, -52 / 3,-28 / 15, 38 / 5], is_rotation=True)
h2 = DualQuaternion([-3 / 10, 833 / 240, -451 / 160, 19 / 24, 0, -427 / 480, -1609 / 720, -1217 / 300], is_rotation=True)
h3 = DualQuaternion([9 / 4, -96 / 385, -3 / 11, 12 / 121, 0, -9 / 22, 18 / 77, -27 / 70], is_rotation=True)

from RationalCurve import RationalCurve

my_f = MotionFactorization([h1r, h2r, h3r])
factorizator = FactorizationProvider()
factorize = factorizator.factorize_for_motion_factorization(my_f)

m = RationalMechanism(factorize)
plt = Plotter(interactive=True, steps=200)
plt.plot(m)

h12 = DualQuaternion([-1 / 4, 13 / 5, -213 / 5, -68 / 15, 0, -52 / 3, -28 / 15,
                      38 / 5], is_rotation=True)
h22 = DualQuaternion([-3 / 10, 833 / 240, -451 / 160, 19 / 24, 0, -427 / 480,
                      -1609 / 720, -1217 / 300], is_rotation=True)
h32 = DualQuaternion([9 / 4, -96 / 385, -3 / 11, 12 / 121, 0, -9 / 22, 18 / 77,
                      -27 / 70], is_rotation=True)
k12 = DualQuaternion([9 / 4, -353293129020116088274366524 /
                      2046064697881244081606857985, 71057440088136127923615537 / 292294956840177725943836855,-770841181127162033209449696 / 3215244525241954985382205405,0,1285925291840670577611498917452530201753024451488061/ 3106776065243685802900593279235046876397736685683310,369908388252939453727221870302615264755351380922518/ 10873716228352900310152076477322664067392078399891585,-521169721602430808498118610255124429182751294514817/ 1977039314245981874573104814058666194071286981798470],is_rotation=True)
k22 = DualQuaternion([-3 / 10, -1584906194063534950985110611269499455843766362303 /
                      748589572249221191178253511800771099480289042160, -1920849060977583492916015219909062271715671063459 / 499059714832814127452169007867180732986859361440,-86070495207833111808000603712957119046286563789/ 74858957224922119117825351180077109948028904216,0,-1493945822997033838889866124648869571241092592593232647136782680036917092967587145383237343823/ 933977246133786589561715243908633265522422939574517641472244621670941695011531052831617095776,-4383675369926674968359098051148186009275012265484280981512415092271834593351409097148552548873/ 7004829346003399421712864329314749491418172046808882311041834662532062712586482896237128218320,4903731966877692899272930689015951377076396084948182642657351735217230221511376454058771819441/ 972892964722694364126786712404826318252523895390122543200254814240564265637011513366267808100],is_rotation=True)
k32 = DualQuaternion([-1 / 4, 71409809286507251213549 / 8803698436759977863535,
                      -24700620508978006448565 / 586913229117331857569,-1322302914772604754264 / 586913229117331857569,0,-6009699344068792249739386119282475891686296/ 344467138512933679794546525402592102589761,-50778766896365492707380727887969151002044936/ 15501021233082015590754593643116644616539245,-8366061133662094762021335484888717836156016/ 5167007077694005196918197881038881538846415], is_rotation=True)

f12 = MotionFactorization([h12, h22, h32])
f22 = MotionFactorization([k12, k22, k32])

m2 = RationalMechanism([f12, f22])
p2 = Plotter(interactive=True, steps=2000)
p2.plot(m2)

p = Plotter(interactive=False, steps=200)
p.plot(f12, t=2)
p.plot(f22, t=2)
p.plot(m, t=2)
