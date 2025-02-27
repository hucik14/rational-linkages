# Changelog

## 1.10.0 (2025-02-03)

### added (12 changes)

- [new Motion Approximation method](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/5b06135e)
- [tests and debug of new methods](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/9090b931)
- [constructors from 3 points, line and point, intersection with other plane](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/d691d967)
- [intersection of lines and planes, cleanup](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/42d41b9d)
- [New NormalizedPlane.py class](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/38355558)
- [numpy-stl added to opt dependencies](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/e37992e6)
- [get_design of RationalMechanism.py returns also actual connection points](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/8fa7365b)
- [print params for onshape new models](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/dda41e25)
- [Synthesis of snapping mechanism for two poses - Wunderlich construction](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/8ce2225d)
- [plotter adds methods triggering controls visibility, saving via command line](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/6d4fe794)
- [plotter adds methods for plotting planes, axis (quiver) between points, line segments between list of points](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/63a6111b)
- [create StaticMechanism.py from algebraic equations (i,j,k,eps)](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/8d57ee1e)


### removed (2 changes)

- [code cleanup, removed ButtonID](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/5a8630fd)
- [examples cleanup](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/4fc947d8)


### changed (2 changes)

- [plotting examples cleanup](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/2f929382)
- [precise Study Quadric check using numpy Polynomials](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/075a36d1)


## 1.9.0 (2024-11-20)

### added (5 changes)

- [rational curve can be numerically checked if it is real motion](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/507db086)
- [rational curve can be created from two quaternions](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/5b591381)
- [interpolation of 7 3D points using cubic curve](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/9a9d2efb)
- [new method for interpolating 5 points with quadratic curve](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/b86d041f)
- [testing for Python 3.13](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/a7a5e1fd)


### removed (2 changes)

- [warning for installation of Exudyn removed](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/d749f620)
- [removed unnecessary warning for RR domain](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/0931bb90)


### changed (3 changes)

- [tests run for python 3.13](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/d83b150a)
- [legend is no longer on plot by default, arg needed](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/f09f4944)
- [ik update, robust run automatically](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/9802a9b1)

## 1.8.0 (2024-08-28)

### added (11 changes)

- [updates to docs, jupyter notebook example on DK IK](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/5d278a60)
- [generate csv with desired trajectory](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/684d6c5d)
- [method to normalize DQ by first element](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/764e1aaa)
- [traj planning for smooth end-effector velocity](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/5026d78d)
- [p2p trajectory planning in joint space](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/e95cfd5e)
- [finding t for splitting curve in equal segments](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/3c1df8ce)
- [numerical ik is for curve and curve.inverse due to numerical stability](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/9a5051d4)
- [calculate numerically inverse kinematics of linkage](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/33cbcdbd)
- [calculate forward/direct kinematics of linkage](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/599609c8)
- [animate mech passing through given angles](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/beceb7f6)
- [util to calculate sum of squares in the list](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/9e6480fa)


### fixed (5 changes)

- [ik debug, didn't take into account tool](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/1c00eb59)
- [reparameterization of theta driving angle to t must use pluecker norm](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/7bca690f)
- [negative angle plotting debug](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/36d19a66)
- [ik solver issues with convergence, added normalization](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/c890e818)
- [bug for negative angle](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/1f1f3337)

## 1.7.0 (2024-07-11)

### added (2 changes)

- [check on Study quadric can be optionally approximate (numerics)](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/8a54df3f)
- [motion factorization has option to return rational values](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/6a5a474b)


### changed (2 changes)

- [the RationalCurve attribudes are created on call using properties](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/bad43e6e)
- [docs: update to DQ and Study kinematics](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/a2a0a150)


### fixed (1 change)

- [sympy 1.13 poly uncompatibility fix](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/f1064b16)


## 1.6.0 (2024-06-27)

### added (5 changes)

- [semi-automated changelog generation](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/1cf44d73)
- [Added gmpy2 dependency for faster Sympy](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/3f63ffa6)
- [docs - mouse hovering tooltip](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/c71c2ac9)
- [option to update physical design of a linkage based on physical joint size](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/dc6b45ef)
- [perform and save animation](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/af2e5ce7)


### changed (2 changes)

- [faster collision check (forgotten simplify deleted)](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/6155127c)
- [MF's linkage is now a property that is obtained when needed](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/5750b834)


### fixed (4 changes)

- [autogenerate changelog.rst via pandoc](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/9fc2ff7a)
- [gmpy2 is fails to build for Python 3.12](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/c5ea1cdc)
- [if Z axes are parallel, take footpoint; extra frame not added anymore; tolerance for checking 2 parallel lines lowered](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/b62ede7b)
- [during DH determination were added axis to screw axes](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/324dd211)

