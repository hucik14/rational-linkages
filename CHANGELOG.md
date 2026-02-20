# Changelog

## 2.5.0 (2026-02-20)

### added (11 changes)

- [line model can be exported as STEP solid using build123d](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/605cc415)
- [trimesh and manifold3d are new optional dependencies](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/83513f1b)
- [docs updated with STL description](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/06d1e33b)
- [tool link and frame can be meshed](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/41521fe7)
- [single STL mesh can be exported to represent whole linkage at home configuration](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/575a6aad)
- [RationalMechanism.get_design prints also joint points in world frame](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/1babb83a)
- [motion designer example script](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/649b082a)
- [extended dot product for DQ](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/dccd8fe2)
- [generate random DQ on SQ with integer element values](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/ade428f9)
- [changelog description accepts lowercase](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/0d0ac452)
- [builds also ARM wheels on Github](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/1caeea03)


### removed (1 change)

- [hoverxref sphinx addon removed as it was deprecated, causing builds to fail](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/79b6d462)


### changed (1 change)

- [motion interpolation methods follow the ARK paper notation](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/9c72c0e0)


### fixed (2 changes)

- [rendering errors in new PyqtGraph version, quickfix to stay at version 0.13](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/d72c1a68)
- [Mechanism from MotionDesigner didn't respect white background](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/75945114)



## 2.4.0 (2025-11-21)

### added (5 changes)

- [linux build runner changed to Github Actions; docs update](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/46f4c2a5)
- [possiblity to run Motion Designer with GUI options](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/a52e9908)
- [method for plane-line intersection point](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/d0b6408e)
- [size of poloted points can be altered](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/2f7ff53b)


### removed (1 change)

- [gmpy2 removed from dependencies as it causes issues on ARM devices](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/5be62c40)


### changed (1 change)

- [update to installation instructions](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/a8bb04ca)


### fixed (3 changes)

- [mocking utils in docs](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/e9958d53)
- [utils were missing in docs](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/6920e442)
- [parsing angle error](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/17d6d355)


## 2.2.3 (2025-09-17)

### added (1 change)

- [MyBinder.org environment setup](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/82ee6690)


### fixed (1 change)

- [jupyter notebooks import error due to Qt](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/e29425c6)


## 2.2.0 (2025-09-17)

### added (5 changes)

- [Windows ARM support, dependencies clean-up and description changed](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/a826d9f9)
- [tutorial with motion recovery](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/aff296b6)
- [new tutorial on obtaining rational axes](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/6f8b6e60)
- [new function for creating rational plucker line vector](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/1e9b039a)
- [new function for creating rational transformation matrices](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/50937f45)


### changed (2 changes)

- [scipy no longer required dependency; matplotlib gets installed on linux](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/81dc09f4)
- [GUI dependencies should no longer cause error on linux while importing](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/69c1c99d)


### fixed (1 change)

- [debug of rational numbers initialization](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/494da866)


## 2.1.0 (2025-08-13)

### added (6 changes)

- [docstring update](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/0b0d2d63)
- [ik evaluation is lambified](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/72cd77df)
- [ik method can return t param](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/3d630e63)
- [optimization for a curved link](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/b25d4d09)
- [print design params with high precision](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/4b00414b)
- [plotting of GL curves](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/1fc6301a)


### fixed (1 change)

- [sympy's zero was not handled](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/fc02caf7)


## 2.0.0 (2025-06-11)

### added (48 changes)

- [for plotting, the mechanisms base can be altered](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/13b858c4)
- [motion designer added to docs](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/7ea5dcf3)
- [gif added to readme and index](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/76ef90b0)
- [dk, ik, and motion plannings docs](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/151eda7a)
- [create PNG animation of rotation](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/7cf377e7)
- [new simple rational curve example](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/23f6489a)
- [tutorial for physical construction of mechanism](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/4837ea1f)
- [generation of interpolated 6r](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/9853ccba)
- [MotionDesigner point values can be edited by textbox](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/5470f622)
- [MotionDesigner point values can be edited by textbox](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/e8430702)
- [MotionDesigner now shows values of control points](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/ca347381)
- [error handling of mechanism creation](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/07906b24)
- [approximation for points draft (not working); force study quadric optimization](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/8eaa5832)
- [Quaternons and DualQuaternions elements can be set directly](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/c179a726)
- [Plotter.py returns a class instance based on given parameters](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/c939847e)
- [class for Gauss-Legendre curves](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/9176c76d)
- [reset of LineSegment counter at update](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/c4630718)
- [LineSegment class has registry, segments are created in circular order of two factorizations](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/6767e89c)
- [LineSegment new arg: default (at inf) line coeffs](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/8785ab71)
- [new method for obtaining a relative motion between two links-joints](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/6df4c29f)
- [MotionsDesigner synthesizes and plots mechanism right away](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/bba4fafa)
- [k-dq x3 calculation in Rust](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/5a7293f5)
- [rust backend - library import and test methods](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/464eca5f)
- [cubic interpolation for both families](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/1fee76fd)
- [swap in visualization of both families solutions](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/832e6fb3)
- [heavy calculations separate lib](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/3c71bb5c)
- [new logo](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/b7c4b7e2)
- [save PNG figure](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/1617916e)
- [pyqtgraph white background option](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/af662eb1)
- [new dependencies](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/e4ee7c64)
- [new utility to extract coefficients from sympy expression](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/4e3dadf0)
- [Cubic interpolation accepts lambda parameter](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/1ca2e2b8)
- [Motion Designer for Cubic curve - experimental](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/5e62490c)
- [normalized plane default plot](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/66bc6c7d)
- [interactive plotting with Pyqtgraph](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/5d52f043)
- [construction of TransfMatrix from single rotation](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/d2afa3cb)
- [motion designer for quadratic curve from poses](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/7dd1d8d9)
- [numerical interpolation of quadratic curve](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/14edeec0)
- [MotionDesigner for quadratic curves via 5 points](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/80ad7653)
- [adding motion designer, not yet working](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/13df3c58)
- [interpolation of Bezier curves numerically](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/3408016c)
- [Vispy plotter backend](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/6c0f1cb6)
- [PointHomogeneous can be multiplied or divided by scalar](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/267aaa92)
- [obtain point orbits using Welzl's algroithm](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/1f3021de)
- [CollisionAnalyser.py takes over calculation for segments orbits](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/4df5d2f4)
- [BezierSegmentControlPoints class avoids init of sympy objects for splitting](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/b84373c0)
- [RationalCurve has the "metric" attribute](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/28390132)
- [added metric attribute RationalMechanism.py](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/fb2aecee)


### removed (2 changes)

- [first point/pose cannot be modified (identity)](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/29a67ea0)
- [Vispy backend removed](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/1127babb)


### changed (8 changes)

- [plotting docs updated for Qt6 backend](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/70b62dec)
- [PyQt6 update for Matplotlib backend, saving PyQtgraph with text overlay](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/1f14b9ea)
- [update of models for latest RL version](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/949f28f0)
- [get_design returns correct points which do not take into account Onshape's design params](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/bf0de27c)
- [__init__.py cleanup](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/74e706f5)
- [new class for Plotting, handling PyQtGraph as default, Matplotlib as secondary backend that is not required](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/5226e0eb)
- [changes in dependencies, exudyn option](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/e93f0754)
- [new default poses for quadratic MD, added test](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/f48dd268)


### fixed (10 changes)

- [get_design returns correct points; newly input is in meters by default](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/87e0c6fe)
- [labeling erased previous render, Gl not processed well](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/427024cf)
- [debug in case of difficult interpolation - perfomed numerically](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/d02844ed)
- [Bennett DH docs fix](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/58ffdf7d)
- [debug example](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/52f5c095)
- [vector check not strict enough](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/50d5bb2e)
- [debug - poses append; solve for t optimized](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/c7e022ef)
- [macos fix for segmentation - too small init windows](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/313fc2fb)
- [DH params were returning wrong d for base](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/13ae0ff1)
- [debug when normal z-coord was 0, cleanup](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/8d9e55c6)



## 1.10.2 (2025-04-11)

### added (1 change)

- [new logo](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/5fe45fad)


### fixed (2 changes)

- [vector check not strict enough](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/179c4868)
- [DH params were returning wrong d for base](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/865d8f48)



## 1.10.1 (2025-03-03)

### added (1 change)

- [github mirroring, notebooks update](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/6eb348b3)

### fixed (1 change)

- [bug: collision searched failed in case of joint-joint collisions](https://git.uibk.ac.at/geometrie-vermessung/rational-linkages/-/commit/9de82417)


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

