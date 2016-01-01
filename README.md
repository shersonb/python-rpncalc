# python-rpncalc
An implementation of a Reverse Polish Notation calculator for Python

The original motivation of this module was to be able to create a
Reverse Polish Notation representation of a SAGE symbolic function
for use in a non-SAGE Python environment. At the time of this writing,
this is a *very* early work, and it is expected that there is lots of
room for development on this module. In particular, support for sympy
is planned for the future.

At the moment, the module is capable of encoding (and decoding)
formulas and piecewise-defined formulas into (and from) strings.

The RPNProgram and PW_Function classes create callable objects that
accept keyword arguments for use in place of variables found in the
RPN code.

For example:

f = rpncalc.decode(u"« x 2 ^ y 2 ^ + sqrt »")

returns an RPNProgram object, equivalent to returning
sqrt(x**2 + y**2) when x and y are passed as keyword arguments to f.

Note that this syntax is inspired by the HP48/49/50 line of RPN
calculators.

The format of a piecewise-defined function is similar to a dict, and
takes the form:

"""
⟪
    « case1 » : « formula1 »,
    « case2 » : « formula2 »,
    ...
⟫
"""

Each « case# » is an RPNProgram object that is expected to return either a
BooleanType or a boolean numpy array. In the case of overlapping cases,
the cases listed first take priority. If an input has no applicable cases,
the PW_Function object will return 0.

Each « formula# » is either an RPNProgram or a PW_Function object.

For example, here is an implementation of the absolute value function:

>>> abs = rpncalc.decode(u"""⟪
...     « x 0 ≥ » : « x »,
...     « x 0 ≤ » : « x +/- »
... ⟫""")
...
>>> abs(x=3)
3
>>> abs(x=-4)
4

The PW_Function also provides a findcase method, which allows one to find
the case applicable to the provided keyword arguments. Its return value will
either be an integer, greater than equal to -1, with -1 indicating no case
found, and other values being the index, or a numpy array of such values.

