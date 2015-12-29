#!/usr/bin/python
# -*- coding: utf-8 -*-

import operator
import math
import cmath
import fractions
import sys
import time
import re

ws_match = re.compile(
    r'[ \t\n\r]*', flags=re.VERBOSE | re.MULTILINE | re.DOTALL)
wc_match = re.compile(
    r'[_A-Za-z][_A-Za-z0-9]*', flags=re.VERBOSE | re.MULTILINE | re.DOTALL)
name_match = re.compile(
    r"'([_A-Za-z][_A-Za-z0-9]*)'", flags=re.VERBOSE | re.MULTILINE | re.DOTALL)


constants = {"i": 1j}

try:
    import numpy
    from numpy import ndarray, float64, float128, complex128, complex256, fft
    from numpy import ones, uint16
except ImportError:
    numpy = None

try:
    import scipy
    from scipy import signal
except ImportError:
    scipy = None

if numpy:
    pi = numpy.pi
    e = numpy.e
    constants.update(dict(pi=pi, e=e))
    pi128 = float128(pi) - numpy.tan(float128(pi))
    e128 = float128(e) - (numpy.log(float128(e)) - 1) / float128(e)
    constants128 = dict(pi=pi128, e=e128)

try:
    import sage.symbolic.all
    import sage.symbolic.constants
    from sage.symbolic.operators import mul_vararg, add_vararg
    from sage.functions import bessel, exp_integral, hyperbolic, hypergeometric, jacobi, log, \
        min_max, other, special, transcendental, trig
    from sage.symbolic.expression import Expression
    from sage.rings.real_mpfr import RealLiteral, RealNumber
    from sage.rings.complex_number import ComplexNumber
    from sage.rings.rational import Rational
    from sage.rings.integer import Integer
    from sage.rings.number_field.number_field_element_quadratic import NumberFieldElement_quadratic
except ImportError:
    sage = None


class RPNError(BaseException):
    pass


class WildCard(object):

    """Do not instantiate this class directly. Use wild(...).
    Only refer to this class when determining type."""

    def __init__(self, name):
        if sage and type(name) == Expression:
            name = unicode(name)
        if not re.match(r'^[_A-Za-z][_A-Za-z0-9]*$', name):
            raise TypeError, "Invalid characters."
        self.name = name

    def encode(self):
        return self.name

    def __repr__(self):
        return self.encode()

    def __str__(self):
        return self.encode()

    def __eq__(self, other):
        return type(self) == type(other) and self.name == other.name

_wildcards = {}


def wild(name):
    if name not in _wildcards:
        _wildcards[name] = WildCard(name)
    return _wildcards[name]


class VarName(WildCard):

    """Do not invoke this class directly. Use var(...).
    Only refer to this class when determining type."""

    def encode(self):
        return "'%s'" % self.name

_varnames = {}


def var(name):
    if name not in _varnames:
        _varnames[name] = VarName(name)
    return _varnames[name]


class func_wrapper(object):

    def __init__(self, func, name, nargs=1, nargsout=1, acceptskeywords=False):
        self.func = func
        self.name = name
        self.nargs = nargs
        self.acceptskeywords = acceptskeywords

    def __repr__(self):
        return "RPN function: %s" % \
            self.name.encode("utf8") if type(
                self.name) is unicode else self.name

    def __call__(self, stack):
        keywords_provided = self.acceptskeywords and \
            len(stack) and type(stack[0]) is dict
        varargs = self.nargs == "*"
        extargs = keywords_provided + varargs
        if keywords_provided:
            kwargs = stack[0]
        else:
            kwargs = {}
        if varargs:
            nargs = stack[1] if keywords_provided else stack[0]
        else:
            nargs = self.nargs
        if len(stack) < nargs + extargs:
            raise RPNError, "%s: Too few arguments." % self
        #args = reversed(stack[extargs:nargs+extargs])
        if extargs > 0:
            args = stack[nargs + extargs - 1:extargs - 1:-1]
        else:
            args = stack[nargs - 1::-1]
        if nargs == 2 and sage and numpy:
            types = map(type, args)
            # Some type coersions are in order to avoid directly operating
            # SAGE data types with NumPy data types, which seems to cause
            # memory leaks.
            if types in ([Expression, ndarray], [ndarray, Expression]):
                if types[0] is ndarray:
                    if args[0].dtype in (float128, complex256):
                        args[1] = constants128.get(str(args[1]),
                                                   constants.get(str(args[1]), args[1]))
                    else:
                        args[1] = constants.get(str(args[1]), args[1])
                    if type(args[1]) == Expression:
                        raise TypeError, "Cowardly refusing to operate a SAGE symbolic with a numpy array"
                elif types[1] is ndarray:
                    if args[1].dtype in (float128, complex256):
                        args[0] = constants128.get(str(args[0]),
                                                   constants.get(str(args[0]), args[0]))
                    else:
                        args[0] = constants.get(str(args[0]), args[0])
                    if type(args[0]) == Expression:
                        raise TypeError, "Cowardly refusing to operate a SAGE symbolic with a numpy array"
            elif types[0] == WildCard and args[0] in constants.keys():
                args[0] = constants[args[0]]
            elif types[1] == WildCard and args[1] in constants.keys():
                args[1] = constants[args[1]]
            elif types == [ComplexNumber, ndarray]:
                if args[1].dtype in (float128, complex256) or args[0].real().precision() > 53:
                    args[0] = float128(args[0])
                else:
                    args[0] = float64(args[0])
            elif types == [ndarray, ComplexNumber]:
                if args[0].dtype in (float128, complex256) or args[1].real().precision() > 53:
                    args[1] = float128(args[1])
                else:
                    args[1] = float64(args[1])
            elif types == [RealLiteral, ndarray]:
                if args[1].dtype in (float128, complex256) or args[0].real().precision() > 53:
                    args[0] = float128(args[0])
                else:
                    args[0] = float64(args[0])
            elif types == [ndarray, RealLiteral]:
                if args[0].dtype in (float128, complex256) or args[1].real().precision() > 53:
                    args[1] = float128(args[1])
                else:
                    args[1] = float64(args[1])
            elif types == [Rational, ndarray]:
                if args[1].dtype in (float128, complex256):
                    args[0] = float128(args[0])
                else:
                    args[0] = float64(args[0])
            elif types == [ndarray, Rational]:
                if args[0].dtype in (float128, complex256):
                    args[1] = float128(args[1])
                else:
                    args[1] = float64(args[1])
            elif types == [Integer, ndarray]:
                args[0] = int(args[0])
            elif types == [ndarray, Integer]:
                args[1] = int(args[1])
        result = self.func(*args, **kwargs)
        del stack[:nargs + extargs]
        stack.insert(0, result)


class StackOperator(object):

    def __init__(self, func, name):
        self.func = func
        self.name = name

    def __call__(self, stack):
        self.func(stack)


def stackop(name):
    return lambda func: StackOperator(func, name)


@stackop(u"list→")
def unpack_list(stack):
    l = stack.pop(0)
    for item in l:
        stack.insert(0, item)

rpn_funcs = {
    "+": func_wrapper(operator.add, "+", nargs=2),
    "-": func_wrapper(operator.sub, "-", nargs=2),
    "+/-": func_wrapper(operator.neg, "+/-"),
    u"⋅": func_wrapper(operator.mul, u"⋅", nargs=2),
    u"÷": func_wrapper(operator.truediv, u"÷", nargs=2),
    "^": func_wrapper(operator.pow, "^", nargs=2),
    "abs": func_wrapper(operator.abs, "abs"),
    "%": func_wrapper(operator.mod, "^", nargs=2),
    u"≤": func_wrapper(operator.le, u"≤", nargs=2),
    u"≥": func_wrapper(operator.ge, u"≥", nargs=2),
    "<": func_wrapper(operator.lt, "<", nargs=2),
    ">": func_wrapper(operator.gt, ">", nargs=2),
    "=": func_wrapper(operator.eq, "=", nargs=2),
    u"≠": func_wrapper(operator.ne, u"≠", nargs=2),
    u"∧": func_wrapper(operator.and_, u"∧", nargs=2),
    u"∨": func_wrapper(operator.or_, u"∨", nargs=2),
    u"⊻": func_wrapper(operator.xor, u"⊻", nargs=2),
    u"¬": func_wrapper(operator.not_, u"¬"),
    u"∍": func_wrapper(operator.contains, u"∍", nargs=2),
    u"∊": func_wrapper(lambda s, S: s in S, u"∊", nargs=2),
    "!": func_wrapper(math.factorial, "!"),
    u"√": func_wrapper(numpy.sqrt, u"√"),
    "max": func_wrapper(numpy.maximum, "max", nargs=2),
    "min": func_wrapper(numpy.minimum, "min", nargs=2),
    "sin": func_wrapper(numpy.sin, "sin"),
    "cos": func_wrapper(numpy.cos, "cos"),
    "tan": func_wrapper(numpy.tan, "tan"),
    "csc": func_wrapper(lambda x: numpy.sin(x) ** -1, "csc"),
    "sec": func_wrapper(lambda x: numpy.cos(x) ** -1, "sec"),
    "cot": func_wrapper(lambda x: numpy.cos(x) / numpy.sin(x), "cot"),
    "arcsin": func_wrapper(numpy.arcsin, "arcsin"),
    "arccos": func_wrapper(numpy.arccos, "arccos"),
    "arctan": func_wrapper(numpy.arctan, "arctan"),
    "arctan2": func_wrapper(numpy.arctan2, "arctan2", nargs=2),
    "sinh": func_wrapper(numpy.sinh, "sinh"),
    "cosh": func_wrapper(numpy.cosh, "cosh"),
    "tanh": func_wrapper(numpy.tanh, "tanh"),
    "arcsinh": func_wrapper(numpy.arcsinh, "arcsinh"),
    "arccosh": func_wrapper(numpy.arccosh, "arccosh"),
    "arctanh": func_wrapper(numpy.arctanh, "arctanh"),
    "exp": func_wrapper(numpy.exp, "exp"),
    "ln": func_wrapper(numpy.log, "ln"),
    "log": func_wrapper(numpy.log10, "log"),
    "lg": func_wrapper(numpy.log2, "lg"),
    u"→list": func_wrapper(lambda *args: list(args), u"→list", nargs="*"),
    u"ℱ": func_wrapper(fft.fft, u"ℱ", nargs=1, acceptskeywords=True),
    u"ℱ₂": func_wrapper(fft.fft2, u"ℱ₂", nargs=1, acceptskeywords=True),
    u"ℱₙ": func_wrapper(fft.fftn, u"ℱₙ", nargs=1, acceptskeywords=True),
    u"invℱ": func_wrapper(fft.ifft, u"invℱ", nargs=1, acceptskeywords=True),
    u"invℱ₂": func_wrapper(fft.ifft2, u"invℱ₂", nargs=1, acceptskeywords=True),
    u"invℱₙ": func_wrapper(fft.ifftn, u"invℱₙ", nargs=1, acceptskeywords=True),
    u"⋆": func_wrapper(signal.fftconvolve, u"⋆", nargs=2, acceptskeywords=True),
    u"list→": unpack_list,
}

rpn_funcs["*"] = rpn_funcs[u"⋅"]
rpn_funcs["/"] = rpn_funcs[u"÷"]
rpn_funcs["!="] = rpn_funcs[u"≠"]
rpn_funcs[">="] = rpn_funcs[u"≥"]
rpn_funcs["<="] = rpn_funcs[u"≤"]
rpn_funcs["sqrt"] = rpn_funcs[u"√"]

numtypes = (int, float, long, complex, numpy.int0, numpy.int8, numpy.int)

if sage:
    op_translate = {
        operator.add: rpn_funcs["+"],
        operator.sub: rpn_funcs["-"],
        operator.neg: rpn_funcs["+/-"],
        operator.mul: rpn_funcs[u"⋅"],
        operator.div: rpn_funcs[u"÷"],
        operator.truediv: rpn_funcs[u"÷"],
        operator.pow: rpn_funcs["^"],
        operator.mod: rpn_funcs["%"],
        operator.lt: rpn_funcs["<"],
        operator.le: rpn_funcs[u"≤"],
        operator.ge: rpn_funcs[u"≥"],
        operator.gt: rpn_funcs[">"],
        operator.ne: rpn_funcs[u"≠"],
        operator.eq: rpn_funcs["="],
        sage.functions.all.factorial: rpn_funcs["!"],
        add_vararg: rpn_funcs["+"],
        mul_vararg: rpn_funcs[u"⋅"],
        trig.sin: rpn_funcs["sin"],
        trig.cos: rpn_funcs["cos"],
        trig.tan: rpn_funcs["tan"],
        trig.asin: rpn_funcs["arcsin"],
        trig.acos: rpn_funcs["arccos"],
        trig.atan: rpn_funcs["arctan"],
        trig.atan2: rpn_funcs["arctan2"],
        trig.csc: rpn_funcs["csc"],
        trig.sec: rpn_funcs["sec"],
        trig.cot: rpn_funcs["cot"],
        hyperbolic.sinh: rpn_funcs["sinh"],
        hyperbolic.cosh: rpn_funcs["cosh"],
        hyperbolic.tanh: rpn_funcs["tanh"],
        hyperbolic.asinh: rpn_funcs["arcsinh"],
        hyperbolic.acosh: rpn_funcs["arccosh"],
        hyperbolic.atanh: rpn_funcs["arctanh"],
        operator.and_: rpn_funcs[u"∧"],
        operator.or_: rpn_funcs[u"∨"],
        operator.xor: rpn_funcs[u"⊻"],
        operator.not_: rpn_funcs[u"¬"],
        sage.functions.all.sqrt: rpn_funcs[u"√"],
        sage.functions.all.abs_symbolic: rpn_funcs["abs"],
        exp_integral.exp: rpn_funcs["exp"],
        exp_integral.log: rpn_funcs["ln"],
        sage.functions.min_max.max_symbolic: rpn_funcs["max"],
        sage.functions.min_max.min_symbolic: rpn_funcs["min"],
    }

    sage_constants = {
        sage.symbolic.constants.e: "e",
        sage.symbolic.constants.pi: u"π",
        sage.symbolic.constants.infinity: u"∞",
        sage.symbolic.constants.I: "i"
    }

    numtypes = (sage.rings.integer.Integer,
                sage.rings.real_mpfr.RealLiteral,
                sage.rings.complex_number.ComplexNumber,
                sage.rings.rational.Rational)

    def symbolic_to_rpn(symbolic):
        # if symbolic in sage_constants.keys():
        #	return RPNProgram([wild(sage_constants[symbolic])])
        if isinstance(symbolic, (int, float, complex)):
            return RPNProgram([symbolic])
        if isinstance(symbolic, Integer):
            return RPNProgram([int(symbolic)])
        elif isinstance(symbolic, (RealLiteral, RealNumber)):
            return RPNProgram([float(symbolic)])
        elif isinstance(symbolic, ComplexNumber):
            return RPNProgram([complex(symbolic)])
        elif isinstance(symbolic, Rational):
            num, den = symbolic.numerator(), symbolic.denominator()
            if den == 1:
                return RPNProgram([int(num)])
            else:
                return RPNProgram([int(num), int(den), rpn_funcs[u"÷"]])
        elif symbolic in constants.keys():
            return RPNProgram([wild(constants[symbolic])])

        # Todo: Implement symbolic matrices.

        try:
            operands = symbolic.operands()
        except:
            print symbolic
            raise
        num, den = symbolic.numerator_denominator()
        op = symbolic.operator()
        if symbolic.is_numeric() and op is None:
            if symbolic.is_real():
                symbolic = symbolic.pyobject()
                return symbolic_to_rpn(symbolic)
            elif symbolic.real():
                return RPNProgram(symbolic_to_rpn(symbolic.real()) +
                                  symbolic_to_rpn(symbolic.imag()) + [
                    1j, rpn_funcs[u"⋅"], rpn_funcs[u"+"]])
            else:
                return RPNProgram(symbolic_to_rpn(symbolic.imag()) + [
                    1j, rpn_funcs[u"⋅"]])

        # Initialize the RPN Program
        rpn = RPNProgram()

        if op in (operator.mul, mul_vararg) and den != 1:
            numrpn = symbolic_to_rpn(num)
            denrpn = symbolic_to_rpn(den)

            if numrpn[-1] == operator.neg:
                rpn.extend(numrpn[:-1])
            else:
                rpn.extend(numrpn)

            rpn.extend(denrpn)
            rpn.append(op_translate[operator.div])
            if numrpn[-1] == operator.neg:
                rpn.append(op_translate[operator.neg])
        # elif symbolic.is_integer():
            # rpn.append(int(symbolic))
        # elif symbolic.is_real():
            # rpn.append(float(symbolic))
        # elif symbolic.is_numeric():
            # rpn.append(complex(symbolic))
        elif op in (operator.add, add_vararg):
            subrpn = symbolic_to_rpn(operands[0])
            rpn.extend(subrpn)
            for term in operands[1:]:
                if term.operator() in (operator.mul, mul_vararg) and term.operands()[-1].is_real() and term.operands()[-1] < 0:
                    subrpn = symbolic_to_rpn(-term) + \
                        [op_translate[operator.neg]]
                else:
                    subrpn = symbolic_to_rpn(term)
                if subrpn[-1] == op_translate[operator.neg]:
                    rpn.extend(subrpn[:-1])
                    rpn.append(op_translate[operator.sub])
                else:
                    rpn.extend(subrpn)
                    rpn.append(op_translate[operator.add])
        elif op in (operator.mul, mul_vararg):
            if operands[-1].is_numeric():
                operands.insert(0, operands[-1])
                del operands[-1]
            isneg = False
            if operands[0] == -1:
                del operands[0]
                isneg = True
            subrpn = symbolic_to_rpn(operands[0])
            rpn.extend(subrpn)
            for factor in operands[1:]:
                subrpn = symbolic_to_rpn(factor)
                rpn.extend(subrpn)
                rpn.append(op_translate[operator.mul])
            if isneg:
                rpn.append(op_translate[operator.neg])
        elif op in (sage.functions.min_max.min_symbolic,
                    sage.functions.min_max.max_symbolic):
            subrpn = symbolic_to_rpn(operands[0])
            rpn.extend(subrpn)
            for operand in operands[1:]:
                subrpn = symbolic_to_rpn(operand)
                rpn.extend(subrpn)
                rpn.append(op_translate[op])
        elif op == operator.pow:
            if operands[1] == Rational("1/2"):
                rpn.extend(symbolic_to_rpn(operands[0]))
                rpn.append(op_translate[sage.functions.all.sqrt])
            elif operands[1] == Rational("-1/2"):
                rpn.append(int(1))
                rpn.extend(symbolic_to_rpn(operands[0]))
                rpn.append(op_translate[sage.functions.all.sqrt])
                rpn.append(op_translate[operator.div])
            elif operands[1] == -1:
                rpn.append(int(1))
                rpn.extend(symbolic_to_rpn(operands[0]))
                rpn.append(op_translate[operator.div])
            else:
                for operand in operands:
                    rpn.extend(symbolic_to_rpn(operand))
                rpn.append(op_translate[op])
        elif op:
            for operand in operands:
                rpn.extend(symbolic_to_rpn(operand))
            rpn.append(op_translate[op])
        else:
            rpn.append(wild(str(symbolic)))
        return rpn

    def conditions_to_rpn(*symbolic):
        # Takes a list of conditions, and expresses their *intersection* in RPN.
        # Unions of conditions to be implemented at a later time.
        # Lack of support for unions is overcome by specifying multiple sets
        # of conditions in PW_Function.
        rpn = symbolic_to_rpn(symbolic[0])
        for sym in symbolic[1:]:
            rpn.extend(symbolic_to_rpn(sym))
            rpn.append(op_translate[operator.and_])
        return rpn

    def piecewise_to_rpn(pieces, verbose=False):
        # 'pieces' follows format
        # [(condition1, formula1), (condition2, formula2), ...]
        # Each pair contains symbolic formulas.
        pw_func = PW_Function()
        for k, (conditions, formula) in enumerate(pieces):
            if verbose:
                print "Generating RPN for formula %d:" % (k + 1), conditions,
                t0 = time.time()
            pw_func.append(
                (conditions_to_rpn(*conditions), symbolic_to_rpn(formula)))
            if verbose:
                print "%5.2f seconds" % (time.time() - t0)
        return pw_func


class PW_Function(list):

    def __init__(self, pieces=None, **kwargs):
        # If not None, 'pieces' follows format
        # [(condition1, formula1), (condition2, formula2), ...]
        # Each pair contains RPNProgram instances.
        self.vars = kwargs
        if pieces:
            list.__init__(self, pieces)

    # To find out what case applies to **kwargs without
    # actually evaluating any formulas.
    def findcase(self, **kwargs):
        args = dict(self.vars)
        args.update(kwargs)
        mask = False
        value = None
        for k, (condition, formula) in enumerate(self):
            cond = condition(**args)
            if type(cond) is ndarray and cond.any():
                if type(mask) is ndarray:
                    value[cond & ~mask] = k
                    mask |= cond
                else:
                    mask = cond
                    # Initialize case array
                    value = -ones(cond.shape, dtype=uint16)
                    value[cond] = k
                if mask.all():
                    return value
            elif type(cond) is not ndarray and cond == True:
                if type(mask) is ndarray:
                    value[~mask] = k
                    return value
                elif mask == False:
                    return k
        return value

    def __call__(self, ret=1, **kwargs):
        cases = self.findcase(**kwargs)

        args = dict(self.vars)
        args.update(kwargs)

        if type(cases) == int:
            return self[cases][1](**args)

        value = numpy.zeros(cases.shape)
        for k, (condition, formula) in enumerate(self):
            mask = cases == k
            filtered_args = {key: val[mask]
                             if type(val) is ndarray
                             else val
                             for key, val in args.items()}
            value[mask] = formula(**filtered_args)
        return value

    def encode(self):
        formatted = []
        for (case, formula) in self:
            formatted.append("    %s : %s" %
                             (encode(case), encode(formula))
                             )
        return u"⟪\n%s\n⟫" % ",\n".join(formatted)

    def __repr__(self):
        return self.encode().encode("utf8")

    @classmethod
    def _decode(cls, string, offset):
        l = len(string)
        offset = ws_match.match(string, offset).end()
        if string[offset] != u"⟪":
            return None
        offset += 1
        result = cls([])
        while True:
            offset = ws_match.match(string, offset).end()
            if offset >= l:
                raise RPNError, "Unexpected end of string."

            if string[offset] == u"⟫":
                return result, offset + 1

            match = RPNProgram._decode(string, offset)
            if match is None:
                raise RPNError, "Expected RPN Program"
            case_rpn, offset = match

            offset = ws_match.match(string, offset).end()

            if string[offset] != ":":
                raise RPNError, "Invalid syntax: Expected colon."
            offset += 1

            for decoder in (RPNProgram._decode, cls._decode):
                match = decoder(string, offset)
                if match:
                    break
            else:
                raise RPNError, "Expected RPN Program or PW_Function."
            obj, offset = match

            result.append((case_rpn, obj))

            offset = ws_match.match(string, offset).end()
            if string[offset] == u"⟫":
                return result, offset + 1

            if string[offset] != ",":
                raise RPNError, "Invalid syntax: Expected comma or end of object delimiter."
            offset += 1


class RPNProgram(list):

    def __init__(self, rpn=None, **kwargs):
        self.vars = kwargs
        if rpn:
            if sage:
                rpn = [op_translate[token] if type(token) is not ndarray
                       and token in op_translate.keys() else token for token in rpn]
            list.__init__(self, rpn)

    def __call__(self, ret=1, stack=None, **kwargs):
        args = dict(self.vars)
        args.update(kwargs)
        if stack is None:
            stack = []
        for k, token in enumerate(self):
            if type(token) == WildCard or (sage and hasattr(token, "is_symbol") and token.is_symbol()):
                if str(token) in args.keys():
                    token = args[str(token)]
                elif str(token) in self.vars.keys():
                    token = self.vars[str(token)]
                elif sage:
                    token = sage.all.var(token)
                else:
                    stack.insert(0, token)
                    continue
            if isinstance(token, func_wrapper):
                try:
                    token(stack)
                except:
                    print >>sys.stderr, "There was a problem evaluating the RPN program at item %d (%s)." % (
                        k, encode(token))
                    print >>sys.stderr, self
                    raise
            elif isinstance(token, StackOperator):
                try:
                    token(stack)
                except:
                    print >>sys.stderr, "There was a serious problem evaluating a Stack Operator at item %d (%s)." % (
                        k, encode(token))
                    print >>sys.stderr, self
                    raise
            elif type(token) == RPNProgram:
                token(ret=0, stack=stack, **args)
            else:
                stack.insert(0, token)
        if ret == 1:
            return stack[0]
        elif ret > 1:
            return stack[:ret]

    def encode(self):
        formatted = map(encode, self)
        return u"« %s »" % " ".join(formatted)

    def __repr__(self):
        return "RPN Program: " + self.encode().encode("utf8")

    @classmethod
    def _decode(cls, string, offset=0):
        l = len(string)
        offset = ws_match.match(string, offset).end()
        if string[offset] != u"«":
            return None
        offset += 1
        result = []
        while True:
            offset = ws_match.match(string, offset).end()
            if offset >= l:
                raise RPNError, "Unexpected end of string."
            if string[offset] == u"»":
                return cls(result), offset + 1
            obj, offset = match_one(string, offset)
            result.append(obj)

int_match = re.compile(
    r'[\+\-]?\d+', flags=re.VERBOSE | re.MULTILINE | re.DOTALL)
float_match = re.compile(
    r'[\+\-]?(?:\d*\.\d+(?:[Ee][\+\-]?[\d]+)?|\d+(?:[Ee][\+\-]?[\d]+))', flags=re.VERBOSE | re.MULTILINE | re.DOTALL)
imaginary_match = re.compile(
    r'([\+\-]?(?:(?:\d*\.[\d]+|\d+)(?:[Ee][\+\-]?[\d]+)?)|[\+\-]?)i', flags=re.VERBOSE | re.MULTILINE | re.DOTALL)
complex_match = re.compile(
    r'([\+\-]?(?:\d*\.\d+|\d+)(?:[Ee][\+\-]?[\d]+)?)([\+\-](?:\d*\.\d+|\d+)?(?:[Ee][\+\-]?[\d]+)?)i', flags=re.VERBOSE | re.MULTILINE | re.DOTALL)


def num_decode(string, offset=0):
    offset = ws_match.match(string, offset).end()

    match = imaginary_match.match(string, offset)
    if match:
        im = match.groups()[0]
        if im == "" or im == "+":
            return 1j, match.end()
        elif im == "-":
            return -1j, match.end()
        else:
            return float(im) * 1j, match.end()

    match = complex_match.match(string, offset)
    if match:
        re, im = match.groups()
        re = float(re)
        if im == "" or im == "+":
            im = 1
        elif im == "-":
            im = -1
        else:
            im = float(im)

        return re + im * 1j, match.end()

    match = float_match.match(string, offset)
    if match:
        return float(match.group()), match.end()

    match = int_match.match(string, offset)
    if match:
        return int(match.group()), match.end()

string_match = re.compile(
    r'"(?:[^\\"]|\\(?:["abfrntv\\]|[0-8]{1,3}|u\d{1,4}|U\d{1,8}|x[0-9A-Fa-f]{1,8}))+"', flags=re.VERBOSE | re.MULTILINE | re.DOTALL)


def string_decode(string, offset=0):
    offset = ws_match.match(string, offset).end()

    match = string_match.match(string, offset)
    if match is None:
        return None
    entities = re.findall(
        r'([^\\"])|\\(?:(["abfrntv\\])|([0-8]{1,3})|x([0-9A-Fa-f]{1,8})|u(\d{1,4})|U(\d{1,8}))', match.group()[1:-1])
    result = ""
    for literal, escape, octal, hexadecimal, utf8, utf16 in entities:
        if literal:
            result += literal
        elif escape == "a":
            result += "\a"
        elif escape == "b":
            result += "\b"
        elif escape == "f":
            result += "\f"
        elif escape == "r":
            result += "\r"
        elif escape == "n":
            result += "\n"
        elif escape == "n":
            result += "\n"
        elif escape == "t":
            result += "\t"
        elif escape == "\\":
            result += "\\"
        elif octal:
            n = int(octal, 8)
            if n >= 128:
                result += unichr(n)
            else:
                result += chr(n)
        elif hexidecimal:
            n = int(hexidecimal, 16)
            if n >= 128:
                result += unichr(n)
            else:
                result += chr(n)
        elif utf8:
            n = int(utf8, 16)
            result += unichr(n)
        elif utf16:
            n = int(utf16, 16)
            result += unichr(n)
    return result, match.end()


def wildcard_decode(string, offset=0):
    offset = ws_match.match(string, offset).end()
    match = wc_match.match(string, offset)
    if match is None:
        return None
    elif match.group() == "True":
        return True, match.end()
    elif match.group() == "False":
        return False, match.end()
    elif match.group() in rpn_funcs.keys():
        return rpn_funcs[match.group()], match.end()
    return wild(match.group()), match.end()


def operator_decode(string, offset=0):
    ops = [re.escape(key)
           for key in rpn_funcs.keys() if not name_match.match(key)]
    ops.sort(key=len, reverse=True)
    op_match = re.compile("|".join(ops),
                          flags=re.VERBOSE | re.MULTILINE | re.DOTALL)
    offset = ws_match.match(string, offset).end()
    match = op_match.match(string, offset)
    if match is None:
        return None
    return rpn_funcs[match.group()], match.end()


def varname_decode(string, offset=0):
    offset = ws_match.match(string, offset).end()
    match = name_match.match(string, offset)
    if match is None:
        return None
    return var(match.groups()[0]), match.end()


def list_decode(string, offset=0):
    l = len(string)
    offset = ws_match.match(string, offset).end()
    if string[offset] != "[":
        return None
    offset += 1
    result = []
    while True:
        offset = ws_match.match(string, offset).end()
        if offset >= l:
            raise RPNError, "Unexpected end of string."

        if string[offset] == "]":
            return result, offset + 1

        obj, offset = match_one(string, offset)
        result.append(obj)

        offset = ws_match.match(string, offset).end()
        if offset >= l:
            raise RPNError, "Unexpected end of string."

        if string[offset] == "]":
            return result, offset + 1

        if string[offset] != ",":
            raise RPNError, "Invalid syntax: Expected comma or end of object delimiter."
        offset += 1


def array_decode(string, offset=0):
    l = len(string)
    offset = ws_match.match(string, offset).end()
    if string[offset] != u"⟦":
        return None
    offset += 1
    result = []
    while True:
        offset = ws_match.match(string, offset).end()
        if offset >= l:
            raise RPNError, "Unexpected end of string."

        if string[offset] == u"⟧":
            return numpy.array(result), offset + 1

        obj, offset = match_one(string, offset)
        result.append(obj)

        offset = ws_match.match(string, offset).end()
        if offset >= l:
            raise RPNError, "Unexpected end of string."


def set_decode(string, offset=0):
    l = len(string)
    offset = ws_match.match(string, offset).end()
    if string[offset] != u"⦃":
        return None
    offset += 1
    result = []
    while True:
        offset = ws_match.match(string, offset).end()
        if offset >= l:
            raise RPNError, "Unexpected end of string."

        if string[offset] == u"⦄":
            return set(result), offset + 1

        obj, offset = match_one(string, offset)
        result.append(obj)

        offset = ws_match.match(string, offset).end()
        if offset >= l:
            raise RPNError, "Unexpected end of string."

        if string[offset] == u"⦄":
            return set(result), offset + 1

        if string[offset] != ",":
            raise RPNError, "Invalid syntax: Expected comma or end of object delimiter."
        offset += 1


def dict_decode(string, offset):
    l = len(string)
    offset = ws_match.match(string, offset).end()
    if string[offset] != "{":
        return None
    offset += 1
    result = {}
    while True:
        offset = ws_match.match(string, offset).end()
        if offset >= l:
            raise RPNError, "Unexpected end of string."

        if string[offset] == "}":
            return result, offset + 1

        match = string_decode(string, offset)
        if match is None:
            raise RPNError, "Expected string."
        key, offset = match

        offset = ws_match.match(string, offset).end()

        if string[offset] != ":":
            raise RPNError, "Invalid syntax: Expected colon."
        offset += 1

        match = match_one(string, offset)
        if match is None:
            raise RPNError, "Invalid syntax???"
        value, offset = match

        result[key] = value

        offset = ws_match.match(string, offset).end()
        if string[offset] == "}":
            return result, offset + 1

        if string[offset] != ",":
            raise RPNError, "Invalid syntax: Expected comma or end of object delimiter."
        offset += 1


decoders = (RPNProgram._decode, PW_Function._decode, list_decode, dict_decode, set_decode,
            array_decode, num_decode, operator_decode, string_decode, wildcard_decode, varname_decode)


def match_one(string, offset=0, end_delimiter=None):
    offset = ws_match.match(string, offset).end()
    result = []
    for decode in decoders:
        result = decode(string, offset)
        if hasattr(result, "group") and hasattr(result, "end"):
            return result.group(), result.end()
        elif result is not None:
            return result
    else:
        msg = "Unable to decode at offset %d. (...%s...)" % \
            (offset, string[max(0, offset - 8):offset + 8])
        if type(msg) is unicode:
            msg = msg.encode("utf8")
        raise RPNError, msg


def encode(token):
    if isinstance(token, (str, unicode)):
        chars = []
        for alpha in token:
            if alpha == "\\":
                chars.append("\\\\")
            elif alpha == "\"":
                chars.append('\\"')
            elif alpha == "\r":
                chars.append('\\r')
            elif alpha == "\t":
                chars.append('\\t')
            elif alpha == "\n":
                chars.append('\\n')
            elif alpha == "\a":
                chars.append('\\a')
            elif alpha == "\b":
                chars.append('\\b')
            elif alpha == "\f":
                chars.append('\\f')
            elif alpha == "\v":
                chars.append('\\v')
            else:
                chars.append(alpha)
        return '"%s"' % "".join(chars)
    elif isinstance(token, (func_wrapper, StackOperator)):
        return token.name
    elif isinstance(token, (int, long, float, float64)):
        return str(token)
    elif isinstance(token, (complex, complex128)):
        re, im = token.real, token.imag
        if re % 1 == 0:
            re = int(re)
        if im % 1 == 0:
            im = int(im)
        if re == 0 and im == 0:
            return "0"
        elif re == 0:
            if im == 1:
                return "i"
            elif im == -1:
                return "-i"
            else:
                return "%si" % im
        elif im == 0:
            return encode(re)
        else:
            if im > 0:
                return "%s+%si" % (encode(re), encode(im))
            else:
                return "%s-%si" % (encode(re), encode(-im))
    elif type(token) == dict:
        formatted = ["%s : %s" % (encode(key), encode(val))
                     for key, val in token.items()]
        return "{ %s }" % ", ".join(formatted)
    elif type(token) == list:
        formatted = map(encode, token)
        return "[ %s ]" % ", ".join(formatted)
    elif type(token) == set:
        formatted = map(encode, token)
        return u"⦃ %s ⦄" % ", ".join(formatted)
    elif type(token) == ndarray:
        formatted = map(encode, token)
        return u"⟦ %s ⟧" % " ".join(formatted)
    elif hasattr(token, "encode") and callable(token.encode):
        return token.encode()


def decode(string, offset=0, end_delimiter=None):
    l = len(string)
    result = []
    while True:
        obj, offset = match_one(string, offset)
        result.append(obj)
        offset = ws_match.match(string, offset).end()
        if offset >= l:
            break
    if len(result) > 1:
        return tuple(result)
    return result[0]
