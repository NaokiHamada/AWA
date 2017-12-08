# -*- coding: utf-8 -*-
'''
AWA - Adaptive Weighted Aggregation

Usage of AWA
------------
>>> async def f(X):
...     return sum(x**2 for x in X), sum((x-1)**2 for x in X)
>>> x0 = [[0, 0],
...       [1, 1]]
>>> awa = AWA(f, x0)
>>> result = awa[(0, 1)]  # get the results for address (0, 1) or run the search
>>> np.allclose(result.x, [1, 1], atol=0.1)  # optimized x
True
>>> np.allclose(awa[(0, 1)].x, [1, 1], atol=0.1)  # results are cached
True
>>> curio.run(awa.search(Address([0, 1])))  # rerun explicitly

Usage of addresses
------------------
>>> a = Address([1, 1])  # make an address (1, 1)
>>> print(a)
(1, 1)

>>> b = Address([2, 2])  # address is canonicalized
>>> print(b)
(1, 1)

>>> a == b  # two addresses are equal when their canonical forms are the same
True

>>> sp = AddressSpace(2, 3)  # an address space for 2-objective and 3-iteration
>>> sp.print(a)  # prints the address in the current form
(2, 2)

>>> for a in sp:    # an address space is a `generator` iterating addresses
...     sp.print(a)
(4, 0)
(3, 1)
(2, 2)
(1, 3)
(0, 4)

>>> for a in sp.new():    # a `generator` iterating only new addresses
...     sp.print(a)
(3, 1)
(1, 3)

>>> print(len(sp))   # `len` returns the size of the address space
5

>>> print(len(sp.new()))   # `len` returns the number of new addresses
2

>>> Address([2, 2]) in sp   # Is an address `in` the address space
True

>>> Address([2, 2]) in sp.new()   # Is an address `in` the new addresses
False
'''
import curio
import numpy as np
import scipy.optimize
import scipy.misc
import cma
import ast

__all__ = (
    'weighted_sum',
    'tchebycheff',
    'augmented_tchebycheff',
    'pbi',
    'ipbi',
    'nelder_mead',
    'l_bfgs_b',
    'cmaes',
    'dvar',
    'dobj',
    'd0',
    'representing_iteration',
    'representing_number',
    'AddressSpace',
    'Address',
    'AWAState',
    'AWA',
    'fmin',
)

###############################################################################
# utilities
###############################################################################
def ffs(x):
    '''Returns the index, counting from 0, of the
    least significant set bit in `x`.

    Parameters
    ----------
    x: int
        An integer representing a bitstring.

    Returns
    -------
    index: int >= 0
        The index of the least significant set bit.
    '''
    return int(x & -x).bit_length() - 1


###############################################################################
# scalarizations
###############################################################################
def weighted_sum(f, w, z=None, **kwargs):
    '''Weighted sum scalarization.

    Parameters
    ----------
    f: awaitable
        A vector-valued function.
        A map from iterable to iterable.
    w: seq
        A weight.
    z: seq
        A utopian point.

    Returns
    -------
    fun: awaitable
        A scalar-valued function.
    '''
    if z is None:
        z = np.zeros(len(w))

    async def fun(x):
        fx = await f(x)
        return np.dot(w, fx - z)

    return fun


def epsilon_constraint(f, w, z=None, **kwargs):
    '''Epsilon constraint scalarization.
    f1 is treated as the primary objective,
    the other objectives are constraints (penalty).

    Parameters
    ----------
    f: awaitable
        A vector-valued function.
        A map from iterable to iterable.
    w: seq
        A weight.
        w[0] is an objective weight while w[1:] are constraint boundaries.
    z: seq
        A utopian point.

    Returns
    -------
    fun: awaitable
        A scalar-valued function.
    '''
    if z is None:
        z = np.zeros(len(w))

    async def fun(x):
        fx = await f(x)
        objective = w[0] * (fx[0] - z[0])
        penalty = np.sum([max(g, 0) for g in w[1:] - (fx[1:] - z[1:])])
        return objective + penalty

    return fun


def tchebycheff(f, w, z=None, **kwargs):
    '''Tchebycheff norm scalarization.

    Parameters
    ----------
    f: awaitable
        A vector-valued function.
    w: seq
        A weight.
    z: seq
        A utopian point.

    Returns
    -------
    fun: awaitable
        A scalar-valued function.
    '''
    if z is None:
        z = np.zeros(len(w))

    async def fun(x):
        fx = await f(x)
        return np.max(w * (fx - z))

    return fun


def augmented_tchebycheff(f, w, z=None, a=1e-8, **kwargs):
    '''Augmented Tchebycheff norm scalarization.

    Parameters
    ----------
    f: awaitable
        A vector-valued function.
    w: seq
        A weight.
    z: seq
        A utopian point.
    a: float
        Coefficients for weighted sum.

    Returns
    -------
    fun: awaitable
        A scalar-valued function.
    '''
    if z is None:
        z = np.zeros(len(w))

    async def fun(x):
        fx = await f(x)
        fx_z = fx - z
        return np.max(w * fx_z) + a * np.dot(w, fx_z)

    return fun


def pbi(f, w, z=None, a=0.25, **kwargs):
    '''Penalty-based boundary intersection.

    Parameters
    ----------
    f: awaitable
        A vector-valued function.
    w: seq
        A weight.
    z: seq
        A utopian point.
    a: float
        Coefficients for weighted sum.

    Returns
    -------
    fun: awaitable
        A scalar-valued function.
    '''
    if z is None:
        z = np.zeros(len(w))

    async def fun(x):
        fx = await f(x)
        fx_z = fx - z
        d1 = np.linalg.norm(np.dot(fx_z, w)) / np.linalg.norm(w)
        d2 = np.linalg.norm(fx_z - d1 * w)
        return d1 + a * d2

    return fun


def ipbi(f, w, n=None, a=0.25, **kwargs):
    '''Inverted penalty-based boundary intersection.

    Parameters
    ----------
    f: awaitable
        A vector-valued function.
    w: seq
        A weight.
    n: seq
        A nadir point.
    a: float
        Coefficients for weighted sum.

    Returns
    -------
    fun: awaitable
        A scalar-valued function.
    '''
    if n is None:
        n = np.ones(len(w))

    async def fun(x):
        fx = await f(x)
        n_fx = n - fx
        d1 = np.dot(n_fx, w) / np.linalg.norm(w)
        d2 = np.linalg.norm(n_fx - d1 * w)
        return -d1 + a * d2

    return fun


###############################################################################
# optimizers
###############################################################################
async def cmaes(f, x0, sigma0=0.2, **kwargs):
    '''CMA-ES.

    Parameters
    ----------
    f: awaitable
        A scalar-valued objective function.
    x0: seq
        An initial solution to start optimization.
    sigma0: float
        An initial standard deviation.
    kwargs: dict
        Parameters to pass options.

    Returns
    -------
    x: seq
        An optimal solution to ``f``.
    '''
    kwargs['verbose'] = -9
    kwargs['bounds'] = [0, 1]
    kwargs['seed'] = np.random.randint(2**32)

    es = cma.CMAEvolutionStrategy(x0, sigma0, inopts=kwargs)

    version1 = cma.__version__.split('.')[0] == '1'
    maxfevals = kwargs.get('maxfevals', float('+inf'))

    def es_result(i):
        return es.result()[i] if version1 else es.result[i]

    X = []
    while not es.stop():
        n = min(es.popsize, maxfevals - es.countevals)
        X = es.ask(n)
        tasks = []
        for x in X:
            tasks.append(await curio.spawn(f(x)))
        fX = await curio.gather(tasks)
        if n < es.popsize:
            break
        es.tell(X, fX)
    i = np.argmin([es.best.f] + fX)
    return es.best.x if i == 0 else X[i - 1]


###############################################################################
# pseudo-distance functions
###############################################################################
def dvar(a, b, **kwargs):
    '''L2-distance in variable space.

    Parameters
    ----------
    a: AWAState
        A point.
    b: AWAState
        A point.
    kwargs: dict
        Not used.

    Returns
    -------
    d: float
        The distance.
    '''
    return np.linalg.norm(a.x - b.x)


def dobj(a, b, **kwargs):
    '''L2-distance in objective space.

    Parameters
    ----------
    a: AWAState
        A point.
    b: AWAState
        A point.
    kwargs: dict
        Not used.

    Returns
    -------
    d: float
        The distance.
    '''
    return np.linalg.norm(a.y - b.y)


def d0(a, b, **kwargs):
    '''Zero function.

    Parameters
    ----------
    a: AWAState
        A point.
    b: AWAState
        A point.
    kwargs: dict
        Not used.

    Returns
    -------
    d: float
        Zero, allways.
    '''
    return 0


###############################################################################
# AWA
###############################################################################
def representing_iteration(m):
    '''The representing iteration for ``m``-objective problems,
    which is the number of iterations required to represent the ``m``-objective Pareto set/front.

    Parameters
    ----------
    m: int > 0
        The number of objectives

    Returns
    -------
    t: int > 0
        The representing iteration.
    '''
    return int(np.ceil(np.log2(m) + 1))


def representing_number(m):
    '''The representing number for ``m``-objective problems,
    which is the number of addresses required to represent the ``(m-1)``-simplex.

    Parameters
    ----------
    m: int > 0
        The number of objectives

    Returns
    -------
    p: int > 0
        The representing number.
    '''
    return len(AddressSpace(m, representing_iteration(m)))


class AddressSpace(object):
    '''An address space'''

    def __init__(self, obj, itr):
        '''Create an address space.

        Parameters
        ----------
        obj: int > 0
            The number of objectives
        itr: int > 0
            The number of iterations

        Returns
        -------
        adrsp: AddressSpace
            A generator yielding an address in the address space
            of ``obj``-objective, ``itr``-th iteration.
        '''
        self.obj = obj
        self.itr = itr

    def __iter__(self):
        '''Generate addresses in this address space.

        Returns
        -------
        adr: generator
            A generator yielding an address, per call, in this address space.
        '''

        def iterate(c, r):
            if len(c) == self.obj - 1:
                yield Address(c + (r,))
            else:
                for i in range(r, -1, -1):
                    yield from iterate(c + (i,), r - i)

        yield from iterate((), 2**(self.itr - 1))

    def __len__(self):
        '''The number of addresses in this address space.

        Returns
        -------
        len: int >= 0
            The number of addresses in this address space.
        '''
        return scipy.misc.comb(2**(self.itr - 1) + self.obj - 1, self.obj - 1, exact=True)

    def new(self):
        '''Generate new addresses in this address space.

        Returns
        -------
        adr: generator
            A generator yielding a new address in this address space.
        '''
        class New(object):
            '''A subspace of the address space, which contains all new addresses.'''

            def __init__(self, parent):
                self.parent = parent

            def __contains__(self, a):
                '''Check if the address is new in this address space.

                Parameters
                ----------
                a: Address
                    The address to test.

                Returns
                -------
                is_new: bool
                    True if the ``a`` is new;
                    False otherwise.
                '''
                return (isinstance(a, Address) and
                        len(a) == self.parent.obj and
                        sum(a) == 2**(self.parent.itr - 1))

            def __iter__(self):
                '''Generate new addresses in this address space.

                Returns
                -------
                adr: generator
                    A generator yielding a new address in this address space.
                '''
                return (a for a in self.parent if a in self)

            def __len__(self):
                '''The number of new addresses in this address space.

                Returns
                -------
                len: int >= 0
                    The number of new addresses in this address space.
                '''
                return len(self.parent) - len(AddressSpace(self.parent.obj, self.parent.itr - 1))

        return New(self)

    def __contains__(self, a):
        '''Check if the address is in this address space.

        Parameters
        ----------
        a: Address
            The address to test.

        Returns
        -------
        contained: bool
            True if ``a`` is in this address space;
            False otherwise.
        '''
        return (isinstance(a, Address) and
                len(a) == self.obj and
                sum(a) <= 2**(self.itr - 1))

    def nearest(self, a):
        '''Find the nearest address within this address space.

        Parameters
        ----------
        a: iterable
            The query.
            ``a`` must have the length equal to the dimension of
            this address space and must iterates nonnegative numbers.

        Returns
        -------
        nearest: Address
            The nearest address in this address space
            (in the sense of the L1 distance).
        '''
        r = 2**(self.itr - 1)
        a = np.asarray(a)
        b = a * r / np.sum(a)
        c = np.asarray(np.rint(b), dtype=int)
        d = np.abs(b - c)
        ix = np.argsort(d)[::-1]
        for i in ix:
            e = r - np.sum(c)
            if e == 0:
                break
            c[i] += np.sign(e)
        return Address(c)

    def average(self, addresses, weights=None):
        '''Take (weighted) average of addresses, and
        find the nearest address to it in this address space.

        Parameters
        ----------
        addresses: iterable
            The addresses to average.
            This must iterate ``Address`` objects, each of which has
            the length equal to the dimension of this address space.
        weights: iterable, optional
            The weights for addresses.

        Returns
        -------
        nearest: Address
            The nearest address to the average in this address space
            (in the sense of the L1 distance).
        '''
        a = np.asarray(addresses)
        s = np.sum(a, axis=1)
        w = np.fromiter((np.prod(np.delete(s, i)) for i in range(len(s))), dtype=int)
        a *= w[:, np.newaxis]
        a = np.average(a, axis=0, weights=weights)
        return self.nearest(a)

    def print(self, a):
        '''Print an address in the current form.

        Parameters
        ----------
        a: iterable
            The address to print.

        Returns
        -------
        None
        '''
        s = 2**(self.itr - 1) // sum(a)
        print(tuple( s*c for c in a))


class Address(tuple):
    '''An address'''

    def __new__(cls, c=[]):
        '''Create an address with coefficients ``c``.
        The coefficients are stored by the canonical form:
        if they are all power of two, then each of them
        are divided by the greatest common power so as to
        contain at least one odd number.

        Parameters
        ----------
        c: iterable, optional
            Coefficients of the address to create.
            The iterable must returns nonnegative values of ``int``
            that satisfy ``sum(c) == 2**n`` for some ``n >= 0``.

        Examples
        --------
        If the coefficients are already canonicalized, they are just used:
        >>> Address([1, 2, 5])
        (1, 2, 5)

        If all coefficients are power of two, they are divided by
        the greatest common power:
        >>> Address([2, 4, 10])
        (1, 2, 5)
        >>> Address([4, 4, 8])
        (1, 1, 2)
        '''
        c = np.fromiter(c, dtype=int)
        bits = np.bitwise_or.reduce(c)
        i = ffs(bits)
        return super().__new__(cls, np.right_shift(c, i))

    def __init__(self, _):
        self.__guide_pairs = None

    def arg_odd(self):
        '''Indices to odd coefficients of this address.

        Returns
        -------
        indices: list
            A list of ``int`` containing all indices of odd coefficients.
            The indices are sorted in ascending order.

        Examples
        --------
        >>> Address([1, 0, 2, 0, 3, 2]).arg_odd()
        [0, 4]

        Note that oddness is always checked for the canonical form.

        >>> Address([2, 0, 4, 0, 6, 4]).arg_odd()
        [0, 4]

        Because

        >>> Address([2, 0, 4, 0, 6, 4])
        (1, 0, 2, 0, 3, 2)
        '''
        return [i for i, c in enumerate(self) if c % 2 == 1]

    def arg_even(self):
        '''Indices to positive even coefficients of this address.

        Returns
        -------
        indices: list
            A list of ``int`` containing all indices of positive even coefficients.
            The indices are sorted in ascending order.

        Examples
        --------
        >>> Address([1, 0, 2, 0, 3, 2]).arg_even()
        [2, 5]

        Note that evenness is always checked for the canonical form.

        >>> Address([2, 0, 4, 0, 6, 4]).arg_even()
        [2, 5]

        Because

        >>> Address([2, 0, 4, 0, 6, 4])
        (1, 0, 2, 0, 3, 2)
        '''
        return [i for i, c in enumerate(self) if c > 0 and c % 2 == 0]

    def arg_zero(self):
        '''Indices to zero coefficients of this address.

        Returns
        -------
        indices: list
            A list of ``int`` containing all indices of zero coefficients.
            The indices are sorted in ascending order.

        Examples
        --------
        >>> Address([1, 0, 2, 0, 3, 2]).arg_zero()
        [1, 3]
        '''
        return [i for i, c in enumerate(self) if c == 0]

    def birth_time(self):
        '''The iteration when this address is created.

        Returns
        -------
        birth_time: int
            The iteration when this address is created.

        Examples
        --------
        >>> Address([1, 0, 2, 0, 3, 2]).birth_time()
        4
        '''
        return int(np.log2(np.sum(self))) + 1

    def deg(self):
        '''The degree of this address.

        Returns
        -------
        degree: int
            The degree of this address.
            The degree is defined as the number of nonzero
            coefficients minus one.

        Examples
        --------
        >>> Address([1, 0, 2, 0, 3, 2]).deg()
        3
        '''
        return len(self) - len(self.arg_zero()) - 1

    def guide_pair(self):
        '''Guide point pairs for this address.

        The number of guide point pairs is equal to
        the degree of the address.

        Returns
        -------
        guide: list
            A ``list`` of ``tuple``s of two ``Address`` objects.
            The length of the list is equal to ``self.deg()``.
            If the degree is zero, then the address has no
            guide point pairs and this method returns an empty list.

        Examples
        --------
        >>> g = Address([1, 0, 2, 0, 3, 2]).guide_pair()
        >>> g == [((0, 0, 1, 0, 2, 1), (1, 0, 1, 0, 1, 1)),
        ...       ((0, 0, 2, 0, 1, 1), (1, 0, 0, 0, 2, 1)),
        ...       ((0, 0, 1, 0, 1, 2), (1, 0, 1, 0, 2, 0))]
        True
        '''
        if self.__guide_pairs:
            return self.__guide_pairs
        e = self.arg_even()
        o = self.arg_odd()
        self.__guide_pairs = []
        b0 = [0] * len(self)
        for i, j in zip(o, [1, -1] * (len(o) // 2)):
            b0[i] = j
        for i in range(self.deg()):
            bi = b0[:]
            if 0 < i and i < len(o) - 1:
                bi[o[i]], bi[o[i - 1]] = bi[o[i - 1]], bi[o[i]]
            elif len(o) - 1 <= i:
                bi[o[-1]] += 2
                bi[e[i - len(o) + 1]] -= 2
            self.__guide_pairs.append(
                (Address(a - b for a, b in zip(self, bi)),
                 Address(a + b for a, b in zip(self, bi))))
        return self.__guide_pairs

    def guide(self):
        '''Generator of guide points for this address.

        A flatten version of ``guide_pair``.

        Returns
        -------
        guide: generator
            A ``generator`` of ``Address`` objects.

        Examples
        --------
        >>> for b in Address([1, 0, 2, 0, 3, 2]).guide():
        ...    print(b)
        (0, 0, 1, 0, 2, 1)
        (1, 0, 1, 0, 1, 1)
        (0, 0, 2, 0, 1, 1)
        (1, 0, 0, 0, 2, 1)
        (0, 0, 1, 0, 1, 2)
        (1, 0, 1, 0, 2, 0)
        '''
        for al, ar in self.guide_pair():
            yield al
            yield ar


class AWAState(object):
    '''The search status of an address in AWA.'''
    def __init__(self, a=None, w=None, x=None, y=None, z=None, n_itr=None, n_eval=None, task=None):
        self.a = a
        self.w = w
        self.x = x
        self.y = y
        self.z = z
        self.n_itr = n_itr
        self.n_eval = n_eval
        self.task = task

    def __repr__(self):
        return 'AWAState(\n a={}\n w={}\n x={}\n y={}\n z={}\n n_itr={}\n n_eval={}\n task={})'.format(
            self.a, self.w, self.x, self.y, self.z, self.n_itr, self.n_eval, self.task)


class AWA(object):
    '''Adaptive Weighted Aggregation'''

    def __init__(self, f, x0, w0=None, s=augmented_tchebycheff, o=cmaes, d=d0, e=1e-3, max_evals=None):
        '''Create an AWA object.

        Parameters
        ----------
        f: callable
            The vector-valued objective function to optimize.
        x0: iterable
            A matrix of initial solutions.
            The matrix shape determines the problem dimensionality.
            When the matrix is ``m`` x ``n``, the objective function ``f`` is
            treated as a map from an n-D space to an m-D space (i.e.,
            m objectives, n variables).
        w0: iterable, optional
            A matrix of initial weights.
            The matrix shape must be ``m`` x ``m`` where the value of ``m`` is
            determined by the shape of ``x0``.
            By default, ``np.eye(m)`` is used.
        s: callable, optional
            A scalarization function.
            This function must accept two arguments, a vector-valued objective
            function ``f`` and a weight ``w``, and returns a scalar-valued
            objective function.
            By default, ``augmented_tchebycheff`` function is used.
        o: callable, optional
            An optimization function.
            This function must accept two arguments, a scalar-valued objective
            function ``f`` and an initial solution ``x``, and returns an
            optimal solution to ``f``.
            By default, ``cmaes`` function is used.
        d: callable, optional
            A puseudo-distance function.
            This function must accept two arguments, an address ``a`` and
            another address ``b``, and returns an puseudo-distance between them.
            By default, ``d0`` function is used.
        e: float, optional
            A tolerance of optimization.
            This value is used to stop the search for an address.
            AWA stops the search when d(a, a_old) <= e holds.
            By default, ``1e-3`` is used.

        Examples
        --------
        >>> async def f(x):
        ...     return sum(v**2 for v in x), sum((v-1)**2 for v in x)
        >>> x0 = [[0, 0, 0],
        ...       [1, 1, 1]]
        >>> awa = AWA(f, x0)
        '''
        self.f = f
        self.s = s
        self.o = o
        self.d = d
        self.e = e
        self.max_evals = max_evals if max_evals else {}
        self.states = {}
        self.lock = curio.Lock()

        obj = len(x0)
        if w0 is None:
            w0 = np.eye(obj)

        for a, w, x in zip(AddressSpace(obj, 1), w0, x0):
            self.states[a] = AWAState(a, w, x)

    def __getitem__(self, key):
        '''Get the search result for an address.
        This function may block long time when the address has not been searched.
        If an asynchronous version is needed, use ``get`` method.

        Parameters
        ----------
        key: Address-like
            An address to get the search result.

        Returns
        -------
        result: AWAState
            An state containing the search result.
        '''
        key = Address(key)
        return curio.run(self.get(key))

    async def get(self, a):
        '''Get the search result for an address, asynchronously.

        Parameters
        ----------
        a: Address
            An address to get the search result.

        Returns
        -------
        result: AWAState
            An state containing the search result.
        '''
        task = await self.search_once(a)
        await task.join()
        return self.states[a]

    async def search_once(self, a):
        '''Search for an address if it has not been searched.

        Parameters
        ----------
        a: Address
            An address to search.

        Returns
        -------
        task: curio.Task
            An asynchronous task of search for the address.
        '''
        if a not in self.states:
            self.states[a] = AWAState(a)
        async with self.lock:
            if self.states[a].task is None:
                self.states[a].task = await curio.spawn(self.search(a))
        return self.states[a].task

    async def search(self, a):
        '''Search for an address whatever it has been searched.

        Parameters
        ----------
        a: Address
            An address to search.
        '''
        await curio.sleep(0)  # await for spawning this task
        ts = []
        for b in a.guide():
            ts.append(await self.search_once(b))
        for t in ts:
            await t.join()

        # initialize
        if a.deg() > 0:
            self.states[a].w = np.mean(
                [self.states[b].w for b in a.guide()], axis=0)
            self.states[a].x = np.mean(
                [self.states[b].x for b in a.guide()], axis=0)

        # relocate
        sigma0 = 1.0 / 6.0
        if a.deg():
            mean_distance = np.mean(
                [dvar(self.states[a], self.states[b]) for b in a.guide()])
            sigma0 = mean_distance / 3.0
        ev = self.max_evals.get(a, float('+inf'))
        while True:
            f = self.s(self.f, self.states[a].w)
            x0 = self.states[a].x[:]
            self.states[a].x = await self.o(
                f, x0, sigma0=sigma0, maxfevals=ev, copy_ratio=self.states[a].w[1])
            if self.d(self.states[a].x, x0) <= self.e:
                self.states[a].y = await self.f(self.states[a].x)
                break
            self.weight_adaptation(a)

    def weight_adaptation(self, a):
        '''Adapt the weight of an address to make a new scalarized objective
        function has an optimal solution at an equi-distant place between each
        guide point pair.

        Parameters
        ----------
        a: Address
            An address to adapt the weight.
        '''
        deg = a.deg()
        if deg == 0:
            return
        w = np.zeros(len(a))
        for al, ar in a.guide_pair():
            dl = self.d(self.states[a], self.states[al])
            dr = self.d(self.states[a], self.states[ar])
            if dl > dr:
                r = (dl + dr) / (2 * dl)
                w += r * self.states[a].w + (1 - r) * self.states[al].w
            elif dr > dl:
                r = (dl + dr) / (2 * dr)
                w += r * self.states[a].w + (1 - r) * self.states[ar].w
            else:
                w += self.states[a].w
        self.states[a].w = w / deg


async def fmin(problem, confopt, logger=None):
    '''Optimize the function by AWA.

    Parameters
    ----------
    problem: awa.Problem
        The vector-valued objective function to optimize.
    confopt: dict
        A dictonary containing configurations of AWA.
        'max_iters', 'max_evals', 'weights'.
    logger: logging.Logger, optional
        A logger.

    Returns
    -------
    awa: AWA
        An AWA object containing results.

    Examples
    --------
    >>> async def f(x):
    ...     return sum(v**2 for v in x), sum((v-1)**2 for v in x)
    >>> f.dim = 2
    >>> f.obj = 2
    >>> conf = {'max_iters': 2}
    >>> results = fmin(f, conf)
    >>> np.allclose(results[(0, 1)].x, [1, 1], atol=0.1)
    True
    '''
    itr = confopt.get('max_iters', representing_iteration(problem.obj))
    evs = confopt.get('max_evals', {})
    evs = {Address(ast.literal_eval(k)):v for k, v in evs.items()}
    seed = confopt.get('seed')
    np.random.seed(seed)
    x0 = np.asarray(
        confopt.get(
            'x0',
            np.random.rand(problem.obj, problem.dim) * 0.5 + 0.25))
    w0 = np.asarray(
        confopt.get(
            'w0',
            np.eye(problem.obj))))
    scalarization = globals()[confopt.get('scalarization', 'weighted_sum')]
    optimization = globals()[confopt.get('optimization', 'cmaes')]
    awa = AWA(problem, x0, w0, s=scalarization, o=optimization, max_evals=evs)

    async def search_all():
        tasks = []
        for a in AddressSpace(problem.obj, itr):
            tasks.append(await curio.spawn(awa.search_once(a)))
        for t in tasks:
            await t.join()

    await search_all()
    return awa
