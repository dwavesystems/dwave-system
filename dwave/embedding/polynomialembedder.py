# Copyright 2016 D-Wave Systems Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
# ================================================================================================
"""

**This module is used as a private source file. It is subject to change and not
recommended for use at this time.**

This file implements a polynomial-time algorithm to find a maximum-sized
native clique embedding in an induced Chimera subgraph.  It also provides
functionality to find maximum-sized native embeddings of complete bipartite
graphs (bicliques).  Additionally, we wrap the polynomial-time algorithm to
compute maximum-sized native (bi)clique embeddings for non-induced subgraphs
(that is, for processors which have both qubit and coupler failures).

The :class:`eden_processor` class provides the implementation of the polynomial-time
algorithm described in [BKR]_.  That algorithm relies, in part, on building a
cache of the number of "horizontal wires" and "vertical wires" across a
processor (that is, paths of horizontally- and vertically- aligned qubits) in
all of the rectangular sub-regions of the processor.  As a side-effect, we can
use the same cache to look at bicliques; so that functionality is also built
into the :class:`eden_processor` class.

The :class:`eden_processor` class is written under the assumption that if a pair of
qubits exists in the processor and the Chimera structure implies that those
qubits are connected by a coupler, then that coupler exists (that is, we
assume that the graph is an induced Chimera subgraph).  That assumption
is not true for all processors, so we need to accomodate this somehow.

The :class:`processor` class is suitable for end-users.  It can be used to either
produce optimal native embeddings, or approximately-optimal native embeddings.


.. [BKR] Kelly Boothby, Andrew D. King, Aidan Roy.  Fast clique minor
    generation in Chimera qubit connectivity graphs. Quantum Information
    Processing, (2015).  http://arxiv.org/abs/1507.04774

"""
from __future__ import division

from random import shuffle, randint, choice, sample
from collections import defaultdict
from itertools import product

__author__ = "Kelly Boothby"


def _accumulate_random(count, found, oldthing, newthing):
    """This performs on-line random selection.

    We have a stream of objects

        o_1,c_1; o_2,c_2; ...

    where there are c_i equivalent objects like o_1.  We'd like to pick
    a random object o uniformly at random from the list

        [o_1]*c_1 + [o_2]*c_2 + ...

    (actually, this algorithm allows arbitrary positive weights, not
    necessarily integers)  without spending the time&space to actually
    create that list.  Luckily, the following works:

        thing = None
        c_tot
        for o_n, c_n in things:
            c_tot += c_n
            if randint(1,c_tot) <= c_n:
                thing = o_n

    This function is written in an accumulator format, so it can be
    used one call at a time:

    EXAMPLE:
        > thing = None
        > count = 0
        > for i in range(10):
        >     c = 10-i
        >     count, thing = accumulate_random(count,c,thing,i)


    INPUTS:
        count: integer, sum of weights found before newthing
        found: integer, weight for newthing
        oldthing: previously selected object (will never be selected
                  if count == 0)
        newthing: incoming object

    OUTPUT:
        (newcount, pick): newcount is count+found, pick is the newly
                          selected object.
    """
    if randint(1, count + found) <= found:
        return count + found, newthing
    else:
        return count + found, oldthing


class eden_processor(object):
    """Class to hold a processor, and embed cliques in said processor.

    Syntactic sugar: (assume proc is a :class:`eden_processor` instance)

    >>> q in proc # q is a working qubit
    True
    >>> proc[q] # returns the neighbors of q connected by a working coupler
    set([...])

    **CAVEAT**: the embedding utilities herein assume that the only failed
    couplers connected to working qubits are between two cells.  Any
    pair of working qubits on opposite sides of the same unit cell are
    assumed to be connected by working couplers.
    """

    def __init__(self, edgelist, M, N, L, random_bundles=False):
        """Constructs an :class:`eden_processor` instance

        Args:
            edgelist (list): a list of tuples (p,q) where each of p and q is
                a qubit address in chimera coordinates.  These are taken to be
                the working couplers.
            M (int): first chimera parameter, :math:`C_{M,N,L}`
            N (int): as above
            L (int): as above
            random_bundles (bool): whether or not to shuffle horizontal and
                vertical line bundles when assembling ells.
        """
        couplers = defaultdict(set)
        for p, q in edgelist:
            couplers[p].add(q)
            couplers[q].add(p)
        self._couplers = couplers
        self.M = M
        self.N = N
        self.L = L
        self._compute_vline_scores()
        self._compute_hline_scores()
        self.random_bundles = random_bundles
        self._biclique_size = {}
        self._biclique_size_computed = False

    def __getitem__(self, q):
        """Return the qubits coupled to `q`.
        """
        return self._couplers[q]

    def __contains__(self, q):
        """Check if the qubit `q` is contained in this processor.
        """
        return q in self._couplers

    def __iter__(self):
        """Iterate over the qubits in this processor.
        """
        for q in self._couplers:
            yield q

    def vline_score(self, x, ymin, ymax):
        """Returns the number of unbroken paths of qubits

        >>> [(x,y,1,k) for y in range(ymin,ymax+1)]

        for :math:`k = 0,1,\cdots,L-1`.  This is precomputed for speed.
        """
        return self._vline_score[x, ymin, ymax]

    def hline_score(self, y, xmin, xmax):
        """Returns the number of unbroken paths of qubits

        >>> [(x,y,0,k) for x in range(xmin,xmax+1)]

        for :math:`k = 0,1,\cdots,L-1`.  This is precomputed for speed.
        """
        return self._hline_score[y, xmin, xmax]

    def _compute_vline_scores(self):
        """Does the hard work to prepare ``vline_score``.
        """
        M, N, L = self.M, self.N, self.L
        vline_score = {}
        for x in range(M):
            laststart = [0 if (x, 0, 1, k) in self else None for k in range(L)]
            for y in range(N):
                block = [0] * (y + 1)
                for k in range(L):
                    if (x, y, 1, k) not in self:
                        laststart[k] = None
                    elif laststart[k] is None:
                        laststart[k] = y
                        block[y] += 1
                    elif y and (x, y, 1, k) not in self[x, y - 1, 1, k]:
                        laststart[k] = y
                    else:
                        for y1 in range(laststart[k], y + 1):
                            block[y1] += 1
                for y1 in range(y + 1):
                    vline_score[x, y1, y] = block[y1]
        self._vline_score = vline_score

    def _compute_hline_scores(self):
        """Does the hard work to prepare ``hline_score``.
        """
        M, N, L = self.M, self.N, self.L
        hline_score = {}
        for y in range(N):
            laststart = [0 if (0, y, 0, k) in self else None for k in range(L)]
            for x in range(M):
                block = [0] * (x + 1)
                for k in range(L):
                    if (x, y, 0, k) not in self:
                        laststart[k] = None
                    elif laststart[k] is None:
                        laststart[k] = x
                        block[x] += 1
                    elif x and (x, y, 0, k) not in self[x - 1, y, 0, k]:
                        laststart[k] = x
                    else:
                        for x1 in range(laststart[k], x + 1):
                            block[x1] += 1
                for x1 in range(x + 1):
                    hline_score[y, x1, x] = block[x1]
        self._hline_score = hline_score

    def _compute_biclique_sizes(self, recompute=False):
        """Calls ``self.biclique_size(...)`` for every rectangle contained in this
        processor, to fill the biclique size cache.

        INPUTS:
            recompute: if ``True``, then we dump the existing cache and compute
                all biclique sizes from scratch.  (default: ``False``)
        """
        if recompute or not self._biclique_size_computed:
            self._biclique_size = {}
            self._biclique_size_to_length = defaultdict(dict)
            self._biclique_length_to_size = defaultdict(dict)
        else:
            return
        M, N = self.M, self.N
        for xmax in range(M):
            for xmin in range(xmax + 1):
                for ymax in range(N):
                    for ymin in range(ymax + 1):
                        ab = self.biclique_size(xmin, xmax, ymin, ymax)
                        wh = xmax - xmin + 1, ymax - ymin + 1
                        self._biclique_size_to_length[ab][
                            wh] = (xmin, xmax, ymin, ymax)
                        self._biclique_length_to_size[wh][
                            ab] = (xmin, xmax, ymin, ymax)

        self._biclique_size_computed = True

    def biclique_size(self, xmin, xmax, ymin, ymax):
        """Returns the size parameters ``(m,n)`` of the complete bipartite graph
        :math:`K_{m,n}` comprised of ``m`` unbroken chains of horizontally-aligned qubits
        and ``n`` unbroken chains of vertically-aligned qubits (known as line
        bundles)

        INPUTS:
            xmin,xmax,ymin,ymax: integers defining the bounds of a rectangle
            where we look for unbroken chains.  These ranges include both
            endpoints.

        OUTPUTS:
            m,n: integers corresponding to the number of horizontal and
                vertical line bundles contained in this rectangle.
        """
        try:
            return self._biclique_size[xmin, xmax, ymin, ymax]
        except KeyError:
            hscore = self.hline_score(ymin, xmin, xmax)
            vscore = self.vline_score(xmin, ymin, ymax)
            if ymin < ymax:
                hscore += self.biclique_size(xmin, xmax, ymin + 1, ymax)[0]
            if xmin < xmax:
                vscore += self.biclique_size(xmin + 1, xmax, ymin, ymax)[1]
            self._biclique_size[xmin, xmax, ymin, ymax] = hscore, vscore
            return hscore, vscore

    def biclique(self, xmin, xmax, ymin, ymax):
        """Compute a maximum-sized complete bipartite graph contained in the
        rectangle defined by ``xmin, xmax, ymin, ymax`` where each chain of
        qubits is either a vertical line or a horizontal line.

        INPUTS:
            xmin,xmax,ymin,ymax: integers defining the bounds of a rectangle
            where we look for unbroken chains.  These ranges include both
            endpoints.

        OUTPUT:
            (A_side, B_side): a tuple of two lists containing lists of qubits.
                the lists found in ``A_side`` and ``B_side`` are chains of qubits.
                These lists of qubits are arranged so that

                >>> [zip(chain,chain[1:]) for chain in A_side]

                and

                >>> [zip(chain,chain[1:]) for chain in B_side]

                are lists of valid couplers.
        """

        Aside = sum((self.maximum_hline_bundle(y, xmin, xmax)
                     for y in range(ymin, ymax + 1)), [])
        Bside = sum((self.maximum_vline_bundle(x, ymin, ymax)
                     for x in range(xmin, xmax + 1)), [])
        return Aside, Bside


    def _contains_line(self, line):
        """Test if a chain of qubits is completely contained in ``self``.  In
        particular, test if all qubits are present and the couplers
        connecting those qubits are also connected.

        NOTE: this function assumes that ``line`` is a list or tuple of
        qubits which satisfies the precondition that ``(line[i],line[i+1])``
        is supposed to be a coupler for all ``i``.

        INPUTS:
            line: a list of qubits satisfying the above precondition

        OUTPUT:
            boolean
        """
        return all(v in self for v in line) and all(u in self[v] for u, v in zip(line, line[1::]))

    def maximum_vline_bundle(self, x0, y0, y1):
        """Compute a maximum set of vertical lines in the unit cells ``(x0,y)``
        for :math:`y0 \leq y \leq y1`.

        INPUTS:
            y0,x0,x1: int

        OUTPUT:
            list of lists of qubits
        """

        y_range = range(y1, y0 - 1, -1) if y0 < y1 else range(y1, y0 + 1)
        vlines = [[(x0, y, 1, k) for y in y_range] for k in range(self.L)]
        return list(filter(self._contains_line, vlines))

    def maximum_hline_bundle(self, y0, x0, x1):
        """Compute a maximum set of horizontal lines in the unit cells ``(x,y0)``
        for :math:`x0 \leq x \leq x1`.

        INPUTS:
            y0,x0,x1: int

        OUTPUT:
            list of lists of qubits
        """
        x_range = range(x0, x1 + 1) if x0 < x1 else range(x0, x1 - 1, -1)
        hlines = [[(x, y0, 0, k) for x in x_range] for k in range(self.L)]
        return list(filter(self._contains_line, hlines))

    def maximum_ell_bundle(self, ell):
        """Return a maximum ell bundle in the rectangle bounded by

            :math:`\{x0,x1\} \\times \{y0,y1\}`

        with vertical component

            :math:`(x0,y0) ... (x0,y1) = {x0} \\times \{y0,...,y1\}`

        and horizontal component

            :math:`(x0,y0) ... (x1,y0) = \{x0,...,x1\} \\times \{y0\}`.

        Note that we don't require :math:`x0 \leq x1` or :math:`y0 \leq y1`.  We go
        through some shenanigans so that the qubits we return
        are all in a path.  A nice side-effect of this is that

            >>> chains = maximum_ell_bundle(...)
            >>> edges = [zip(path,path[:-1]) for path in chains]

        where ``edges`` will be a list of lists of chain edges.

        INPUTS::
            ell: a tuple of 4 integers defining the ell, ``(x0, x1, y0, y1)``

        OUTPUT::
            chains: list of lists of qubits

        Note: this function only to be called to construct a
        native clique embedding *after* the block embedding has
        been constructed.  Using this to evaluate the goodness
        of an ell block will be slow.
        """
        (x0, x1, y0, y1) = ell
        hlines = self.maximum_hline_bundle(y0, x0, x1)
        vlines = self.maximum_vline_bundle(x0, y0, y1)

        if self.random_bundles:
            shuffle(hlines)
            shuffle(vlines)

        return [v + h for h, v in zip(hlines, vlines)]

    def _combine_clique_scores(self, rscore, hbar, vbar):
        """Computes the score of a partial native clique embedding given the score
        attained by the already-placed ells, together with the ell block
        defined by ``hbar = (y0,xmin,xmax)``, and  ``vbar = (x0,ymin,ymax)``.

        In the plain :class:`eden_processor` class, this is simply the number of ells
        contained in the partial native clique after adding the new ells.
        """
        (y0, xmin, xmax) = hbar
        (x0, ymin, ymax) = vbar
        if rscore is None:
            rscore = 0
        hscore = self.hline_score(y0, xmin, xmax)
        vscore = self.vline_score(x0, ymin, ymax)
        if vscore < hscore:
            score = rscore + vscore
        else:
            score = rscore + hscore
        return score

    def maxCliqueWithRectangle(self, R, maxCWR):
        """This does the dirty work for :func:`nativeCliqueEmbed`.  Not meant to be
        understood or called on its own.  Guaranteed to maintain the inductive
        hypothesis that ``maxCWR`` is optimal.  We put in the tiniest amount of
        effort to return a uniform random choice of a maximum-sized native
        clique embedding.

        INPUTS:
            R (tuple): the rectangle specified as a tuple ``(xmin,xmax,ymin,ymax)``

            maxCWR (dict): the dictionary we're building inductively which maps
            ``R -> (score, ell, r, num)`` where score is the size of the best
            clique with working rectangle ``R``, and ``ell`` and ``r`` indicate how to
            reconstruct that clique: add the maximum line bundle in ``ell``, and
            recursively examine the working rectangle ``r``.  The parameter ``num`` is the number
            of cliques with ``R`` as a working rectangle and size equal to ``score``.

        OUTPUT:
            score (int): the score for the returned clique (just ``len(clique)``
            in the class :class:`eden_processor`; may differ in subclasses)

            best (tuple): a tuple ``(score, ell, parent, num)`` to be stored in
            ``maxCWR[R]``.

            * score: as above
            * ell: ``(x0,x1,y0,y1)`` defines the unit cells in an ell-shaped
                region: ``(x0,y1),...,(x0,y0),...,(x1,y0)``
            * parent: the rectangle ``R1`` for which the clique generated
                recursively by looking up ``maxCWR[R1]`` as described above.
            * num: the number of partial native clique embeddings with ``R``
                as a working rectangle and a score of ``score``.

        """

        (xmin, xmax, ymin, ymax) = R
        best = nothing = 0, None, None, 1
        bestscore = None
        count = 0
        N = self.N

        Xlist = (xmin, xmax, xmin + 1, xmax), (xmax, xmin, xmin, xmax - 1)
        Ylist = (ymin - 1, ymax, ymin - 1,
                 ymax), (ymax + 1, ymin, ymin, ymax + 1)
        XY = [(X, Y) for X in Xlist for Y in Ylist if 0 <= Y[2] <= Y[3] < N]

        bests = []
        for X, Y in XY:
            x0, x1, nxmin, nxmax = X
            y0, y1, nymin, nymax = Y
            r = nxmin, nxmax, nymin, nymax
            try:
                rscore, rell, rparent, nr = maxCWR[r]
            except:
                rscore, nr = None, 1
            score = self._combine_clique_scores(
                rscore, (y0, xmin, xmax), (x0, nymin, nymax))
            if bestscore is None or score > bestscore:
                bestscore = score
                count = 0
            if score == bestscore:
                count, best = _accumulate_random(
                    count, nr, best, (score, (x0, x1, y0, y1), r, nr))

        return bestscore, best

    def nativeCliqueEmbed(self, width):
        """Compute a maximum-sized native clique embedding in an induced
        subgraph of chimera with all chainlengths ``width+1``.

        INPUTS:
            width: width of the squares to search, also `chainlength`-1

        OUTPUT:
            score: the score for the returned clique (just ``len(clique)``
            in the class :class:`eden_processor`; may differ in subclasses)

            clique: a list containing lists of qubits, each associated
            to a chain.  These lists of qubits are carefully
            arranged so that

            >>> [zip(chain,chain[1:]) for chain in clique]

            is a list of valid couplers.

        """
        maxCWR = {}

        M, N = self.M, self.N
        maxscore = None
        count = 0
        key = None
        for w in range(width + 2):
            h = width - w - 2
            for ymin in range(N - h):
                ymax = ymin + h
                for xmin in range(M - w):
                    xmax = xmin + w
                    R = (xmin, xmax, ymin, ymax)
                    score, best = self.maxCliqueWithRectangle(R, maxCWR)
                    maxCWR[R] = best
                    if maxscore is None or (score is not None and maxscore < score):
                        maxscore = score
                        key = None  # this gets overwritten immediately
                        count = 0  # this gets overwritten immediately
                    if maxscore == score:
                        count, key = _accumulate_random(count, best[3], key, R)

        clique = []
        while key in maxCWR:
            score, ell, key, num = maxCWR[key]
            if ell is not None:
                meb = self.maximum_ell_bundle(ell)
                clique.extend(meb)
        return maxscore, clique

    def largestNativeClique(self, max_chain_length=None):
        """Returns the largest native clique embedding we can find on the
        processor, with the shortest chainlength possible (for that
        clique size).

        OUTPUT:
            score: the score for the returned clique (just ``len(clique)``
            in the class :class:`eden_processor`; may differ in subclasses)

            clique: a list containing lists of qubits, each associated
            to a chain.  These lists of qubits are carefully
            arranged so that

             >>> [zip(chain,chain[1:]) for chain in clique]

            is a list of valid couplers.

        CAVEAT: we assume that the only failed couplers connected to working
        qubits are between two cells.  Any pair of working qubits on opposite
        sides of the same unit cell are assumed to be connected by working
        couplers.
        """
        bigclique = []
        bestscore = None
        if max_chain_length is None:
            wmax = min(self.M, self.N)
        else:
            wmax = max_chain_length - 1

        for w in range(wmax + 1):
            score, clique = self.nativeCliqueEmbed(w)
            if bestscore is None or score > bestscore:
                bigclique = clique
                bestscore = score
        return bestscore, bigclique

    def tightestNativeClique(self, n):
        """Returns a native clique embedding with the shortest chains and
        at least ``n`` completely connected chains, or ``None`` on failure.

        INPUTS:
            n: size of clique to return

        OUTPUT:
            score: the score for the returned clique (just ``len(clique)``
            in the class :class:`eden_processor`; may differ in subclasses)

            clique: a list containing lists of qubits, each associated
            to a chain.  These lists of qubits are carefully arranged so that

            >>> [zip(chain,chain[1:]) for chain in clique]

            is a list of valid couplers.  If no ``n``-clique is available on
            the processor, this is ``None``.

        """
        for w in range(1, min(self.M, self.N) + 1):
            score, clique = self.nativeCliqueEmbed(w)
            if len(clique) >= n:
                return score, clique[:n]
        return None, []

    def largestNativeBiClique(self, chain_imbalance=0, max_chain_length=None):
        """Returns a native embedding for the complete bipartite graph :math:`K_{n,m}`
        for :math:`n \leq m`; where :math:`n` is as large as possible and :math:`m` is as large as
        possible subject to :math:`n`.  The native embedding of a complete bipartite
        graph is a set of horizontally-aligned qubits connected in lines
        together with an equal-sized set of vertically-aligned qubits
        connected in lines.

        INPUTS:
            chain_imbalance: how big of a difference to allow between the
            chain lengths on the two sides of the bipartition. If ``None``,
            then we allow an arbitrary imbalance.  (default: ``0``)

            max_chain_length: longest chain length to consider or ``None`` if chain
            lengths are allowed to be unbounded.  (default: ``None``)

        OUTPUT:
            score (tuple): the score for the returned clique (just ``(n,m)`` in the class
            :class:`eden_processor`; may differ in subclasses)

            embedding (tuple): a tuple of two lists containing lists of qubits.
            If ``embedding = (A_side, B_side)``, the lists found in ``A_side`` and
            ``B_side`` are chains of qubits.
            These lists of qubits are arranged so that

            >>> [zip(chain,chain[1:]) for chain in A_side]

            and

            >>> [zip(chain,chain[1:]) for chain in B_side]

            are lists of valid couplers.

        """
        self._compute_biclique_sizes()
        Len2Siz = self._biclique_length_to_size
        Siz2Len = self._biclique_size_to_length
        overkill = self.M + self.N
        if max_chain_length is None:
            max_chain_length = overkill
        if chain_imbalance is None:
            chain_imbalance = overkill

        def acceptable_chains(t):
            a, b = t
            return a <= max_chain_length and b <= max_chain_length and abs(a - b) <= chain_imbalance

        def sortedpair(k):
            return min(k), max(k)

        feasible_sizes = {mn for mn, S in Siz2Len.items()
                          if any(map(acceptable_chains, S))}
        m, n = max(feasible_sizes, key=sortedpair)
        best_r = None
        best_ab = overkill, overkill
        for mn in set(((m, n), (n, m))) & feasible_sizes:
            for ab, r in Siz2Len[mn].items():
                ab = max(ab), min(ab)
                if acceptable_chains(ab) and ab < best_ab:
                    best_ab = ab
                    best_r = r

        bestsize = sortedpair(self.biclique_size(*best_r))
        bestbiclique = self.biclique(*best_r)
        return bestsize, (bestbiclique[0], bestbiclique[1])

    def tightestNativeBiClique(self, n, m=None, chain_imbalance=0, max_chain_length=None):
        """Returns a native embedding for the complete bipartite graph :math:`K_{n,m}`
        with the shortest chains possible.  The native embedding of a complete
        bipartite graph is a set of horizontally-aligned qubits connected
        in lines together with an equal-sized set of vertically-aligned qubits
        connected in lines.

        INPUTS:
            n (int): target size for one side of the bipartition

            m (int): target size for the other side of the bipartition, or
            None for the special case :math:`m=n` (default: ``None``)

            chain_imbalance (int): how big of a difference to allow between the
            chain lengths on the two sides of the bipartition.  If
            ``None``, we allow an arbitrary imbalance. (default: ``0``)

            max_chain_length (int): longest chain length to consider or ``None`` if
            chain lengths are allowed to be unbounded.  (default: ``None``)

        OUTPUT:
            score (tuple): the score for the returned biclique (just ``(min(n,m),max(n,m))`` in the
            class :class:`eden_processor`; may differ in subclasses).  If no :math:`K_{n,m}`
            is available on the processor, this is None.

            embedding (tuple): a tuple of two lists containing lists of qubits.
            If `embedding = (A_side, B_side)`, the lists found in `A_side` and
            `B_side` are chains of qubits.
            These lists of qubits are arranged so that

            >>> [zip(chain,chain[1:]) for chain in A_side]

            and

            >>> [zip(chain,chain[1:]) for chain in B_side]

            are lists of valid couplers.  If no :math:`K_{n,m}` is available on
            the processor, this is None.

        """
        if m is None:
            m = n

        self._compute_biclique_sizes()
        Len2Siz = self._biclique_length_to_size
        Siz2Len = self._biclique_size_to_length
        overkill = self.M + self.N + 1
        if max_chain_length is None:
            max_chain_length = overkill
        if chain_imbalance is None:
            chain_imbalance = overkill

        def acceptable_chains(t):
            a, b = t
            return (a <= max_chain_length and b <= max_chain_length and
                    abs(a - b) <= chain_imbalance)

        def acceptable_size(t):
            m0, n0 = t
            return (m0 >= m and n0 >= n) or (m0 >= n and n0 >= m)

        feasible_sizes = {mn for mn, S in Siz2Len.items()
                          if acceptable_size(mn) and any(map(acceptable_chains, S))}

        best_r = None
        best_ab = overkill, overkill
        best_mn = None
        for mn in feasible_sizes:
            for ab, r in Siz2Len[mn].items():
                ab = max(ab), min(ab)
                if acceptable_chains(ab) and ab < best_ab:
                    best_ab = ab
                    best_r = r
                    best_mn = mn
        if best_r is None:
            return None, None
        bestsize = self.biclique_size(*best_r)
        bestbiclique = self.biclique(*best_r)

        best_m, best_n = best_mn
        if m <= best_m and n <= best_n:
            return bestsize,  (bestbiclique[1][:n], bestbiclique[0][:m])
        else:
            return bestsize,  (bestbiclique[0][:n], bestbiclique[1][:m])


class processor:
    """A class representing a subgraph of Chimera :math:`C_{M,N,L}` for the purpose of
    embedding cliques.  The clique embedding algorithm is polynomial if there
    are a fixed number of broken couplers between working qubits, but
    exponential otherwise (provided those broken couplers connect qubits in
    the same unit cell).  This class wraps the :class:`eden_processor` class, where
    the polynomial algorithm is implemented, and we do the exponential work
    here.

    If there are a small number of bad couplers, we simply store a few
    copies with certain qubits deleted.  Otherwise, memory use would grow
    too large, so we are forced to construct each subprocessor on the fly.
    Perfect optimization gradually becomes impossible, so we default to
    examine no more than 64 subprocessors.  This limit can be increased with
    a parameter to :func:`__init__`.
    """

    def __init__(self, edgelist, M=8, N=None, L=4, proc_limit=64, linear=True,
                 random_bundles=False):
        """Constructs a :class:`processor` instance

        INPUTS:
            edgelist: a list of tuples (p,q) where each of p and q is
                a qubit address in chimera coordinates.  These are taken
                to be the working couplers.
            M,N,L: the chimera parameters, :math:`C_{M,N,L}`.  If N is None, set
                N=M.  Defaults: M=8, N=None, L=4
            proc_limit: the maximum number of subprocessors to examine
                (default 64)
            linear: True if the couplers are given as linear indices
                instead of chimera indices.  (default True)
            random_bundles: True if ell bundles should be shuffled; use this
                to generate a large number of embeddings with (perhaps)
                different performance characteristics. (default False)
        """
        if N is None:
            N = M
        self.M, self.N, self.L = M, N, L
        if linear:
            edgelist = [(_to_chimera(M, N, L, p), _to_chimera(M, N, L, q)) for p, q in edgelist]

        self._linear = linear
        self._proc_limit = proc_limit
        self._qubits = set(q for e in edgelist for q in e)
        self._edgelist = edgelist
        self._random_bundles = random_bundles
        self._proc0 = eden_processor(edgelist, M, N, L, random_bundles=random_bundles)
        self._find_evil()
        self._compute_deletions()

    def _compute_all_deletions(self):
        """Returns all minimal edge covers of the set of evil edges.
        """
        minimum_evil = []
        for disabled_qubits in map(set, product(*self._evil)):
            newmin = []
            for s in minimum_evil:
                if s < disabled_qubits:
                    break
                elif disabled_qubits < s:
                    continue
                newmin.append(s)
            else:
                minimum_evil = newmin + [disabled_qubits]
        return minimum_evil

    def _subprocessor(self, disabled_qubits):
        """Create a subprocessor by deleting a set of qubits.  We assume
        this removes all evil edges, and return an :class:`eden_processor`
        instance.
        """
        edgelist = [(p, q) for p, q in self._edgelist if
                    p not in disabled_qubits and
                    q not in disabled_qubits]
        return eden_processor(edgelist, self.M, self.N, self.L, random_bundles=self._random_bundles)

    def _compute_deletions(self):
        """If there are fewer than self._proc_limit possible deletion
        sets, compute all subprocessors obtained by deleting a
        minimal subset of qubits.
        """
        M, N, L, edgelist = self.M, self.N, self.L, self._edgelist
        if 2**len(self._evil) <= self._proc_limit:
            deletions = self._compute_all_deletions()
            self._processors = [self._subprocessor(d) for d in deletions]
        else:
            self._processors = None

    def _random_subprocessor(self):
        """Creates a random subprocessor where there is a coupler between
        every pair of working qubits on opposite sides of the same cell.
        This is guaranteed to be minimal in that adding a qubit back in
        will reintroduce a bad coupler, but not to have minimum size.

        OUTPUT:
            an :class:`eden_processor` instance
        """
        deletion = set()
        for e in self._evil:
            if e[0] in deletion or e[1] in deletion:
                continue
            deletion.add(choice(e))
        return self._subprocessor(deletion)

    def _random_subprocessors(self):
        """Produces an iterator of subprocessors.  If there are fewer than
        self._proc_limit subprocessors to consider (by knocking out a
        minimal subset of working qubits incident to broken couplers),
        we work exhaustively.  Otherwise, we generate a random set of
        ``self._proc_limit`` subprocessors.

        If the total number of possibilities is rather small, then we
        deliberately pick a random minimum subset to avoid coincidences.
        Otherwise, we give up on minimum, satisfy ourselves with minimal,
        and randomly generate subprocessors with :func:`self._random_subprocessor`.

        OUTPUT:
            an iterator of eden_processor instances.
        """
        if self._processors is not None:
            return (p for p in self._processors)
        elif 2**len(self._evil) <= 8 * self._proc_limit:
            deletions = self._compute_all_deletions()
            if len(deletions) > self._proc_limit:
                deletions = sample(deletions, self._proc_limit)
            return (self._subprocessor(d) for d in deletions)
        else:
            return (self._random_subprocessor() for i in range(self._proc_limit))

    def _map_to_processors(self, f, objective):
        """Map a function to a list of processors, and return the output that
        best satisfies a transitive objective function.  The list of
        processors will differ according to the number of evil qubits and
        :func:`_proc_limit`, see details in :func:`self._random_subprocessors`.

        INPUT:
            f (callable): the function to call on each processor

            objective (callable): a function where objective(x,y) is True if x is
                better than y, and False otherwise.  Assumes transitivity!

        OUTPUT:
            best: the object returned by f that maximizes the objective.
        """

        P = self._random_subprocessors()
        best = f(next(P))
        for p in P:
            x = f(p)
            if objective(best, x):
                best = x
        return best[1]

    def _objective_bestscore(self, old, new):
        """An objective function that returns True if new has a better score
        than old, and ``False`` otherwise.

        INPUTS:
            old (tuple): a tuple (score, embedding)

            new (tuple): a tuple (score, embedding)

        """
        (oldscore, oldthing) = old
        (newscore, newthing) = new
        if oldscore is None:
            return True
        if newscore is None:
            return False
        return oldscore < newscore

    def _objective_qubitcount(self, old, new):
        """An objective function that returns True if new uses fewer qubits
        than old, and False otherwise.  This objective function should only be
        used to compare embeddings of the same graph (or at least embeddings of
        graphs with the same number of qubits).

        INPUTS:
            old (tuple): a tuple (score, embedding)

            new (tuple): a tuple (score, embedding)

        """
        (oldscore, oldthing) = old
        (newscore, newthing) = new

        def measure(chains):
            return sum(map(len, chains))

        if oldscore is None:
            return True
        if newscore is None:
            return False
        if len(newthing):
            if not len(oldthing):
                return True
            elif isinstance(newthing, tuple):
                newlengths = sum(map(measure, newthing))
                oldlengths = sum(map(measure, oldthing))
                return newlengths < oldlengths
            else:
                return measure(newthing) < measure(oldthing)
        else:
            return False

    def _find_evil(self):
        """A utility function that computes a list of missing couplers which
        should connect two working qubits in the same cell.  The presence
        of (a nonconstant number of) these breaks the polynomial-time
        claim for our algorithm.  Note: we're only actually hurt by missing
        intercell couplers.
        """
        M, N, L = self.M, self.N, self.L
        proc = self._proc0
        evil = []
        cells = [(x, y) for x in range(M) for y in range(N)]
        spots = [(u, v) for u in range(L) for v in range(L)]
        for x, y in cells:
            for u, v in spots:
                p = (x, y, 0, u)
                q = (x, y, 1, v)
                if p in proc and q in proc and p not in proc[q]:
                    evil.append((p, q))
        self._evil = evil

    def tightestNativeClique(self, n):
        """Returns the native clique embedding with the shortest chains and
        at least ``n`` completely connected chains, or ``None`` on failure.

        INPUTS:
            n: size of clique to return

        OUTPUT:
            clique (list): a list containing lists of qubits, each associated
            to a chain.  These lists of qubits are carefully
            arranged so that

            >>> [zip(chain,chain[1:]) for chain in clique]

            is a list of valid couplers.  If no ``n``-clique is
            available on the processor, this is ``None.``

        Note: this function does not return a uniformly random choice,
        as the nativeCliqueEmbed function only guarantees uniform choice
        of maximum cliques.
        """
        def f(x):
            return x.tightestNativeClique(n)
        objective = self._objective_qubitcount
        return self._translate(self._map_to_processors(f, objective))

    def largestNativeClique(self, max_chain_length=None):
        """Returns the largest native clique embedding we can find on the
        processor, with the shortest chainlength possible (for that clique
        size).  If possible, returns a uniform choice among all largest
        cliques.

        INPUTS:
            max_chain_length (int): longest chain length to consider or ``None`` if chain
            lengths are allowed to be unbounded.  (default: ``None``)

        OUTPUT:
            clique (list): a list containing lists of qubits, each associated to a
            chain.  These lists of qubits are carefully arranged so that

            >>> [zip(chain,chain[1:]) for chain in clique]

            is a list of valid couplers.

        Note: this fails to return a uniform choice if there are broken
        intra-cell couplers between working qubits. (the choice is
        uniform on a particular subprocessor)
        """
        def f(x):
            return x.largestNativeClique(max_chain_length=max_chain_length)
        objective = self._objective_bestscore
        return self._translate(self._map_to_processors(f, objective))

    def nativeCliqueEmbed(self, width):
        """Compute a maximum-sized native clique embedding in an induced
        subgraph of chimera with chainsize ``width+1``.  If possible,
        returns a uniform choice among all largest cliques.

        INPUTS:
            width: width of the squares to search, also `chainlength-1`

        OUTPUT:
            clique: a list containing lists of qubits, each associated
            to a chain.  These lists of qubits are carefully
            arranged so that

            >>> [zip(chain,chain[1:]) for chain in clique]

            is a list of valid couplers.

        Note: this fails to return a uniform choice if there are broken
        intra-cell couplers between working qubits. (the choice is
        uniform on a particular subprocessor)
        """
        def f(x):
            return x.nativeCliqueEmbed(width)
        objective = self._objective_bestscore
        return self._translate(self._map_to_processors(f, objective))

    def largestNativeBiClique(self, chain_imbalance=0, max_chain_length=None):
        """Returns a native embedding for the complete bipartite graph :math:`K_{n,m}`
        for `n <= m`; where `n` is as large as possible and `m` is as large as
        possible subject to `n`.  The native embedding of a complete bipartite
        graph is a set of horizontally-aligned qubits connected in lines
        together with an equal-sized set of vertically-aligned qubits
        connected in lines.

        INPUTS:
            chain_imbalance: how big of a difference to allow between the
            chain lengths on the two sides of the bipartition. If ``None``,
            then we allow an arbitrary imbalance.  (default: ``0``)

            max_chain_length: longest chain length to consider or None if chain
            lengths are allowed to be unbounded.  (default: ``None``)

        OUTPUT:
            embedding (tuple): a tuple of two lists containing lists of qubits.
            If ``embedding = (A_side, B_side)``, the lists found in ``A_side`` and
            ``B_side`` are chains of qubits. These lists of qubits are arranged so that

            >>> [zip(chain,chain[1:]) for chain in A_side]

            and

            >>> [zip(chain,chain[1:]) for chain in B_side]

            are lists of valid couplers.
        """
        def f(x):
            return x.largestNativeBiClique(chain_imbalance=chain_imbalance,
                                           max_chain_length=max_chain_length)
        objective = self._objective_bestscore
        emb = self._map_to_processors(f, objective)
        return self._translate_partitioned(emb)

    def tightestNativeBiClique(self, n, m=None, chain_imbalance=0, max_chain_length=None):
        """Returns a native embedding for the complete bipartite graph :math:`K_{n,m}`
        with the shortest chains possible.  The native embedding of a complete
        bipartite graph is a set of horizontally-aligned qubits connected
        in lines together with an equal-sized set of vertically-aligned qubits
        connected in lines.

        INPUTS:
            n (int): target size for one side of the bipartition

            m (int): target size for the other side of the bipartition, or
            None for the special case ``m=n`` (default: ``None``)

            chain_imbalance (int): how big of a difference to allow between the
            chain lengths on the two sides of the bipartition.  If
            ``None``, we allow an arbitrary imbalance. (default: ``0``)

            max_chain_length (int): longest chain length to consider or ``None`` if
            chain lengths are allowed to be unbounded.  (default: ``None``)

        OUTPUT:
            embedding (tuple): a tuple of two lists containing lists of qubits.
            If ``embedding = (A_side, B_side)``, the lists found in ```A_side``` and
            ``B_side`` are chains of qubits.
            These lists of qubits are arranged so that

            >>> [zip(chain,chain[1:]) for chain in A_side]

            and

            >>> [zip(chain,chain[1:]) for chain in B_side]

            are lists of valid couplers.  If no :math:`K_{n,m}` is available on
            the processor, this is ``None``.
        """
        def f(x):
            return x.tightestNativeBiClique(n, m=m,
                                            chain_imbalance=chain_imbalance,
                                            max_chain_length=max_chain_length)
        objective = self._objective_qubitcount
        emb = self._map_to_processors(f, objective)
        return self._translate_partitioned(emb)

    def _translate_partitioned(self, embedding):
        """Translates a partitioned embedding back to linear coordinates if
        necessary.  This is useful, for example, in biclique embeddings.  A
        partitioned embedding is a tuple of lists of qubits.
        """
        if embedding is None:
            return None
        return tuple(map(self._translate, embedding))

    def _translate(self, embedding):
        "Translates an embedding back to linear coordinates if necessary."
        if embedding is None:
            return None
        if not self._linear:
            return embedding
        return [_bulk_to_linear(self.M, self.N, self.L, chain) for chain in embedding]


def _bulk_to_linear(M, N, L, qubits):
    "Converts a list of chimera coordinates to linear indices."
    return [2 * L * N * x + 2 * L * y + L * u + k for x, y, u, k in qubits]


def _to_linear(M, N, L, q):
    "Converts a qubit in chimera coordinates to its linear index."
    (x, y, u, k) = q
    return 2 * L * N * x + 2 * L * y + L * u + k


def _bulk_to_chimera(M, N, L, qubits):
    "Converts a list of linear indices to chimera coordinates."
    return [(q // N // L // 2, (q // L // 2) % N, (q // L) % 2, q % L) for q in qubits]


def _to_chimera(M, N, L, q):
    "Converts a qubit's linear index to chimera coordinates."
    return (q // N // L // 2, (q // L // 2) % N, (q // L) % 2, q % L)


def _chimera_neighbors(M, N, L, q):
    "Returns a list of neighbors of (x,y,u,k) in a perfect :math:`C_{M,N,L}`"
    (x, y, u, k) = q
    n = [(x, y, 1 - u, l) for l in range(L)]
    if u == 0:
        if x:
            n.append((x - 1, y, u, k))
        if x < M - 1:
            n.append((x + 1, y, u, k))
    else:
        if y:
            n.append((x, y - 1, u, k))
        if y < N - 1:
            n.append((x, y + 1, u, k))
    return n


def random_processor(M, N, L, qubit_yield, num_evil=0):
    """A utility function that generates a random :math:`C_{M,N,L}` missing some
    percentage of its qubits.

    INPUTS:
        M,N,L: the chimera parameters
        qubit_yield: ratio (0 <= qubit_yield <= 1) of #{qubits}/(2*M*N*L)
        num_evil: number of broken in-cell couplers between working qubits

    OUTPUT:
        proc (:class:`processor`): a :class:`processor` instance with a random
            collection of qubits and couplers as specified
    """
    # replacement for lambda in edge filter below that works with bot h
    def edge_filter(pq):
        # we have to unpack the (p,q) edge
        p, q = pq
        return q in qubits and p < q

    qubits = [(x, y, u, k) for x in range(M) for y in range(N) for u in [0, 1] for k in range(L)]
    nqubits = len(qubits)
    qubits = set(sample(qubits, int(nqubits * qubit_yield)))
    edges = ((p, q) for p in qubits for q in _chimera_neighbors(M, N, L, p))
    edges = list(filter(edge_filter, edges))
    possibly_evil_edges = [(p, q) for p, q in edges if p[:2] == q[:2]]
    num_evil = min(num_evil, len(possibly_evil_edges))
    evil_edges = sample(possibly_evil_edges, num_evil)
    return processor(set(edges) - set(evil_edges), M=M, N=N, L=L, linear=False)
