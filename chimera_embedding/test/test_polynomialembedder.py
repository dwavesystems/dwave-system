#
#Copyright 2016 D-Wave Systems Inc.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
#


from chimera_embedding import polynomialembedder
from polynomialembedder import processor, _bulk_to_linear, _to_linear, _bulk_to_chimera, _to_chimera, _chimera_neighbors, random_processor
from collections import defaultdict
from nose import with_setup


def setup_eden_tests():
    global eden_proc

    M = 12
    N = 7
    L = 4

    eden_qubits = [(x, y, u, k) for x in xrange(M)
                   for y in xrange(N) for u in (0, 1) for k in xrange(L)]

    # one K_12, contains K_{8,8}
    Cliq1 = {(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (0, 2)}

    # another K_12
    Cliq2 = {(1, 4), (0, 5), (0, 6), (1, 5), (1, 6), (2, 5)}

    # K_{12,16} after deleting all horizontal qubits in alternating rows
    Rect1 = {(x, y) for x in range(4, 7) for y in range(7)}
    kill1 = {}

    # deleting those alternating rows
    Rect2 = {(x, y) for x in range(4, 7) for y in range(1, 7, 2)}
    kill2 = {(0, 0), (0, 1), (0, 2), (0, 3)}

    # K_{12,12} after deleting all u=1,k=3
    Rect3 = {(x, y) for x in range(8, 12) for y in range(3)}
    kill3 = {(1, 3)}

    # K_{8,16}
    Rect4 = {(x, y) for x in range(8, 12) for y in range(4, 6)}
    kill4 = {}

    XYFilter = Cliq1 | Cliq2 | Rect1 | Rect2 | Rect3 | Rect4
    eden_qubits = filter(lambda (x, y, u, k): (x, y) in XYFilter, eden_qubits)

    for rect, kill in zip((Rect1, Rect2, Rect3, Rect4), (kill1, kill2, kill3, kill4)):
        killf = lambda (x, y, u, k): (
            ((x, y) not in rect) or ((u, k) not in kill))
        eden_qubits = filter(killf, eden_qubits)

    eden_qubits = set(eden_qubits)
    eden_couplers = [(q, n) for q in eden_qubits for n in set(
        _chimera_neighbors(M, N, L, q)) & eden_qubits]
    eden_couplers.extend(((x, y, 0, 0), (x, y, 1, 0))
                         for x, y in ((2, 2), (2, 3), (3, 3)))

    eden_couplers = [_bulk_to_linear(M, N, L, c) for c in eden_couplers]
    eden_proc = processor(eden_couplers, M=M, N=N, L=L)
    eden_proc._linear = False


def setup_eden2_tests():
    global eden2_proc

    M = 6
    N = 6
    L = 4

    eden_qubits = [(x, y, u, k) for x in xrange(M)
                   for y in xrange(N) for u in (0, 1) for k in xrange(L)]
    dead_qubits = [(3, 3, 1, 2), (4, 3, 0, 0), (4, 3, 1, 0), (5, 3, 1, 0), (3, 4, 0, 1), (3, 4, 1, 0),
                   (3, 4, 1, 2), (4, 4, 0, 0), (4, 4, 1, 1), (3, 5, 0, 0), (3, 5, 0, 1), (3, 5, 0, 2)]
    XYFilter = [(0, 2), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5),
                (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (5, 0), (5, 1), (5, 2), (5, 3)]
    eden_qubits = filter(lambda (x, y, u, k): (
        x, y) in XYFilter, set(eden_qubits) - set(dead_qubits))
    eden_qubits = set(eden_qubits)
    eden_couplers = [(q, n) for q in eden_qubits for n in set(
        _chimera_neighbors(M, N, L, q)) & eden_qubits]
    eden_couplers = [_bulk_to_linear(M, N, L, c) for c in eden_couplers]
    eden2_proc = processor(eden_couplers, M=M)
    eden2_proc._linear = False


def verify_chains(Proc, emb):
    proc = Proc._proc0
    for i, e in enumerate(emb):
        for u in e:
            assert u in proc, "qubit %s doesn't exist" % (u,)
        for u, v in zip(e, e[1:]):
            assert (u in proc[v]) and (
                v in proc[u]), "chain is not a connected subgraph"
        E = set(e)
        for f in emb[:i]:
            assert not any(v in E for v in f), "qubit appears in two chains!"


def verify_clique(Proc, emb, size, length):
    verify_chains(Proc, emb)
    proc = Proc._proc0

    assert len(emb) == size, "expected a K_{%s}" % size
    for i, e in enumerate(emb):
        assert len(e) == length, "expected chain lengths to be %s but found a chain of length %s" % (
            length, len(e))
        for f in emb[:i]:
            assert any(
                u in proc[v] for u in e for v in f), "no coupler between %s and %s" % (e, f)


def verify_biclique(Proc, emb, num_A, num_B, len_A, len_B):
    proc = Proc._proc0

    A, B = emb
    verify_chains(Proc, A)
    verify_chains(Proc, B)

    assert (len(A) == num_A) and (len(
        B) == num_B), "expected a K_{%s,%s} but got a K_{%s,%s}" % (num_A, num_B, len(A), len(B))
    assert all(len(e) == len_A for e in A) and all(
        len(e) == len_B for e in B), "expected chain lengths %s and %s" % (len_A, len_B)

    for e in A:
        for f in B:
            assert any(
                u in proc[v] for u in e for v in f), "no coupler between %s and %s" % (e, f)


@with_setup(setup_eden_tests)
def clique_12_uniformity_test():
    global eden_proc
    cliques = defaultdict(int)
    runs = 100
    for _ in range(runs):
        emb = eden_proc.largestNativeClique(max_chain_length=4)
        cliques[frozenset(map(tuple, emb))] += 1
    assert len(cliques) == 2, "should have found exactly 2 cliques."
    A, B = cliques.keys()
    verify_clique(eden_proc, list(A), 12, 4)
    verify_clique(eden_proc, list(B), 12, 4)

    a, b = cliques.values()
    assert abs(a - b) < 3 * (runs**.5), "%s and %s should be roughly equal.  This test has about a 2%% chance of failure due to random chance." % (a, b)
    # shoddy statistics:
    # >>> from random import randint
    # >>> M = 100
    # >>> N = 1000000
    # >>> f = sum(abs(2*sum(randint(0,1) for _ in xrange(M))-M)>3*(M**.5) for _ in range(N))/float(N)
    # >>> print "%.2f chance of failure"%f


@with_setup(setup_eden_tests)
def clique_12_test():
    global eden_proc
    emb = eden_proc.largestNativeClique()
    verify_clique(eden_proc, emb, 12, 4)


@with_setup(setup_eden2_tests)
def clique_18_test():
    global eden2_proc
    emb = eden2_proc.largestNativeClique()
    verify_clique(eden2_proc, emb, 18, 7)
    emb = eden2_proc.nativeCliqueEmbed(6)
    verify_clique(eden2_proc, emb, 18, 7)


@with_setup(setup_eden2_tests)
def clique_17_test():
    global eden2_proc
    emb = eden2_proc.largestNativeClique(max_chain_length=6)
    verify_clique(eden2_proc, emb, 17, 6)
    emb = eden2_proc.nativeCliqueEmbed(5)
    verify_clique(eden2_proc, emb, 17, 6)
    emb = eden2_proc.tightestNativeClique(17)
    verify_clique(eden2_proc, emb, 17, 6)


def clique_15_test():
    global eden2_proc
    emb = eden2_proc.largestNativeClique(max_chain_length=5)
    verify_clique(eden2_proc, emb, 15, 5)
    emb = eden2_proc.nativeCliqueEmbed(4)
    verify_clique(eden2_proc, emb, 15, 5)
    emb = eden2_proc.tightestNativeClique(15)
    verify_clique(eden2_proc, emb, 15, 5)


def clique_toobig_test():
    global eden2_proc
    emb = eden2_proc.tightestNativeClique(20)
    assert emb == []


@with_setup(setup_eden_tests)
def biclique_balanced_test():
    global eden_proc
    cliques = defaultdict(int)
    emb = eden_proc.largestNativeBiClique(chain_imbalance=0)
    verify_biclique(eden_proc, emb, 12, 9, 3, 3)


@with_setup(setup_eden_tests)
def biclique_balanced_length2_test():
    global eden_proc
    cliques = defaultdict(int)
    emb = eden_proc.largestNativeBiClique(
        chain_imbalance=0, max_chain_length=2)
    verify_biclique(eden_proc, emb, 8, 8, 2, 2)


@with_setup(setup_eden_tests)
def biclique_balanced_length3_test():
    global eden_proc
    cliques = defaultdict(int)
    emb = eden_proc.largestNativeBiClique(
        chain_imbalance=None, max_chain_length=3)
    verify_biclique(eden_proc, emb, 12, 9, 3, 3)


@with_setup(setup_eden_tests)
def biclique_largest_test():
    global eden_proc
    cliques = defaultdict(int)
    emb = eden_proc.largestNativeBiClique(chain_imbalance=None)
    verify_biclique(eden_proc, emb, 16, 12, 3, 7)


@with_setup(setup_eden_tests)
def biclique_tightest_test():
    global eden_proc
    cliques = defaultdict(int)
    emb = eden_proc.tightestNativeBiClique(8, 12, chain_imbalance=None)
    verify_biclique(eden_proc, emb, 8, 12, 3, 2)


@with_setup(setup_eden_tests)
def biclique_tightest88_test():
    global eden_proc
    cliques = defaultdict(int)
    emb = eden_proc.tightestNativeBiClique(8, chain_imbalance=None)
    verify_biclique(eden_proc, emb, 8, 8, 2, 2)


@with_setup(setup_eden_tests)
def biclique_tightest_test():
    global eden_proc
    cliques = defaultdict(int)
    emb = eden_proc.tightestNativeBiClique(8, 12, chain_imbalance=None)
    verify_biclique(eden_proc, emb, 8, 12, 3, 2)


def evil_K_2_2_test():
    couplers = [((0, 0, 0, i), (1, 0, 0, i)) for i in range(4)]
    couplers += [((0, 0, 1, i), (0, 1, 1, i)) for i in range(4)]
    couplers += [((0, 0, 0, 0), (0, 0, 1, 0))]
    proc = processor(couplers, M=2, N=2, L=4, linear=False, proc_limit=2**16)

    emb = proc.tightestNativeBiClique(1, 1)
    verify_biclique(proc, emb, 1, 1, 1, 1)

    for _ in range(100):  # this should be plenty
        proc = processor(couplers, M=2, N=2, L=4, linear=False, proc_limit=4)
        emb = proc.tightestNativeBiClique(1, 1)
        if emb is not None:
            break
    verify_biclique(proc, emb, 1, 1, 1, 1)


def objectives_test():
    couplers = [((0, 0, 0, i), (1, 0, 0, i)) for i in range(4)]
    couplers += [((0, 0, 1, i), (0, 1, 1, i)) for i in range(4)]
    couplers += [((0, 0, 0, 0), (0, 0, 1, 0))]
    proc = processor(couplers, M=2, N=2, L=4, linear=False, proc_limit=2**16)

    empty = proc._subprocessor(proc._proc0) #an eden_processor with all qubits disabled
    emb = proc.tightestNativeBiClique(0,0)
    proc._processors = [empty] + proc._processors + [empty] 
    verify_biclique(proc, emb, 0, 0, 0, 0)

    empty.largestNativeBiClique = lambda *a,**k:(None,None)
    emb = proc.largestNativeBiClique()
    verify_biclique(proc, emb, 1, 1, 1, 1)


def proclimit_cornercase_test():
    couplers  = [((0, y, 1, i), (0, y + 1, 1, i)) for y in xrange(2) for i in xrange(4)]
    couplers += [((0, y, 1, i), (0, y, 0, j)) for y in xrange(3) for i in xrange(4) for j in xrange(4) if i != 0 or j != y]
    emb = None
    count = 0
    while emb is None and count < 100:
        proc = processor(couplers, M=1, N=3, L=4, linear=False, proc_limit=2000)
        emb = proc.tightestNativeBiClique(3, 9, chain_imbalance=None)
        count += 1
    verify_biclique(proc, emb, 3, 9, 3, 1)


def linear_embedding_test():
    proc = random_processor(1,1,2,1)
    proc._linear = True
    emb = proc.largestNativeClique()
    assert len(emb) == 2
    emb0 = sorted(emb[0])
    assert emb0 == [0,2] or emb0 == [1,3]
    emb1 = sorted(emb[1])
    assert emb1 == [0,2] or emb1 == [1,3]

def qubits_and_couplers_test():
    M = N = L = 2
    qubits = {(x,y,u,k) for x in xrange(M) for y in xrange(N) for u in xrange(2) for k in xrange(L)}
    couplers = [(p,q) for q in qubits for p in _chimera_neighbors(M,N,L,q)]
    proc = processor(couplers,M=M,N=N,L=L,linear=False)._proc0
    for q in proc:
        assert proc[q] == set(_chimera_neighbors(M,N,L,q)), "Neighborhood is wrong!"
        qubits.remove(q)
    assert not qubits, "qubits are missing from proc"

def biclique_cache_test():
    proc = random_processor(2,2,2,1)._proc0
    proc._compute_biclique_sizes()
    proc._biclique_size[None] = None
    proc._compute_biclique_sizes()
    assert None in proc._biclique_size
    proc._compute_biclique_sizes(recompute=True)
    assert None not in proc._biclique_size


def random_bundle_test():
    cliques = defaultdict(int)
    couplers = [((0, 0, 0, 0), (0, 0, 1, 0)), ((0, 0, 0, 0), (0, 0, 1, 1))]
    proc = processor(couplers, M=1, N=1, L=2, linear=False)
    proc0 = proc._proc0
    proc0.random_bundles = True

    runs = 100
    for _ in range(runs):
        emb = proc0.largestNativeClique(max_chain_length=2)[1]
        cliques[frozenset(map(tuple, emb))] += 1
    assert len(cliques) == 2, "should have found exactly 2 cliques, got %s" % (len(cliques))
    A, B = cliques.keys()
    verify_clique(eden_proc, list(A), 1, 2)
    verify_clique(eden_proc, list(B), 1, 2)

    a, b = cliques.values()
    assert abs(a - b) < 3 * (runs**.5), "%s and %s should be roughly equal.  This test has about a 2%% chance of failure due to random chance." % (a, b)
