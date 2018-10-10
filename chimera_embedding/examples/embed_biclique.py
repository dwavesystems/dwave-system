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
#~ .. _embed-biclique-example:
#~
#~ Finding Native Biclique Embeddings
#~ **********************************
#~
#~ This is an example of how to use the :py:mod:`chimera_embedding` module to
#~ find a maximum-sized native embeddings of complete bipartite graphs, also
#~ known as bicliques.  One side-effect of this example is to produce a file
#~ which describes the embedding of a biclique.  That file will be used in the
#~ :ref:`solve-biclique-example` example.
#~
#~ Constructing a :class:`processor` instance
#~ ------------------------------------------
#~
#~ The :class:`processor` class holds the implementation of our embedding
#~ algorithms.  This section is devoted to gathering the information necessary
#~ to start finding embeddings.
#~
#~ Some imports
#~ ^^^^^^^^^^^^
#~
#~ This is pretty self-explanatory; we'll need these things later on.

from dwave_sapi2.util import get_hardware_adjacency
from dwave_sapi2.local import local_connection
from chimera_embedding import processor

#~ Connecting to the solver
#~ ^^^^^^^^^^^^^^^^^^^^^^^^
#~
#~ Here, we're going to use a local solver because various users will have
#~ access to different systems.  When you extend this code, you'll want to
#~ modify it to use a remote connection with your API key, and connect to
#~ the appropriate solver.
#~

solver_name = "c4-sw_sample"
solver = local_connection.get_solver(solver_name)
# or, if you're using a remote solver, uncomment this block and put your
# authentication information here
# from dwave_sapi2.core import remote
# sapi_url = 'https://dw2x.dwavesys.com/sapi'
# sapi_token = 'ENTER SAPI TOKEN'
# solver_name = 'DW2X'
# connection = remote.RemoteConnection(sapi_url, sapi_token)
# solver = connection.get_solver(solver_name)

#~ Getting the hardware structure
#~ ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#~
#~ The ``c4-sw_sample`` solver is a :math:`C_{4,4,4}` Chimera graph, and does
#~ not (at the time of writing) populate its Chimera parameters.  More recent
#~ hardware solvers do populate these properties.  So for this example, we
#~ have included default values to accomodate the software solver.  Below,
#~ we ask the solver (assuming that it has a Chimera structure, which may not
#~ be true of future solvers) for the parameters :math:`M`, :math:`N` and :math:`L`
#~ that define the ideal hardware graph :math:`C_{M,N,L}`.

params = solver.properties.get('parameters', {})
M = params.get('M', 4)
N = params.get('N', 4)
L = params.get('L', 4)

#~
#~ Additionally, we need the list of active couplers.  In hardware solvers,
#~ certain qubits and couplers will be disabled.  We need to know precisely
#~ which elements are available for our use.  We'll fetch the set of available
#~ couplers, and infer which qubits we can use.

hardware_adj = get_hardware_adjacency(solver)

#~ This is just a set of tuples ``(p, q)`` of qubit labels.  Using
#~ the ``c4-sw_optimize`` solver, this is simply a :math:`C_{4,4,4}` Chimera
#~ graph.  With hardware solvers, this will be something more interesting.
#~
#~ Making the instance
#~ ^^^^^^^^^^^^^^^^^^^
#~
#~ Now, we've got enough information to run the native clique embedder.  We
#~ instantiate a :class:`processor` first.  The native clique embedding
#~ algorithm uses dynamic programming, and populates a cache which can be used
#~ to produce several embeddings for the same processor.

embedder = processor(hardware_adj, M=M, N=N, L=L)

#~ Finding a maximum-sized native biclique embedding
#~ -------------------------------------------------
#~
#~ The :meth:`largestNativeBiClique()` function tries to find a maximum-sized
#~ native biclique embedding.  Since a biclique has two components, that
#~ question is not well-defined (is a :math:`K_{1,100}` bigger than a
#~ :math:`K_{50,50}`?) so for this method, we have somewhat arbitrarily
#~ decided to first maximize the smaller part and then the larger part.  The
#~ :class:`processor` class provides two functions which can be used to find
#~ native biclique embeddings with certain requirements, but we start with a
#~ plain function call.
#~

embedding = embedder.largestNativeBiClique()

#~ Now, ``embedding`` is a tuple of two lists of chains.   All of the chains
#~ in one list share a coupler with every chain of the other list.  Let's see
#~ what we've found.


print "Found a native biclique embedding of size (%s, %s)"%(len(embedding[0]),len(embedding[1]))
lengthcounts = {}
for chain in embedding[0]+embedding[1]:
    if len(chain) in lengthcounts:
        lengthcounts[len(chain)]+= 1
    else:
        lengthcounts[len(chain)] = 1
for length in sorted(lengthcounts):
    print "    %s chains of length %s"%(lengthcounts[length], length)


#~ Writing out the biclique embedding
#~ ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#~
#~ We need to produce an embedding file for use in the
#~ :ref:`solve-biclique-example` example, so let's do that here.  We'll use
#~ the simplest file format possible: each line will correspond to a chain of
#~ qubits, represented by a space-delimited sequence of qubit labels
#~ (integers).  We'll place a blank line between the two sides of the
#~ bipartition.
#~

embedding_file = open("embedded_biclique_largest", 'w')
embedding_out = embedding[0]
embedding_out+= [[]] #this will become a blank line
embedding_out+= embedding[1]
for chain in embedding_out:
    chainstrings = map(str, chain)
    line = " ".join(chainstrings)
    embedding_file.write(line + "\n")
    print line


#~ Finding a maximum-sized native biclique embedding with bounded chainlengths
#~ ---------------------------------------------------------------------------
#~
#~ The :meth:`largestNativeBiClique` method provides some additional arguments
#~ to allow a user to, for example, bound the chainlength of the embeddings
#~ under consideration.  Of course, more details can be found at
#~ :ref:`chimera_embedding`.

embedding = embedder.largestNativeBiClique(max_chain_length=3)

#~ Let's see what we got.

lengthcounts = {}
for chain in embedding[0]+embedding[1]:
    if len(chain) in lengthcounts:
        lengthcounts[len(chain)]+= 1
    else:
        lengthcounts[len(chain)] = 1
for length in sorted(lengthcounts):
    print "    %s chains of length %s"%(lengthcounts[length], length)

#~
#~ Finding a native embedding of :math:`K_{a,b}`
#~ ---------------------------------------------
#~
#~ Where it may be interesting to have an maximum-sized native biclique
#~ embedding, not every problem we try to solve will be of that size.  Since
#~ solver performance decreases with chain length, we want to minimize the
#~ chainlength for a given problem size.  Let's find an embedding of
#~ :math:`K_{6,9}` that uses the shortest chains possible (subject to all
#~ chains having the same size).
#~

embedding = embedder.tightestNativeBiClique(6,9)

#~ Let's see what we got.

lengthcounts = {}
for chain in embedding[0]+embedding[1]:
    if len(chain) in lengthcounts:
        lengthcounts[len(chain)]+= 1
    else:
        lengthcounts[len(chain)] = 1
for length in sorted(lengthcounts):
    print "    %s chains of length %s"%(lengthcounts[length], length)

#~ By default, :meth:`tightestNativeBiClique` and
#~ :meth:`largestNativeBiClique` only consider embeddings that have perfectly
#~ uniform chainlengths.  To override that default behavior, we specify a
#~ nonzero value for ``chain_imbalance``.  By setting
#~ ``chain_imbalance = None`` we can turn that off entirely, and we can allow a
#~ limited imbalance by specifying a nonzero integer value.

embedding = embedder.tightestNativeBiClique(6,9, chain_imbalance = None)
lengthcounts = {}
for chain in embedding[0]+embedding[1]:
    if len(chain) in lengthcounts:
        lengthcounts[len(chain)]+= 1
    else:
        lengthcounts[len(chain)] = 1
for length in sorted(lengthcounts):
    print "    %s chains of length %s"%(lengthcounts[length], length)
