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
#~ .. _solve-biclique-example:
#~
#~ Solving Biclique Problems From Existing Embeddings
#~ **************************************************
#~
#~ This is an example of how to use python to solve a complete bipartite problem
#~ using an existing embedding.  We assume that the example :ref:`embed-biclique-example`
#~ has been executed in the current directory, and as a result, the file
#~ ``embedded_biclique_largest`` exists in our directory.
#~
#~ The python code in this document can be found in the ``examples`` directory,
#~ along with c and matlab implementations of the same.
#~
#~ Loading an embedding
#~ --------------------
#~
#~ In this section, we'll show how to collect the necessary information to use
#~ an existing clique embedding.
#~
#~
#~ Some imports
#~ ^^^^^^^^^^^^
#~
#~ This is pretty self-explanatory; we'll need these things later on.

from random import uniform
from dwave_sapi2.util import get_hardware_adjacency
from dwave_sapi2.embedding import embed_problem, unembed_answer
from dwave_sapi2.core import solve_ising
from dwave_sapi2.local import local_connection

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

#~
#~ Now that we're connected, let's pull down the adjacency information from
#~ the solver.  This is just a set of tuples ``(p, q)`` of qubit labels.  Using
#~ the ``c4-sw_optimize`` solver, this is simply a :math:`C_{4,4,4}` Chimera
#~ graph.  If you're using a hardware solver, this will be something more
#~ interesting.
#~

hardware_adj = get_hardware_adjacency(solver)


#~ Reading the file in
#~ ^^^^^^^^^^^^^^^^^^^
#~
#~ Here, we're going to read the embedding file.  As explained in the previous
#~ example, we've used the simplest file format possible; a text file where
#~ every line corresponds to a chain of qubits, represented by a space-
#~ delimited list of qubit labels.  The file also contains a single blank line
#~ separating the two sides of the partition.
#~

embedding_file = open("embedded_biclique_largest")
found_blank = False
embedding = [[], []]
for line in embedding_file:
    line = line.strip()  # remove trailing newline if present
    if not line:
        found_blank = True
        continue
    qubit_strings = line.split()
    chain = map(int, qubit_strings)
    if found_blank:
        embedding[1].append(chain)
    else:
        embedding[0].append(chain)


#~
#~ Now we've got the embedding, let's see how big it is.
#~

n1 = len(embedding[0])
n2 = len(embedding[1])
print "Loaded a biclique embedding with partition sizes %s and %s"%(n1,n2)
lengthcounts = {}
for chain in embedding:
    if len(chain) in lengthcounts:
        lengthcounts[len(chain)]+= 1
    else:
        lengthcounts[len(chain)] = 1
for length in sorted(lengthcounts):
    print "    %s chains of length %s"%(lengthcounts[length], length)


#~ Solving problems with the embedding
#~ -----------------------------------
#~
#~ In this section, we'll show how to generate a random problem and use our
#~ embedding to solve that problem.  The problems we solve in this example are
#~ not necessarily interesting, but they're conveniently easy to generate for
#~ the sake of illustration.
#~
#~ Problem generation
#~ ^^^^^^^^^^^^^^^^^^
#~
#~ Now, let's generate our bipartite problem.  Since this example is meant to illustrate
#~ the use of embeddings, we don't attempt to solve an interesting problem.
#~ Instead, we'll just generate a random Ising spin glass problem.
#~
#~ We're going to define two variables, ``h`` and ``J`` which describe the
#~ hamiltonian
#~
#~ :math:`\sum_{i=0}^{n_1+n_2-1} h_i x_i + \sum_{i=0}^{n_1-1}\sum_{j=n_1}^{n_1+n_2-1} J_{i,j} x_i x_j`
#~
#~ where every :math:`h_i` and :math:`J_{i,j}` is chosen uniformly between -1 and 1.
#~

h = [uniform(-1, 1) for i in xrange(n1+n2)]
J = {(i, j): uniform(-1, 1) for i in xrange(n1) for j in xrange(n1,n1+n2)}

#~
#~ Embedding the problem
#~ ^^^^^^^^^^^^^^^^^^^^^
#~
#~ Now we've got the two ingredients in place; an embedding and a problem to
#~ embed.  The :func:`embed_problem` is just what we need to combine these two
#~ ingredients.
#~
#~ Embedding a problem is accomplished by
#~
#~ * if two qubits are in the same chain, and adjacent in `hardware_adj`, then
#~   the coupler between them is set to -1.
#~ * if two qubits are in different chains whose variables interact (all
#~   pairs of variables interact in this particular example) then their
#~   interaction is distributed among the couplers which have ends in both
#~   chains. (that is, the sum of the coupler values is equal to the
#~   corresponding :math:`J_{i,j}`)
#~ * the variable biases are distributed among the qubit biases,
#~   that is, the qubit biases in the chain corresponding to variable :math:`i` sum
#~   to :math:`h_i`.
#~

flat_embedding = embedding[0] + embedding[1]
(emb_h, prob_J, chain_J, new_emb) = embed_problem(h, J, flat_embedding, hardware_adj)

#~
#~ Setting the chain strength
#~ --------------------------
#~
#~ We're nearly ready to solve our problem, but first we need to combine the
#~ problem interactions with the chain interactions.  Above, we said that the
#~ couplers between adjacent qubits in a chain are set to -1, but the story
#~ is a little more interesting than that.
#~
#~ The matter of chain interactions is still poorly-understood.  One expects
#~ better chain performance (that is, for qubits in chains to have the same
#~ spin) with a higher chain strength, but if the chain strength is set too
#~ high, then the problem interactions will suffer a loss of precision. A
#~ plausible heuristic is to set the chain strength to :math:`\sqrt{n}` to
#~ balance chain breaks against problem interactions.
#~
#~ For the sake of illustration, let's take a look at what happens when the
#~ chain strength is too low.
#~

chain_strength = .01
for (p,q) in chain_J:
    prob_J[p,q] = -chain_strength

#~
#~ Solving the problem
#~ ^^^^^^^^^^^^^^^^^^^
#~
#~ Now that we've got our chain interactions set, let's send the problem off to
#~ the solver.
#~

result = solve_ising(solver, emb_h, prob_J, num_reads=1)

#~
#~ Ideally, the spin in every chain would be the same.  Let's see how we did.
#~

res0 = result['solutions'][0]
num_broken = 0
for chain in flat_embedding:
    spins = ['+' if res0[q] == 1 else '-' for q in chain]
    if len(set(spins)) != 1:
        num_broken += 1
        print "broken chain:", "".join(spins)
print "%s of %s chains were broken"%(num_broken, len(flat_embedding))

#~
#~ We intentionally set the chain strength to a weak value to show that
#~ chains can and will break. So now, let's run the problem with stronger
#~ chains.  For the sake of simplicity, we'll go with :math:`\sqrt{n}` as
#~ recommended.
#~
#~ Note that we're setting ``num_reads=1`` to get a single sample back
#~ from the solver.  That's not an efficient way to use the hardware
#~ since, for the time being, it takes significantly longer to program
#~ the chip with a problem than it does to do a single anneal.
#~

chain_strength = (n1+n2)**.5
for (p,q) in chain_J:
    prob_J[p,q] = -chain_strength

result = solve_ising(solver, emb_h, prob_J, num_reads=1)
res0 = result['solutions'][0]
num_broken = 0
for chain in flat_embedding:
    spins = ['+' if res0[q] == 1 else '-' for q in chain]
    if len(set(spins)) != 1:
        num_broken += 1
        print "broken chain:", "".join(spins)
print "%s of %s chains were broken"%(num_broken, len(flat_embedding))

#~
#~ Now that's much better.  Of course, we're using a software solver for this
#~ example, so none of the chains are broken.  Let's break some intentionally,
#~ to make things look a little more realistic.
#~

from random import randint
if solver_name[:5] == 'c4-sw': #don't spoil solutions provided by hardware
    num_broken = 0
    for chain in flat_embedding:
        for q in chain:
            if randint(1,len(chain)) == 1:
                res0[q] = -res0[q]
        spins = ['+' if res0[q] == 1 else '-' for q in chain]
        if len(set(spins)) != 1:
            num_broken += 1
            print "broken chain:", "".join(spins)
    print "%s of %s chains were broken"%(num_broken, len(flat_embedding))

#~
#~ Now, we've broken a few chains.  We modified the solution array in place,
#~ so as we execute the next section, it will use this broken soltuion.
#~
#~ Interpreting embedded solutions
#~ -------------------------------
#~
#~ Now, we'll take a look at the interpretation of solutions with broken
#~ chains.  This is still a matter of active research, but the sapi client
#~ provides a few strategies.
#~
#~ We'll use the ``minimize_energy`` strategy.  The description of the
#~ strategies provided by the sapi client, can be found in the documentation
#~ for :func:`unembed_answer`.  The ``minimize_energy`` strategy considers the
#~ chains one by one, and computes the hamiltonian energies associated with
#~ setting all spins up and all spins down, across the chain, and picks the
#~ spin value for that chain which attains the lower energy before moving on
#~ to the next chain.
#~

new_answer = unembed_answer(result['solutions'], new_emb, 'minimize_energy', h, J)
print "spin | chain spins"
for spin, chain in zip(new_answer[0], flat_embedding):
    if spin == -1:
        spin = '-'
    else:
        spin = '+'
    spins = ['+' if res0[q] == 1 else '-' for q in chain]
    print "  %s  | %s"%(spin, "".join(spins))
