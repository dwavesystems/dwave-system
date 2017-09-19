#!/usr/bin/env python

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

from chimera_embedding import processor

def _to_chimera(M, N, L, q):
    "Converts a qubit's linear index to chimera coordinates."
    return (q / N / L / 2, (q / L / 2) % N, (q / L) % 2, q % L)

def draw_tikz(proc, emb, spins=None, target=None, midhole=False, hues=None, color_bits=True):
    """
    Draw an embedding on a given hardware graph.
    """
    import os
    import sys
    from collections import defaultdict
    from random import shuffle
    if spins is None:
        spins = defaultdict(int)
    evil = proc._evil
    proc = proc._proc0
    M, N, L = proc.M, proc.N, proc.L
    if midhole:
        def shift(z):
            return [z - .5, z + .5][z > 0]
    else:
        def shift(z):
            return z

    def xcoord(q):
        x, y, u, k = q
        return (L + 2) * y + shift(k - L / 2. + .5) * (1-u)

    def ycoord(q):
        x, y, u, k = q
        return -( (L + 2) * x + shift(k - L / 2. + .5) * u )

    def label(q):
        x, y, u, k = q
        return N * L * 2 * x + L * 2 * y + L * u + k

    if any(not isinstance(q, tuple) for chain in emb for q in chain):
        emb = [[_to_chimera(M, N, L, q) for q in chain] for chain in emb]

    goodbits = {q for q in proc}
    usedbits = {q for chain in emb for q in chain}
    darkbits = usedbits - goodbits
    drawbits = usedbits | goodbits

    qubits = ",".join("%d/%.1f/%.1f" %
                      (label(q), xcoord(q), ycoord(q)) for q in drawbits)
    ecoup = ",".join("%d/%d" % (label(a), label(b)) for a, b in evil)
    chains = emb[:]
    if hues is None:
        hues = [j / float(len(chains)) for j in range(len(chains))]
        shuffle(chains)
    edges = [[] for b in chains] + [[], []]
    for p in proc:
        for q in proc[p]:
            if p > q:
                continue
            j = -1
            for jb, b in enumerate(chains):
                if p in b and q in b:
                    j = jb

            if p[0] == q[0] and p[1] == q[1] and p[2] != q[2]:
                edges[j].append("%d/%d" % (label(p), label(q)))
            if abs(p[0] - q[0]) == 1 and p[1] == q[1] and p[2] == q[2] == 0 and p[3] == q[3]:
                edges[j].append("%d/%d" % (label(p), label(q)))
            if abs(p[1] - q[1]) == 1 and p[0] == q[0] and p[2] == q[2] == 1 and p[3] == q[3]:
                edges[j].append("%d/%d" % (label(p), label(q)))

    if target is None:
        os.system("mkdir .chimera_drawing_temp_dir > /dev/null")
        outf = open(".chimera_drawing_temp_dir/chimera.tex", "w")
    elif target == 'stdout':
        outf = sys.stdout
    else:
        outf = open(target, 'w')
    outf.write("\\documentclass{article}\n")
    outf.write("\\usepackage{tikz}\n")
    outf.write("\\usepackage{geometry}\n")
    outf.write("\\geometry{left=1cm,right=1cm,top=1cm,bottom=1cm}\n")
    outf.write("\\begin{document}\n")
    outf.write(" \\begin{center}\n")
    outf.write("  \\thispagestyle{empty}\n")
    outf.write("  \\pagestyle{empty}\n")
    outf.write("  \\begin{tikzpicture}[scale=.2]\n")
    outf.write("    \\foreach \\c/\\x/\\y in {%s} {\n" % qubits)
    outf.write("        \\coordinate (q\\c) at (\\x,\\y);\n")
    outf.write("    }")
    ed = ",".join(edges[-1])
    outf.write("    \\foreach \\u/\\v in {%s} {\n" % ed)
    outf.write("        \\draw[thick,gray!30] (q\\u) to (q\\v);\n")
    outf.write("    }\n")
    ed = ",".join(edges[-2])
    outf.write("    \\foreach \\u/\\v in {%s} {\n" % ed)
    outf.write("        \\draw[thick,gray!70] (q\\u) to (q\\v);\n")
    outf.write("    }\n")
    for j in range(len(chains)):
        outf.write(
            "    \\definecolor{col%s}{hsb}{%f,1,.5};\n" % (j, hues[j]))
    outf.write("    \\definecolor{col%s}{hsb}{0,0,.5};\n" % len(chains))
    for j in range(len(chains)):
        ed = ",".join(edges[j])
        outf.write("    \\foreach \\u/\\v in {%s} {\n" % ed)
        outf.write("        \\draw[thick,col%s] (q\\u) to (q\\v);\n" % j)
        outf.write("    }\n")

    if color_bits:
        graybits = set(proc) - set(sum(chains, []))
        bits = enumerate(chains + [graybits])
    else:
        bits = [(len(chains), set(proc))]
    for j, chain in bits:
        chain = map(label, chain)
        outf.write("    \\foreach \\c in {%s} {\n" %
                   ','.join(map(str, chain)))
        outf.write("        \\fill[col%s] (q\\c) circle (4pt);\n" % j)
        outf.write("    }\n")

    plab = map(label, proc)

    def plus(q):
        return spins[q] == 1

    def minus(q):
        return spins[q] == -1
    up = filter(plus, plab)
    dn = filter(minus, plab)
    if up:
        outf.write(
            "    \\foreach \\c in {%s} {\n" % ','.join(map(str, up)))
        outf.write("        \\draw[white] (q\\c) circle (4pt);\n")
        outf.write("    }\n")
    if dn:
        outf.write(
            "    \\foreach \\c in {%s} {\n" % ','.join(map(str, dn)))
        outf.write("        \\draw[black] (q\\c) circle (4pt);\n")
        outf.write("    }\n")

    if ecoup:
        outf.write("    \\foreach \\u/\\v in {%s} {\n" % ecoup)
        outf.write(
            "        \\draw[line width=1mm,dashed,red] (q\\u) to (q\\v);\n")
        outf.write("    }\n")

    if darkbits:
        outf.write("    \\foreach \\c in {%s} {\n" %
                   ','.join(map(str, map(label, darkbits))))
        outf.write("        \\draw[red,dotted] (q\\c) circle (6pt);\n")
        outf.write("    }\n")

    outf.write("  \\end{tikzpicture}\n")
    outf.write(" \\end{center}\n")
    outf.write("\\end{document}\n")
    if target != 'stdout':
        outf.close()
    if target is None:
        os.system("pdflatex .chimera_drawing_temp_dir/chimera.tex > /dev/null")

_description = """Polynomial-time 'ell' clique embedder.

Computes a native clique embedding, either reading a coupler list from
stdin (default) or from file.  If reading from stdin, prints the
embedding to stdout; otherwise the embedding is written to file. """

_erata = """
Input format:
A list of couplers, one on each line.  Qubits are specified as linear
indices, separated by a single space or a single comma. Note: the size
of the chimera graph is inferred from the largest qubit seen.

Output format:
Each line is a comma-separated list of qubits, specified as linear indices
corresponding to chains in the embedding.  Lines are arranged so that
each qubit is adjacent to the next.
"""

if __name__ == '__main__':
    import argparse
    import sys

    # Define the argument format
    parser = argparse.ArgumentParser(
        description=_description,
        epilog=_erata,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-l', '--chainlength', type=int, metavar='k',
        help="all chains have length k", dest='chainlength')

    parser.add_argument('-c', '--cliquesize', nargs='+', type=int, metavar=('n', 'm'), dest='cliquesize',
        help="Takes 1 or 2 values. Makes a n-clique or fails.  m must only be specified when looking for a biclique (if unspecified, m=n).")

    parser.add_argument('--dim', nargs='+', type=int, metavar=('M', 'N'),
        help="Takes 1, 2, or 3 values. Sets the shape of the chimera subgraph to C_{M,M,4}, C_{M,N,4} or C_{M,N,L}", default=[], dest='dim')

    parser.add_argument('-i', '--infile', type=argparse.FileType('r'), dest='infile', default='-',
        help="Read the couplers from the file")

    parser.add_argument('-o', '--outfile', type=argparse.FileType('w'), dest='outfile', default=None,
        help="Writes embedding to this file. With an input file defaults to INFILE.out, otherwise stdout.")

    parser.add_argument('--draw', action='store_true', dest='draw', default=False,
        help="Draws the embedding in the file 'chimera.pdf'")

    parser.add_argument('--bipartite', action='store_true', dest='bipartite', default=False,
        help="finds the largest complete balanced bipartite embedding")

    # This actually parses the sys.argv list
    args = parser.parse_args()
    # Some value testing is done by the parser, but we are going to do a little more.

    # Check the chainlength
    if args.chainlength and args.chainlength <= 0:
        parser.error("argument --chainlength: must be greater than zero.")
    chainlength = args.chainlength

    # Check the cliquesize
    if args.cliquesize:
        if any(map(lambda c: c <= 0, args.cliquesize)):
            parser.error("argument --cliquesize: values must be greater than zero")
        if len(args.cliquesize) not in [1, 2]:
            parser.error("argument --cliquesize: expects one or two arguments""")
        if len(args.cliquesize) == 2:
            args.bipartite = True
    cliquesize = args.cliquesize

    # Check the dimensions of the input graph
    dim = None
    if args.dim:
        if any(map(lambda c: c <= 0, args.dim)):
            parser.error("argument --dim: values must be greater than zero")
        if len(args.dim) not in [1, 2, 3]:
            parser.error("argument --dim: expects one to three arguments""")

        dim = args.dim
        if len(dim) == 1:
            dim.append(dim[0])
        if len(dim) == 2:
            dim.append(4)

    # Check the output file
    if args.outfile is None:
        if not args.infile or args.infile == sys.stdin:
            args.outfile = sys.stdout
        else:
            args.outfile = open(args.infile.name + '.out', 'w')

    # Read in our adjacency file as a series of lines
    txt = args.infile.readlines()

    # Parse the input file
    edges = []
    for coupler in txt:
        coupler = coupler.strip()
        if len(coupler) == 0:
            continue
        elif coupler.count(' ') == 1:
            a, b = coupler.split()
        elif coupler.count(',') == 1:
            a, b = coupler.split(",")
        else:
            err = "infile should contain a list of couplers in the format 'x,y' or 'x y' where x and y are integers"
            raise RuntimeError(err)
        a, b = int(a), int(b)
        edges.append((a, b))

    # If the dimension of the input graph was not set guess it based on the adjacency
    if dim is None:
        q = max(map(max, edges))
        if q < 512:
            M = N = 8
        elif q < 2048:
            M = N = 16
        else:
            M = N = 32
        L = 4
    else:
        M, N, L = dim

    # Initialize our model of the functional chimera graph of the D-Wave chip provided
    proc = processor(edges, M=M, N=N, L=L)
    emb = []

    # Make the appropriate call to the
    if args.bipartite:
        if cliquesize is not None:
            A, B = proc.tightestNativeBiClique(
                cliquesize[0], m=cliquesize[-1], chain_imbalance=None, max_chain_length=chainlength)
            emb = A + B
        elif chainlength is not None:
            emb = proc.largestNativeBiClique(
                max_chain_length=chainlength, chain_imbalance=None)
        else:
            emb = proc.largestNativeBiClique()
    else:
        if chainlength is not None:
            emb = proc.nativeCliqueEmbed(chainlength - 1)
        elif cliquesize is not None:
            emb = proc.tightestNativeClique(cliquesize[0])
        else:
            emb = proc.largestNativeClique()

    if cliquesize is not None:
        if args.bipartite:
            if len(emb) < cliquesize[0] + cliquesize[-1]:
                raise RuntimeError("Failed to obtain a K_{%s,%s}" % (cliquesize[0], cliquesize[-1]))
        else:
            if len(emb) < cliquesize[0]:
                raise RuntimeError(
                    "Failed to obtain a %s-clique with chainlength %s" % (cliquesize[0], chainlength))
            else:
                emb = emb[:cliquesize[0]]

    if args.draw:
        draw_tikz(proc, emb)

    # Encode and print our output
    out = "\n".join(",".join(map(str, c)) for c in emb) + '\n'
    args.outfile.write(out)
