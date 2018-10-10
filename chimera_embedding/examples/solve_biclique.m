%
%Copyright 2016 D-Wave Systems Inc.
%
%Licensed under the Apache License, Version 2.0 (the "License");
%you may not use this file except in compliance with the License.
%You may obtain a copy of the License at
%
%    http://www.apache.org/licenses/LICENSE-2.0
%
%Unless required by applicable law or agreed to in writing, software
%distributed under the License is distributed on an "AS IS" BASIS,
%WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
%See the License for the specific language governing permissions and
%limitations under the License.
%

%create a local sapi connection
conn = sapiLocalConnection();

%or, make a remote connection to a D-Wave system.  You'll need to fill in your
%authentication information here and and change solverName below
% url = '';
% token = '';
% conn = sapiRemoteConnection(url, token);

%now, open up the solver
solverName = 'c4-sw_sample';
solver = sapiSolver(conn, solverName);


%fetch the graph structure of the solver
A = getHardwareAdjacency(solver);

%read in the embedding file.  Since it's a space-delimited text file,
%that's really easy
embeddings = num2cell(dlmread('embedded_clique_largest'),2)';

%how many variables are we playing with?
[~,n] = size(embeddings);

%construct a random Ising spin glass problem on n variables
J = triu(ones(n),1);
J(J==1) = 1-2*rand((n*n-n)/2,1);
h = 1-2*rand(1,n);

%now, embed the problem.  Problem interactions are stored in j0 and chain
%edges are stored in jc.  The fields from h are distributed over the chains
%in h0.
[h0, j0, jc, ~] = sapiEmbedProblem(h, J, embeddings, A);

%use chain strength=.25 for the sake of this example only, we typically use
%sqrt(n) but in this case we want some broken chains so that the
%unembedding does something interesting.
embedded_J = j0+.25*jc;

%Now, grab a sample from the solver.  This isn't the most efficient way to
%use the hardware, but this is just an example.
answer = sapiSolveIsing(solver, h0, embedded_J, 'num_reads', 1);

%Now, unembed the answer.  There are several strategies that can be used.
%See the documentation of sapiUnembedAnswer for more information.
newAnswer = sapiUnembedAnswer(answer.solutions, embeddings, 'minimize_energy', h, J)
