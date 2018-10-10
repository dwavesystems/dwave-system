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

# This is a little utility meant to mix runnable python scripts with reST-
# formatted text and build the result into our documentation.  This enables
# us to use a nice-ish literate programming style for our examples.

import sys
import subprocess
from uuid import uuid4

if len(sys.argv) < 2:
    print "usage: python literate.py filename"
    sys.exit()

filename = sys.argv[1]
script = open(filename)

def execute(lines):
    print "running", 


    pysession.communicate(input="\n".join(lines))

rest_chunks = ['']
py_in_chunks = []

runlines = ['']
for line in script:
    line = line.rstrip()
    if line[:2] == "#~":
        if runlines:
            py_in_chunks.append(runlines)
            runlines = []
        if line[2:3].isspace():
            rest_chunks[-1]+= line[3:]+"\n"
        else:
            rest_chunks[-1]+= line[2:]+"\n"
    elif line[:1] == "#":
        continue
    else:
        runlines.append(line.rstrip())
        if rest_chunks[-1]:
            rest_chunks.append('')

py_in_chunks.append(runlines) #empty or not
if len(rest_chunks) < len(py_in_chunks):
    rest_chunks.append('')

py_in_hunks = map("\n".join, py_in_chunks)
hunk_sep = str(uuid4())
hunker = "\nprint %r\n"%hunk_sep
py_in = hunker.join(py_in_hunks)

pysession = subprocess.Popen('python', stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)

(out, err) = pysession.communicate(py_in)
if err:
    raise RuntimeError, "received something on stderr, I can't handle this! %s"%err

out_chunks = out.split(hunk_sep)

for in_chunk, out_chunk, rest_chunk in zip(py_in_chunks, out_chunks, rest_chunks):
    if any(in_chunk) and not "".join(in_chunk).isspace():
        print ".. code-block:: python"
        print
        for line in in_chunk:
            if line and line[0].isspace():
                print '    ... '+line
            elif line: 
                print '    >>> '+line
            else:
                pass##print
        for line in out_chunk.lstrip('\n').split("\n"):
            print '    '+line
        print
    print rest_chunk

