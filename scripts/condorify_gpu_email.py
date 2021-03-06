#!/usr/bin/env python

"""
A very simple wrapper for a function call, making a corresponding condor script
and submitting it.
"""

import os, sys

CondorScript = """
universe = vanilla
+GPUJob = true
#requirements = GPU
#requirements = TARGET.GPUSlot
requirements = TARGET.GPUSlot && NumJobStarts == 0

notify_user = h.rahul@utexas.edu
Notification = Complete

request_GPUs = 1

#request_memory = 12 * 1024
#request_memory = 4 * 1024

getenv = true

Initialdir = %s
Executable = %s

%s

+Group   = "GRAD"
+Project = "IR"
+ProjectDescription = "CS395T"

Arguments = %s
Queue
"""

#OutputLine = """
#Error = err.%s
#Output = %s
#Log = results.log
#"""
OutputLine = """
Error = err.%s
Output = %s
Log = results.log
"""

RawExecutable = sys.argv[1]
Arguments = ' '.join(sys.argv[2:-1])
OutputFile = sys.argv[-1]

Executable = os.popen('/bin/which %s' % RawExecutable).read()
CurrentDir = os.popen('/bin/pwd').read()

# remove path information
SafeOutputFile = '-'.join(OutputFile.split('/'))

if OutputFile == "/dev/null":
    outputlines = ""
else:
    outputlines = OutputLine % (SafeOutputFile, OutputFile)

    condor_file = '/tmp/%s.condor' % (SafeOutputFile)
    f = open(condor_file, 'w')
    f.write(CondorScript % (CurrentDir, Executable, outputlines, Arguments))
    f.close()

    os.popen('/lusr/opt/condor/bin/condor_submit %s' % condor_file)
