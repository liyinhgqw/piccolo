#!/usr/bin/python

import sys; sys.path += ['src/examples/test']
import runutil, math
fixed_base = 2500
scaled_base = 1000
iterations = 3

def edge(n):
    v = round(scaled_base * pow(n, 1./3.) / 250)
    return int(v) * 250

def test_scaled_perf():
  for n in runutil.parallelism[4:]:                            
     runutil.run_example('MatrixMultiplication', 
                          n=n,
                          logfile_name='MatrixMultiplication.scaled_size',
                          args=['--iterations=%s' % iterations,
                                '--sleep_time=0.0001',
                                '--work_stealing=false',
                                '--edge_size=%d' % edge(n),
                                ])


def test_fixed_perf():
  for n in runutil.parallelism:   
     runutil.run_example('MatrixMultiplication', 
                          n=n,
                          logfile_name='MatrixMultiplication.fixed_size',
                          args=['--iterations=%s' % iterations,
                                '--sleep_time=0.0001',
                                '--edge_size=%d' % fixed_base,
                                '--work_stealing=false',
                                ])

#test_fixed_perf()
test_scaled_perf()
