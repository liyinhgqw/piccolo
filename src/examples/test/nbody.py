#!/usr/bin/python

import os
import sys; sys.path += ['src/examples/test']
import runutil, math
iterations=5
base_size=100000

def particles(n):
  return int(math.sqrt(n) * base_size)

def test_scaled_perf():
  for n in runutil.parallelism:
    runutil.run_example('NBody', 
                        n=n,
                        logfile_name='NBody.scaled_size',
                        args=['--particles=%s' % particles(n),
                              '--log_prefix=false',
                              '--sleep_time=0.00001',
                              '--iterations=%s' % iterations,
                              '--work_stealing=false'])

def test_fixed_perf(): 
  for n in runutil.parallelism:
    runutil.run_example('NBody', 
                        n=n,
                        logfile_name='NBody.fixed_size',
                        args=['--particles=%s' % base_size,
                              '--log_prefix=false',
                              '--sleep_time=0.00001',
                              '--iterations=%s' % iterations,
                              '--work_stealing=false'])


def test_checkpoint_perf():
  #os.system('pdsh -f20 -g muppets rm -rf %s' % '/scratch/power/checkpoints')
  os.system('rm -rf /scratch/power/checkpoints/')
  runutil.run_example('NBody', 
                    n=1,
                    logfile_name='NBody.checkpoint',
                    hostfile=None,
                    args=['--particles=%s' % base_size,
                          '--log_prefix=false',
                          '--sleep_time=0.00001',
                          '--iterations=%s' % iterations,
                          '--checkpoint_read_dir=/scratch/power/checkpoints/nbody.%d' % base_size,
                          '--checkpoint_write_dir=/scratch/power/checkpoints/nbody.%d' % base_size,
                          '--checkpoint=true',
                          '--restore=true',
                          '--work_stealing=false'])

  runutil.run_example('NBody', 
                    n=1,
                    logfile_name='NBody.nocheckpoint',
                    hostfile=None,
                    args=['--particles=%s' % base_size,
                          '--log_prefix=false',
                          '--sleep_time=0.00001',
                          '--iterations=%s' % iterations,
                          '--checkpoint=false',
                          '--restore=false',
                          '--work_stealing=false'])

test_checkpoint_perf()
