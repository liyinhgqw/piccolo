#!/usr/bin/python

import traceback
import os
import atexit
import sys; sys.path += ['src/examples/test']
import runutil, math

checkpoint_write_dir="/scratch/power/checkpoints/"
checkpoint_read_dir="/scratch/power/checkpoints/"
scaled_base_size=5
fixed_base_size=100
shards=1
memory_graph=0
iterations=10

def cleanup(size):
  print "Removing old checkpoints..."
  return
  os.system('rm -rf /scratch/power/')
  os.system('pdsh -f20 -g muppets mkdir -p %s/%sM' % (checkpoint_write_dir, size))
  os.system('pdsh -f20 -g muppets rm -rf %s/%sM' % (checkpoint_write_dir, size))

def system(cmd):
  print cmd
  os.system(cmd)

def make_graph(size, hostfile='conf/mpi-localhost'):
  if memory_graph: return
  system(' '.join(
                  ['mpirun',
                   '-hostfile %s' % hostfile,
                   '-bynode',
                   '-n %s' % runutil.hostfile_info(hostfile)[0],
                   'bin/release/examples/example-dsm',
                   '--runner=Pagerank',
                   '--build_graph',
                   '--nodes=%s' % (size * 1000 * 1000),
                   '--shards=%s' % shards,
                   '--iterations=0',
                   '--work_stealing=false',
                   '--graph_prefix=/scratch/pagerank_test/%sM/pr' % size]))

def run_pr(fname, size, n, args=None, **kw):
  try:
    runutil.run_example('Pagerank', 
                        n=n,
                        logfile_name=fname,
                        build_type='release',
                        args=['--iterations=%s' % iterations,
                              '--sleep_time=0.001',
                              '--nodes=%s' % (size * 1000 * 1000),
                              '--memory_graph=%d' % memory_graph,
                              '--shards=%s' % (n * 16),
                              '--work_stealing=true',
                              '--checkpoint_write_dir=%s/%sM' % (checkpoint_write_dir, size),
                              '--checkpoint_read_dir=%s/%sM' % (checkpoint_read_dir, size),
                              '--graph_prefix=/scratch/pagerank_test/%sM/pr' % (size),
                              ] + args,
                        **kw)
  except SystemError:
    traceback.print_exc()
    return

# test scaling with work size, and with a fixed size of data
def test_scaled_perf():
  for n in runutil.parallelism[5:]:
    graphsize = scaled_base_size * n
    make_graph(graphsize)
    run_pr('Pagerank.scaled_size', graphsize, n, ['--checkpoint=false'])


def test_fixed_perf():
  for n in runutil.parallelism:  
    graphsize = fixed_base_size
    make_graph(graphsize)
    run_pr('Pagerank.fixed_size', graphsize, n, ['--checkpoint=false'])


def test_work_stealing():
  make_graph(graphsize, hostfile='slow_hostfile')
  run_pr('Pagerank.with_stealing', 
         graphsize,
         n,
         hostfile='slow_hostfile',
         args=['--checkpoint=false', '--work_stealing=true'])

def test_checkpointing():    
  cleanup(10)
  run_pr('Pagerank.checkpoint.10M', 10, 1, ['--checkpoint=true', '--restore=false'])
  run_pr('Pagerank.nocheckpoint.10M', 10, 1, ['--checkpoint=false', '--restore=false'])

def test_slow_worker():
  n = 64
  graphsize = scaled_base_size * n * 2

  os.system('ssh beaker-10 "pkill -f cpuloop.sh"')
  run_pr('Pagerank_noslow_with_stealing', graphsize, n, 
         ['--checkpoint=false', '--restore=false', '--work_stealing=true', ])

  run_pr('Pagerank_noslow', graphsize, n, ['--checkpoint=false', '--restore=false', '--work_stealing=false', ])

  atexit.register(os.system, 'ssh beaker-10 "pkill -f cpuloop.sh"')
  os.system('ssh beaker-10 "taskset -c 0x1 /home/power/cpuloop.sh </dev/null &>/dev/null&"')
  
  run_pr('Pagerank_slow_no_stealing', graphsize, n, ['--checkpoint=false', '--restore=false', '--work_stealing=false', ])  
  run_pr('Pagerank_slow_with_stealing', graphsize, n, ['--checkpoint=false', '--restore=false', '--work_stealing=true', ])

#test_checkpointing()
#test_fixed_perf()
#test_scaled_perf()
#test_slow_worker()
make_graph(10)
