#!/usr/bin/python

import time
import os, sys, re, subprocess

parallelism = [8, 16, 32, 64]
#parallelism = [64, 32, 16, 8]

def hostfile_info(f):
  cores = machines = 0
  if not f: return cores, machines

  mdict = {}
  for l in open(f).readlines():
    cores += int(l.split('=')[1])
    mdict[l.split()[0]] = 1
    machines = len(mdict)
  return cores, machines


def system(*args):
  os.system(*args)

logfile = None
def log(fmt, *args):
  print fmt % args
  print >>logfile, fmt % args
  logfile.flush()

def init_log(results_dir, n, logfile_name):
  if not logfile_name:
    raise KeyError, 'Need a logfile target!'
  
  output_dir="%s/parallelism.%s/" % (results_dir, n)
  system('mkdir -p %s' % output_dir)
  global logfile
  logfile = open('%s/%s' % (output_dir, logfile_name), 'w')
    
  log("Writing output to: %s", output_dir)

def run_example(runner, 
                n=64, 
                build_type='release',
                results_dir='results',
                hostfile='conf/mpi-cluster',
                logfile_name=None,
                args=None):
  if not logfile_name: logfile_name = runner
  if not args: args = []
  
  init_log(results_dir, n, logfile_name)

  cores, machines = hostfile_info(hostfile)
  affinity = 0 if n >= cores else 1


  log("Running with %s machines, %s cores" % (machines, cores))
  log("Runner: %s", runner)
  log("Parallelism: %s", n)
  log("Processor affinity: %s", affinity)

  cmd = ' '.join(['mpirun',
                  '-hostfile %s' % hostfile if hostfile else '',
                  '-bycore',
                  '-nooversubscribe',
                  '-mca mpi_paffinity_alone %s' % affinity,
                  '-n %s ' % (n + 1),
                  'bin/%s/examples/example-dsm' % build_type,
                  '--runner=%s' % runner,
                  '--log_prefix=false',
                  ]
                  + args)
  run_command(cmd, n, 
              results_dir=results_dir,
              logfile_name=logfile_name)

def run_command(cmd, n,
                results_dir='results',
                logfile_name=None):

  init_log(results_dir, n, logfile_name)

  system('rm -f profile/*')
  log("Flushing buffer cache...")
  #system("pdsh -f 100 -g muppets -l root 'echo 3 > /proc/sys/vm/drop_caches'")
  log("Killing existing workers...")
  #system("pdsh -f 100 -g muppets 'pkill -9 example-dsm || true'")

  log('Command: %s', cmd)
  start = time.time()
  handle = subprocess.Popen(cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            shell=True)

  while handle.returncode == None:
    handle.poll()
    l = handle.stdout.readline().strip()
    if l: log('%s', l)

  end = time.time()
  log('Finished in: %s', end - start)

  if handle.returncode != 0:
    log('Error while running %s!', cmd)
    return 1
