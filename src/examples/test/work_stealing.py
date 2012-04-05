#!/usr/bin/python

import os, sys, math
import numpy as N

shards = [int(v) for v in open('/home/power/shard_sizes').read().split() if v]
average = 0

class Bucket:
    def __init__(self, id):
        self.id = id
        self.shards = []
        self.bad = 0

    def cutoff(self, limit):
        self.shards.sort()
        self.shards.reverse()
        
        b = 0
        for i in range(len(self.shards) - 1):
            if b >= limit:
                return i
            b += self.shards[i]
        return -1
    
    def biggest(self, fs):
        i = self.cutoff(fs)
#        print fs, sum(self.shards), i
        if i == -1: return -1
        return self.shards[i + 1]
        
    def __lt__(self, other):
        return sum(self.shards) < sum(other.shards)
    
    def steal(self, other, reverse=0):
        cutoff = other.biggest(sum(self.shards))
        i = other.shards.index(cutoff)

        print cutoff, i, other.shards
        
        if reverse:
            self.shards += [other.shards[-1]]
            del other.shards[-1]
        else:
            self.shards += [other.shards[i]]
            del other.shards[i]
        self.bad = 1
        return True

    def __repr__(self):
        return '%d' % sum(self.shards)

    def skew(self):
        return (sum(self.shards) - average) / average


def steal_one(buckets, rev, tries=1):
    fastest = min(buckets, key=lambda b: sum(b.shards))
    fs = sum(fastest.shards)
    buckets.sort(key=lambda b: b.biggest(fs))
    buckets.reverse()

    for i in range(tries):
        slowest = buckets[i]
        if slowest.biggest(fs) == -1: break
        if fastest.steal(slowest, rev): return True
    return False


def TestStealing(bucket_count, tries):
  buckets = [Bucket(i) for i in range(bucket_count)]
  for i in range(len(shards)): buckets[i % bucket_count].shards += [shards[i]]

  for i in range(tries):
      if not steal_one(buckets, False, tries=1):
          print 'Stole: ', i
          break

  skew = sorted([b.skew() for b in buckets])
  open('workstealing.%d.fit.tries=%d' % (bucket_count, tries), 'w').write('\n'.join([str(v) for v in skew]))

def TestAssignments(bucket_count):
  global average, shards
  average = sum(shards) / float(bucket_count)
  TestStealing(bucket_count, 0)
  TestStealing(bucket_count, 10)

  shards = sorted(shards)
  shards.reverse()

  fit = [0 for i in range(bucket_count)]
  for s in shards:
      fit.sort()
      fit[0] += s

  fit = sorted([(v - average)/average for v in fit])
  open('preselect.%d.fit' % bucket_count, 'w').write('\n'.join([str(v) for v in fit]))

#TestAssignments(4)
#TestAssignments(12)
TestAssignments(64)

p = os.popen('gnuplot -persist', 'w')
p.write('''plot './workstealing.64.fit.tries=0' with lines,\
 './workstealing.64.fit.tries=10' with lines,\
 './preselect.64.fit' with lines,\
 './workstealing.12.fit.tries=10' with lines,\
 './preselect.12.fit' with lines
 ''')
p.flush()
