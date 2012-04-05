#!/usr/bin/python

__doc__ = '''
ppp: The Piccolo Pre-Processor.  

Yes, alliteration is fun.

Transforms .pp files to C++, replacing instances of the PMap and PReduce
calls with calls to generated kernels and barriers.
'''

import os, sys, re, _sre

class PrettyPattern(object):
  def __init__(self, txt, p):
    self._txt = txt
    self._p = p

  def __repr__(self): return self._txt

  def match(self, *args): return self._p.match(*args)
  def search(self, *args): return self._p.search(*args)

def compile(p, name=None, flags=0):
  if isinstance(p, PrettyPattern): return p
  if not name: name = p
  return PrettyPattern(name, re.compile(p, flags))

class Scanner(object):
  ws = compile(r'[ \t\n\r]+', name='whitespace')
  comment = compile(r'(//[^\n]*)|(/\*.*?\*/)', name='comment', flags=re.DOTALL)
  id = compile(r'([a-zA-Z][a-z0-9_]*)', name='identifier')

  def __init__(self, f):
    self._f = f
    self._d = open(f).read()
    self._out = ''
    self.line = 1
    self.offset = 0
    self._stack = []

  def update(self, g, update_pos):
    if update_pos:
      self._out += self._d[:g.start()]
      s = g.end()

      # this is horrible...
      self.line += self._d.count('\n', 0, s)
      self.offset = s - self._d.rfind('\n', 0, s)

      self._d = self._d[g.end():]


  def slurp(self, *patterns):
    matched = True
    while matched:
      matched = False
      for p in patterns:
        p = compile(p)
        g = p.match(self._d)
        if g:
          matched = True
          self.update(g, True)

  def match(self, pattern, update_pos=True, must_match=True):
    g = compile(pattern).match(self._d)
    if not g and must_match:
      raise ValueError, 'Expected `%s\' in input at line %d (offset %d), "...%s...", while parsing:\n >>>%s' % \
                        (pattern, self.line, self.offset, self._d[:20], '\n >>>'.join([s[0] for s in self._stack]))
    self.update(g, update_pos)
    return g

  def search(self, pattern, update_pos=True):
    g = compile(pattern).search(self._d)
    if g: self.update(g, update_pos)
    return g

  def read(self, *patterns):
    g = []
    for p in patterns:
      self.slurp(Scanner.ws, Scanner.comment)
      g.append(self.match(p).group(0))
    return g

  def peek(self, pattern):
    self.slurp(Scanner.ws, Scanner.comment)
    return self.match(pattern, update_pos=False, must_match=False) != None

  def push(self, fname):
    self._stack += [(fname, self.line, self.offset)]

  def pop(self): self._stack.pop()


HEADER = '''
#include "client/client.h"

using namespace piccolo;
'''

MAP_KERNEL = '''
class %(prefix)sMapKernel%(id)s : public DSMKernel {
public:
  virtual ~%(prefix)sMapKernel%(id)s() {}
  template <class K, %(klasses)s>
  void run_iter(const K& k, %(decl)s) {
#line %(line)s "%(filename)s"
    %(code)s;
  }
  
  template <class TableA>
  void run_loop(TableA* a) {
    typename TableA::Iterator *it =  a->get_typed_iterator(current_shard());
    for (; !it->done(); it->Next()) {
      run_iter(it->key(), %(calls)s);
    }
    delete it;
  }
  
  void map() {
      run_loop(%(main_table)s);
  }
};

REGISTER_KERNEL(%(prefix)sMapKernel%(id)s);
REGISTER_METHOD(%(prefix)sMapKernel%(id)s, map);
'''

RUN_KERNEL = '''
class %(prefix)sRunKernel%(id)s : public DSMKernel {
public:
  virtual ~%(prefix)sRunKernel%(id)s () {}
  void run() {
#line %(line)s "%(filename)s"
      %(code)s;
  }
};

REGISTER_KERNEL(%(prefix)sRunKernel%(id)s);
REGISTER_METHOD(%(prefix)sRunKernel%(id)s, run);
'''

class Counter(object):
  def __init__(self):
    self.c = 0

  def __call__(self):
    self.c += 1
    return self.c

get_id = Counter()

def ParseKeys(s):
  s.push('ParseKeys')
  k, _, t = s.read(Scanner.id, ':', Scanner.id)
  result = [(k, t)]

  if s.peek(','):
    s.read(',')
    result += ParseKeys(s)

  s.pop()
  return result

# match brackets
def ParseCode(s):
  s.push('ParseCode')
  s.read('{')
  code = ''
  while 1:
    code += s.search(r'[^{}]*', re.DOTALL).group(0)
    if s.peek('{'): code += '{' + ParseCode(s) + '}'
    if s.peek('}'):
      s.read('}')
      s.pop()
      return code

def ParsePMap(s):
  s.push('ParsePMap')
  _, keys, _, code, _ = s.read(r'\(', '{'), ParseKeys(s), s.read('}', ','), ParseCode(s), s.read(r'\)', ';')


  filename = s._f
  prefix = re.sub(r'[\W]', 'P_', filename)

  id = get_id()
  main_table = keys[0][1]

  s._out += 'm.run_all("%sMapKernel%d", "map", %s);' % (prefix, id, main_table)

  i = 0
  klasses, decls, calls = [], [], []
  for k, v in keys:
    klasses += ['class Value%d' % i]
    if i == 0:
      decls += ['Value%d &%s' % (i, k)]
      calls += ['it->value()']
    else:
      decls += ['const Value%d &%s' % (i, k)]
      calls += ['%s->get(it->key())' % v]
    i += 1

  s._d += MAP_KERNEL % dict(filename=filename,
                            prefix=prefix,
                            line=s._stack[-1][1],
                            id=id,
                            decl=','.join(decls),
                            calls=','.join(calls),
                            klasses=','.join(klasses),
                            code=code,
                            main_table=main_table) + '\n'
  s.pop()


def ParsePRunOne(s):
  s.push('ParsePRunOne')
  _, table, _, code, _ = s.read(r'\(',), s.read(Scanner.id)[0], s.read(','), ParseCode(s), s.read(r'\)', ';')

  filename = s._f
  prefix = re.sub(r'[\W]', '_', filename)

  id = get_id()
  s._out += 'm.run_one("%sRunKernel%d", "run", %s);' % (prefix, id, table)

  s._d += RUN_KERNEL % dict(filename=filename,
                            prefix=prefix,
                            line=s._stack[-1][1],
                            id=id,
                            code=code) + '\n'
  s.pop()
  return code

def ParsePRunAll(s):
  s.push('ParsePRunAll')
  _, table, _, code, _ = s.read(r'\(',), s.read(Scanner.id)[0], s.read(','), ParseCode(s), s.read(r'\)', ';')

  filename = s._f
  prefix = re.sub(r'[\W]', '_', filename)

  id = get_id()
  s._out += 'm.run_all("%sRunKernel%d", "run", %s);' % (prefix, id, table)

  s._d += RUN_KERNEL % dict(filename=filename,
                            prefix=prefix,
                            line=s._stack[-1][1],
                            id=id,
                            code=code) + '\n'
  s.pop()
  return code


def ProcessFile(f_in, f_out):
  global c
  s = Scanner(f_in)
  print >> f_out, HEADER
  print >> f_out, '#line 1 "%s"' % f_in

  while 1:
    g = s.search('PMap|PRunOne|PRunAll')
    if not g: break
    if g.group(0) == 'PMap': ParsePMap(s)
    elif g.group(0) == 'PRunOne': ParsePRunOne(s)
    elif g.group(0) == 'PRunAll': ParsePRunAll(s)

  print >> f_out, s._out
  print >> f_out, s._d

if __name__ == '__main__':
  n_in = sys.argv[1]
  n_out = n_in + '.cc' if len(sys.argv) < 3 else sys.argv[2]
  f_out = open(n_out, 'w')
  try:
    ProcessFile(n_in, f_out)
  except ValueError, e:
    print >> sys.stderr, 'Parse error:', e
    sys.exit(1)

  f_out.close()

