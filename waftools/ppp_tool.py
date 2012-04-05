#!/usr/bin/env python

from waflib import Logs

from waflib import TaskGen
TaskGen.declare_chain(
        name='ppp',
        rule='${PPP} ${SRC} ${TGT}',
        shell=False,
        ext_in='.pp',
        ext_out='.cc',
        reentrant=True,
)

def configure(ctx):
  ctx.msg('Setting up Piccolo Pre-Processor', 1)
  ctx.env['PPP'] = ctx.path.find_resource('util/ppp.py').abspath()
