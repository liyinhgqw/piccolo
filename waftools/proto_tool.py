#!/usr/bin/env python

from waflib.Task import Task
from waflib.TaskGen import extension

class proto(Task):
    run_str = 'protoc ${SRC[0].abspath()} -I ${SRCDIR} --cpp_out ${bld.out_dir}'
    color = 'BLUE'
    ext_out = ['.pb.cc', '.pb.h']

@extension('.proto')
def process_idl(self, node):
    cc_node = node.change_ext('.pb.cc')
    h_node = node.change_ext('.pb.h')
    self.create_task('proto', node, [cc_node, h_node])
    self.source.append(cc_node)

# verify we have a recent protocol buffer library
PROTOBUF_FRAGMENT = '''
#include <string>
namespace google {
  namespace protobuf {
    namespace internal {
      extern std::string kEmptyString;
    }
  }
}

int main() {
  return 0;
}
'''

def configure(conf):
  conf.check(features='cxx cxxprogram',
           fragment=PROTOBUF_FRAGMENT,
           lib='protobuf pthread')

