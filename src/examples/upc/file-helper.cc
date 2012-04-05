#include "util/file.h"
#include "examples/examples.pb.h"
#include "file-helper.h"

using namespace piccolo;

// Wrapper struct to return to C-land.
struct RFile {
  RFile(RecordFile* r) : f(r) {}
  ~RFile() { delete f; }
  RecordFile *f;
};

RFile *RecordFile_Open(const char* f, const char *mode) {
  return new RFile(new RecordFile(f, mode));
}

GraphEntry* RecordFile_ReadGraphEntry(RFile *r) {
  PathNode n;
  if (r->f->read(&n)) {
    GraphEntry *e = new GraphEntry;
    e->num_neighbors = n.target_size();
    e->neighbors = (int*)malloc(sizeof(int) * e->num_neighbors);
    for (int i = 0; i < e->num_neighbors; ++i) {
      e->neighbors[i] = n.target(i);
    }
    e->id = n.id();

    return e;
  }

  return NULL;
}

void RecordFile_Close(RFile* r) {
  delete r;
}
