#include <upc.h>
#include <stdio.h>
#include <stdlib.h>
#include "test/file-helper.h"

#define BS 1000
#define NUM_NODES 1000000

static int ROW_START(int id) {
  int block = id / BS;
  return block * BS * THREADS;
}

static int IDX(int id) {
  int block = id / BS;
  return (block * BS * THREADS) + (MYTHREAD * BS) + (id % BS);
}

static int PRIMARY_IDX(int id) {
  int block = id / BS;
  int primary = block % THREADS;
  return (block * BS * THREADS) + (primary * BS) + (id % BS);
}

#define MIN(a, b) ( (a < b) ? a : b )

// Entries are sharded according to the following scheme, assuming
// 4 nodes:
//
// 1: [OWNER][COPY][COPY][COPY]
// 2: [COPY][OWNER][COPY][COPY]
// 3: [COPY][COPY][OWNER][COPY]
// ...
// Each thread has a copy of a block; the canonical copy for block K
// is located at (BS * K * THREADS) {start of the row} +
// (K % THREADS) * BS {offset of the block}.

shared [BS] int distance[THREADS * NUM_NODES];

int main(int argc, char** argv) {
  char srcfile[1000];
  int current_entry, num_entries;
  int i, j, k, idx;
  struct RFile *r;
  GraphEntry *e, *entries;

  int *local_copy;

  // Load graph from disk...
  current_entry = 0;

  local_copy = malloc(sizeof(int) * BS * THREADS);

  entries = (GraphEntry*)malloc(NUM_NODES * sizeof(GraphEntry));
  sprintf(srcfile, "testdata/sp-graph.rec-%05d-of-%05d", MYTHREAD, THREADS);

  r = RecordFile_Open(srcfile, "r");;
  while ((e = RecordFile_ReadGraphEntry(r))) {
    entries[current_entry++] = *e;
  }
  RecordFile_Close(r);

  num_entries = current_entry;
  fprintf(stderr, "Done reading: %d entries\n", num_entries);

  for (i = BS * MYTHREAD; i < NUM_NODES * THREADS; i += BS * THREADS) {
    for (j = 0; j < BS; ++j) {
      distance[i + j] = 1000000;
    }
  }

  distance[0] = 0;

  upc_barrier;

  for (i = 0; i < 20; ++i) {
    fprintf(stderr, "Thread %d, iteration: %d\n", MYTHREAD, i);

    // Propagate distances from our local node - we write to a local block that will be later fetched
    // by the owner of the target.
    for (j = 0; j < num_entries; ++j) {
      e = &entries[j];
      for (k = 0; k < e->num_neighbors; ++k) {
        distance[IDX(e->neighbors[k])] = MIN(distance[IDX(e->neighbors[k])], distance[IDX(e->id)] + 1);
      }
    }

    upc_barrier;

    // For each block of entries that we own, fetch remote blocks and compare against our local block
    for (j = BS * MYTHREAD; j < NUM_NODES; j += BS * THREADS) {
      upc_memget(local_copy, &distance[ROW_START(j)], BS * THREADS * sizeof(int));

      for (k = 0; k < BS * THREADS; ++k) {
        idx = j + (k % BS);
        distance[IDX(idx)] = MIN(local_copy[k], distance[IDX(idx)]);
      }
    }

    upc_barrier;
  }

//  if (MYTHREAD == 0) {
//    for (i = 0; i < NUM_NODES; ++i) {
//      if (i % 40 == 0) { printf("\n%d: ", i); }
//      printf("%d ", distance[PRIMARY_IDX(i)]);
//    }
//  }
//  printf("\n");
//  fflush(stdout);
}
