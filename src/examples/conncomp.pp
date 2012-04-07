#include "examples/examples.h"
#include "kernel/disk-table.h"

using std::vector;

using namespace piccolo;

DEFINE_int32(cc_num_nodes, 10000, "Default number of nodes in graph");
DEFINE_int32(cc_dump_nodes, 0, "0 to not dump final lengths, or >0 to dump that many nodes");
DEFINE_int32(cc_dump_nodes_periodic, 0, "same as dump_nodes, but after every iteration");
DEFINE_int32(cc_iters, 10, "How many iterations to perform");

static int NUM_WORKERS = 0;
static TypedGlobalTable<int, int>* maxcomp_map;
static RecordTable<PathNode>* nodes;

static void BuildGraph(int shards, int nodes, int density) {
  vector<RecordFile*> out(shards);
  File::Mkdirs("testdata/");
  for (int i = 0; i < shards; ++i) {
    out[i] = new RecordFile(
        StringPrintf("testdata/cc-graph.rec-%05d-of-%05d", i, shards), "w");
  }

  srandom(nodes);	//repeatable graphs
  fprintf(stderr, "Building graph: \n");

  for (int i = 0; i < nodes; i++) {
    PathNode n;
    n.set_id(i);

    for (int j = 0; j < density; j++) {
      n.add_target(random() % nodes);
    }

    out[i % shards]->write(n);
    if (nodes > 50 && i % (nodes / 50) == 0) {
      fprintf(stderr, ".");
    }
  }
  fprintf(stderr, "\nGraph built.\n");

  for (int i = 0; i < shards; ++i) {
    delete out[i];
  }
}

class cc_kern : public DSMKernel {
public:
  void cc_driver() {
    TypedTableIterator<long unsigned int, PathNode>* it = nodes->get_typed_iterator(current_shard());
    for(;!it->done(); it->Next()) {
        PathNode n = it->value();
        for (int j = 0; j < n.target_size(); ++j) {
          if (n.target(j) != it->key()) {
            maxcomp_map->update(n.target(j), maxcomp_map->get(n.id()));
          }
        }
    }
    delete it;
  }
};

REGISTER_KERNEL(cc_kern);
REGISTER_METHOD(cc_kern,cc_driver);

int ShortestPath(const ConfigData& conf) {
  NUM_WORKERS = conf.num_workers();

  maxcomp_map = CreateTable(0, FLAGS_shards, new Sharding::Mod,
                             new Accumulators<int>::Max);
  if (!FLAGS_build_graph) {
  nodes = CreateRecordTable < PathNode > (1, "testdata/cc-graph.rec*", false);
  }

  StartWorker(conf);
  Master m(conf);

  if (FLAGS_build_graph) {
    BuildGraph(FLAGS_shards, FLAGS_cc_num_nodes, 4);
    return 0;
  }

  if (!m.restore()) {
  PRunAll(maxcomp_map, {
          vector<int> v;
    // Initialize all vertices
    for(int i=current_shard();i<FLAGS_cc_num_nodes;i+=FLAGS_shards) {
      maxcomp_map->update(i, i);   //Initialize all distances to very large.
    }
  });
  }
  fprintf(stderr,"Loaded/initialized %d vertices\n",FLAGS_cc_num_nodes);

  //Start the timer!
  struct timeval start_time, end_time;
  gettimeofday(&start_time, NULL);

  vector<int> cptables;
  for (int i = 0; i < FLAGS_cc_iters; ++i) {
    cptables.clear();
    cptables.push_back(0);

    RunDescriptor sp_rd("cc_kern", "cc_driver", nodes, cptables);

    //Switched to a RunDescriptor so that checkpointing can be used.
    sp_rd.checkpoint_type = CP_CONTINUOUS;
    m.run_all(sp_rd);

    if (FLAGS_cc_dump_nodes_periodic > 0) {
      fprintf(stderr, "Dumping CC lengths for iteration %d\n",i);
      char dump_fname[256];
      sprintf(dump_fname,"CC_dump_periodic_i%d",i);
      FILE* fh = fopen(dump_fname,"w");
      for (int i = 0; i < FLAGS_cc_dump_nodes_periodic; ++i) {
        int d = maxcomp_map->get(i);
        fprintf(fh, "%8d: %d\n", i, (int)d);
      }
      fclose(fh);
    }

  }

//Finish the timer!
  gettimeofday(&end_time, NULL);
  long long totaltime = (long long) (end_time.tv_sec - start_time.tv_sec)
      * 1000000 + (end_time.tv_usec - start_time.tv_usec);
  fprintf(stderr, "Total CC time: %.3f seconds\n", totaltime / 1000000.0);

  if (FLAGS_cc_dump_nodes > 0) {
    fprintf(stderr, "Dumping CC lengths for final results\n");
    FILE* fh = fopen("CC_dump","w");
    /*PDontRunOne(maxcomp_map, {*/
        for (int i = 0; i < FLAGS_cc_dump_nodes; ++i) {
          int d = maxcomp_map->get(i);
          fprintf(fh, "%8d: %d\n", i, (int)d);
        }
    /*});*/
    fclose(fh);
  }
  return 0;
}
REGISTER_RUNNER(ShortestPath);
