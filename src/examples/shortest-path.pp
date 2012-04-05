#include "examples/examples.h"
#include "kernel/disk-table.h"

using std::vector;

using namespace piccolo;

DEFINE_int32(num_nodes, 10000, "Default number of nodes in graph");
DEFINE_int32(dump_nodes, 0, "0 to not dump final lengths, or >0 to dump that many nodes");
DEFINE_int32(dump_nodes_periodic, 0, "same as dump_nodes, but after every iteration");
DEFINE_int32(sssp_iters, 10, "How many iterations to perform");

static int NUM_WORKERS = 0;
static TypedGlobalTable<int, double>* distance_map;
static RecordTable<PathNode>* nodes;

static void BuildGraph(int shards, int nodes, int density) {
  vector<RecordFile*> out(shards);
  File::Mkdirs("testdata/");
  for (int i = 0; i < shards; ++i) {
    out[i] = new RecordFile(
        StringPrintf("testdata/sp-graph.rec-%05d-of-%05d", i, shards), "w");
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

class sssp_kern : public DSMKernel {
public:
  void sssp_driver() {
//    for(int i=0;i<100;i++) {
    TypedTableIterator<uint64_t, PathNode>* it = nodes->get_typed_iterator(current_shard());
    for(;!it->done(); it->Next()) {
        PathNode n = it->value();
        for (int j = 0; j < n.target_size(); ++j) {
          if (n.target(j) != it->key())
            distance_map->update(n.target(j), distance_map->get(n.id()) + 1);
        }
    }
    delete it;
//    }
  }
};

REGISTER_KERNEL(sssp_kern);
REGISTER_METHOD(sssp_kern,sssp_driver);

int ShortestPath(const ConfigData& conf) {
  NUM_WORKERS = conf.num_workers();

  distance_map = CreateTable(0, FLAGS_shards, new Sharding::Mod,
                             new Accumulators<double>::Min);
  if (!FLAGS_build_graph) {
  nodes = CreateRecordTable < PathNode > (1, "testdata/sp-graph.rec*", false);
  }

  StartWorker(conf);
  Master m(conf);

  if (FLAGS_build_graph) {
    BuildGraph(FLAGS_shards, FLAGS_num_nodes, 4);
    return 0;
  }

  distance_map->resize(2*FLAGS_num_nodes);

  if (!m.restore()) {
  PRunAll(distance_map, {
          vector<double> v;
    for(int i=current_shard();i<FLAGS_num_nodes;i+=FLAGS_shards) {
      distance_map->update(i, 1e9);   //Initialize all distances to very large.
    }

    // Initialize a root node.
    distance_map->update(0, 0);
  });
  }

  //Start the timer!
  struct timeval start_time, end_time;
  gettimeofday(&start_time, NULL);

  vector<int> cptables;
  for (int i = 0; i < FLAGS_sssp_iters; ++i) {
    cptables.clear();
    cptables.push_back(0);

    RunDescriptor sp_rd("sssp_kern", "sssp_driver", nodes, cptables);

    //Switched to a RunDescriptor so that checkpointing can be used.
    sp_rd.checkpoint_type = CP_CONTINUOUS;
    m.run_all(sp_rd);

    if (FLAGS_dump_nodes_periodic > 0) {
      fprintf(stderr, "Dumping SSSP lengths for iteration %d\n",i);
      char dump_fname[256];
      sprintf(dump_fname,"SSSP_dump_periodic_i%d",i);
      FILE* fh = fopen(dump_fname,"w");
      for (int i = 0; i < FLAGS_dump_nodes_periodic; ++i) {
        double d = distance_map->get(i);
        if (d >= 1000) {d = -1;}
        fprintf(fh, "%8d: %d\n", i, (int)d);
      }
      fclose(fh);
    }

  }

//Finish the timer!
  gettimeofday(&end_time, NULL);
  long long totaltime = (long long) (end_time.tv_sec - start_time.tv_sec)
      * 1000000 + (end_time.tv_usec - start_time.tv_usec);
  fprintf(stderr, "Total SSSP time: %.3f seconds\n", totaltime / 1000000.0);

  if (FLAGS_dump_nodes > 0) {
    fprintf(stderr, "Dumping SSSP lengths for final results\n");
    FILE* fh = fopen("SSSP_dump","w");
    /*PDontRunOne(distance_map, {*/
        for (int i = 0; i < FLAGS_dump_nodes; ++i) {
          double d = distance_map->get(i);
          if (d >= 1000) {d = -1;}
          fprintf(fh, "%8d: %d\n", i, (int)d);
        }
    /*});*/
    fclose(fh);
  }
  return 0;
}
REGISTER_RUNNER(ShortestPath);
