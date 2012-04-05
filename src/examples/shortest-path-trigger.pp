#include "examples/examples.h"
#include "kernel/disk-table.h"

using std::vector;

using namespace piccolo;

DEFINE_int32(tnum_nodes, 10000, "");
DEFINE_int32(tdump_nodes, 0, "");

static int NUM_WORKERS = 0;
static TypedGlobalTable<int, double>* distance_map;
static RecordTable<PathNode>* nodes_record;
static TypedGlobalTable<int, vector<double> >* nodes;

namespace piccolo {
template<> struct Marshal<vector<double> > : MarshalBase {
  static void marshal(const vector<double>& t, string *out) {
    int i;
    double j;
    int len = t.size();
    out->append((char*) &len, sizeof(int));
    for (i = 0; i < len; i++) {
      j = t[i];
      out->append((char*) &j, sizeof(double));
    }
  }
  static void unmarshal(const StringPiece &s, vector<double>* t) {
    int i;
    double j;
    int len;
    memcpy(&len, s.data, sizeof(int));
    if (len < 0) LOG(FATAL) << "Unmarshalled vector of size < 0";
    t->clear();
    for (i = 0; i < len; i++) {
      memcpy(&j, s.data + (i + 1) * sizeof(double), sizeof(double));
      t->push_back(j);
    }
  }
};
}
// This is the trigger. In order to experiment with non-trigger version,
// I limited the maximum distance will be 20.

struct SSSPTrigger: public HybridTrigger<int, double> {
public:
  bool Accumulate(double* a, const double& b) {
    if (*a <= b)		//not better
		return false;
    *a = b;
    return true;
  }
  bool LongFire(const int key, bool lastrun) {
    double distance = distance_map->get(key);
    vector<double> thisnode = nodes->get(key);
    vector<double>::iterator it = thisnode.begin();
    for (; it != thisnode.end(); it++)
      if ((*it) != key)
        distance_map->update((*it), distance + 1);
    return false;
  }
};

static void BuildGraph(int shards, int nodes, int density) {
  vector<RecordFile*> out(shards);
  File::Mkdirs("testdata/");
  for (int i = 0; i < shards; ++i) {
    out[i] = new RecordFile(
        StringPrintf("testdata/sp-graph.rec-%05d-of-%05d", i, shards), "w");
  }

  srandom(nodes);	//repeatable graphs
  fprintf(stderr, "Building graph with %d nodes and %d shards:\n", nodes, shards);

  for (int i = 0; i < nodes; i++) {
    PathNode n;
    n.set_id(i);

    for (int j = 0; j < density; j++) {
      n.add_target(random() % nodes);
    }

    out[i % shards]->write(n);
    if (i % (nodes / 50) == 0) {
      fprintf(stderr, ".");
    }
  }
  fprintf(stderr, "\n");

  for (int i = 0; i < shards; ++i) {
    delete out[i];
  }
}

class ssspt_kern : public DSMKernel {
public:
  void ssspt_driver() {
    if (current_shard() == 0) {
      distance_map->update(0, 0);
    }
  }
};

REGISTER_KERNEL(ssspt_kern);
REGISTER_METHOD(ssspt_kern,ssspt_driver);

int ShortestPathTrigger(const ConfigData& conf) {
  NUM_WORKERS = conf.num_workers();

  distance_map = CreateTable(0, FLAGS_shards, new Sharding::Mod,
                             new Accumulators<double>::Replace, 1);
    if (!FLAGS_build_graph) {
  	  nodes_record = CreateRecordTable<PathNode>(1, "testdata/sp-graph.rec*", false);
    }
  nodes = CreateTable(2, FLAGS_shards, new Sharding::Mod,
                      new Accumulators<vector<double> >::Replace);
  //TriggerID trigid = distance_map->register_trigger(new SSSPTrigger);

  StartWorker(conf);
  Master m(conf);

  if (FLAGS_build_graph) {
    BuildGraph(FLAGS_shards, FLAGS_tnum_nodes, 4);
    return 0;
  }

  if (!m.restore()) {
    distance_map->resize(FLAGS_tnum_nodes);
    nodes->resize(FLAGS_tnum_nodes);
    PRunAll(distance_map, {
       vector<double> v;
       v.clear();
       for(int i=current_shard();i<FLAGS_tnum_nodes;i+=FLAGS_shards) {
         distance_map->update(i, 1e9);	//Initialize all distances to very large.
         nodes->update(i,v);	//Start all vectors with empty adjacency lists
       }
    });


    //Build adjacency lists by appending RecordTables' contents
    PMap({n: nodes_record}, {
      vector<double> v=nodes->get(n.id());
      for(int i=0; i < n.target_size(); i++) {
        v.push_back(n.target(i));
		nodes->update(n.id(),v);
      }
	});

	PRunAll(distance_map, {
	  distance_map->swap_accumulator((Trigger<int,double>*)new SSSPTrigger);
	});
  }

  //Start the timer!
  struct timeval start_time, end_time;
  gettimeofday(&start_time, NULL);

    //Start it all up by poking the thresholding process with a "null" update on the newly-initialized nodes.
    vector<int> cptables;
	cptables.clear();	
	cptables.push_back(0);
	cptables.push_back(2);

    RunDescriptor pr_rd("ssspt_kern", "ssspt_driver", distance_map, cptables);

    //Switched to a RunDescriptor so that checkpointing can be used.
    pr_rd.checkpoint_type = CP_CONTINUOUS;
    m.run_all(pr_rd);

  PRunAll(distance_map, {
    distance_map->swap_accumulator(new Triggers<int,double>::NullTrigger);
  });

  //Finish the timer!
  gettimeofday(&end_time, NULL);
  long long totaltime = (long long) (end_time.tv_sec - start_time.tv_sec)
      * 1000000 + (end_time.tv_usec - start_time.tv_usec);
  fprintf(stderr, "Total SSSP time: %.3f seconds \n", totaltime / 1000000.0);

  if (FLAGS_tdump_nodes > 0) {
    FILE* fh = fopen("SSSPT_dump","w");
    /*PDontRunOne(distance_map, {*/
      for (int i = 0; i < FLAGS_tdump_nodes; ++i) {
        int d = (int)distance_map->get(i);
        if (d >= 1000) {d = -1;}
        fprintf(fh, "%8d:\t%3d\n", i, d);
      }
    /*});*/
    fclose(fh);
  }
  return 0;
}
REGISTER_RUNNER(ShortestPathTrigger);
