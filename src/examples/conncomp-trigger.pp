#include "examples/examples.h"
#include "kernel/disk-table.h"

using std::vector;

using namespace piccolo;

DEFINE_int32(tcc_num_nodes, 10000, "");
DEFINE_int32(tcc_dump_nodes, 0, "");
DEFINE_bool(tcc_directed, false, "Set to change edges to be directed");

static int NUM_WORKERS = 0;
static TypedGlobalTable<int, int>* maxcomp_map;
static RecordTable<PathNode>* nodes_record;
static TypedGlobalTable<int, vector<int> >* nodes;

namespace piccolo {
template<> struct Marshal<vector<int> > : MarshalBase {
  static void marshal(const vector<int>& t, string *out) {
    int i,j;
    int len = t.size();
    out->append((char*) &len, sizeof(int));
    for (i = 0; i < len; i++) {
      j = t[i];
      out->append((char*) &j, sizeof(int));
    }
  }
  static void unmarshal(const StringPiece &s, vector<int>* t) {
    int i,j;
    int len;
    memcpy(&len, s.data, sizeof(int));
    if (len < 0) LOG(FATAL) << "Unmarshalled vector of size < 0";
    t->clear();
    for (i = 0; i < len; i++) {
      memcpy(&j, s.data + (i + 1) * sizeof(int), sizeof(int));
      t->push_back(j);
    }
  }
};
}
// This is the trigger. In order to experiment with non-trigger version,
// I limited the maximum distance will be 20.

struct cctrigger: public HybridTrigger<int, int> {
public:
  bool Accumulate(int* a, const int& b) {
    if (*a >= b)		//new is not bigger
		return false;
    *a = b;
    return true;
  }
  bool LongFire(const int key, bool lastrun) {
    int maxcomp = maxcomp_map->get(key);
    vector<int> thisnode = nodes->get(key);
    vector<int>::iterator it = thisnode.begin();
    for (; it != thisnode.end(); it++)
      if ((*it) != key)
        maxcomp_map->update((*it), maxcomp);
    return false;
  }
};

static void BuildGraph(int shards, int nodes, int density) {
  vector<RecordFile*> out(shards);
  File::Mkdirs("testdata/");
  for (int i = 0; i < shards; ++i) {
    out[i] = new RecordFile(
        StringPrintf("testdata/cc-graph.rec-%05d-of-%05d", i, shards), "w");
  }

  srandom(nodes);	//repeatable graphs
  fprintf(stderr, "Building graph with %d nodes and %d shards:\n", nodes, shards);

  int edges = 0;

  if (FLAGS_tcc_directed) {
    for (int i = 0; i < nodes; i++) {
      PathNode n;
      n.set_id(i);
  
      for (int j = 0; j < random() % density; j++) {
        n.add_target(random() % nodes);
        edges++;
      }
  
      out[i % shards]->write(n);
      if (i % (nodes / 50) == 0) {
        fprintf(stderr, ".");
      }
    }
  } else {
    // Pass 1: edge generation
    vector< vector<int> > edgestage;
    edgestage.resize(nodes);
    for (int i = 0; i < nodes; i++) {
      for (int j = 0; j < random() % density; j++) {
	    int neighbor = random() % nodes;
        edgestage[neighbor].push_back(i);
        edgestage[i].push_back(neighbor);
        edges++;
      }
  
      if (i % (nodes / 50) == 0) {
        fprintf(stderr, ",");
      }
    }
    // Pass 2: write to file
    for (int i = 0; i < nodes; i++) {
      PathNode n;
      n.set_id(i);
  
      for (int j = 0; j < edgestage[i].size(); j++) {
        n.add_target(edgestage[i][j]);
      }
  
      out[i % shards]->write(n);
      if (i % (nodes / 50) == 0) {
        fprintf(stderr, ".");
      }
    }
  }
  fprintf(stderr, "\n");

  for (int i = 0; i < shards; ++i) {
    delete out[i];
  }
  fprintf(stderr,"Generated graph with %d vertices and %d edges\n",nodes,edges);
}


class cct_kern: public DSMKernel {
public:
  void cct_driver() {
    TypedGlobalTable<int,int>::Iterator *it =
      maxcomp_map->get_typed_iterator(current_shard());
    for(; !it->done(); it->Next()) {
      maxcomp_map->update(it->key(),it->key());
    }
  }
};

REGISTER_KERNEL(cct_kern);
REGISTER_METHOD(cct_kern,cct_driver);

int ShortestPathTrigger(const ConfigData& conf) {
  NUM_WORKERS = conf.num_workers();

  maxcomp_map = CreateTable(0, FLAGS_shards, new Sharding::Mod,
                             new Accumulators<int>::Replace, 1);
    if (!FLAGS_build_graph) {
  	  nodes_record = CreateRecordTable<PathNode>(1, "testdata/cc-graph.rec*", false);
    }
  nodes = CreateTable(2, FLAGS_shards, new Sharding::Mod,
                      new Accumulators<vector<int> >::Replace);

  StartWorker(conf);
  Master m(conf);

  if (FLAGS_build_graph) {
    BuildGraph(FLAGS_shards, FLAGS_tcc_num_nodes, 145);
    return 0;
  }

  if (!m.restore()) {
    maxcomp_map->resize(FLAGS_tcc_num_nodes);
    nodes->resize(FLAGS_tcc_num_nodes);
    PRunAll(maxcomp_map, {
       vector<int> v;
       v.clear();
       for(int i=current_shard();i<FLAGS_tcc_num_nodes;i+=FLAGS_shards) {
         maxcomp_map->update(i, -1);	//Initialize all distances to min_ID-1
         nodes->update(i,v);	//Start all vectors with empty adjacency lists
       }
    });


    //Build adjacency lists by appending RecordTables' contents
    PMap({n: nodes_record}, {
      vector<int> v=nodes->get(n.id());
      for(int i=0; i < n.target_size(); i++) {
        v.push_back(n.target(i));
		nodes->update(n.id(),v);
      }
	});

	PRunAll(maxcomp_map, {
	  maxcomp_map->swap_accumulator((Trigger<int,int>*)new cctrigger);
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

    RunDescriptor pr_rd("cct_kern", "cct_driver", maxcomp_map, cptables);

    //Switched to a RunDescriptor so that checkpointing can be used.
    pr_rd.checkpoint_type = CP_CONTINUOUS;
    m.run_all(pr_rd);

  PRunAll(maxcomp_map, {
    maxcomp_map->swap_accumulator(new Triggers<int,int>::NullTrigger);
  });

  //Finish the timer!
  gettimeofday(&end_time, NULL);
  long long totaltime = (long long) (end_time.tv_sec - start_time.tv_sec)
      * 1000000 + (end_time.tv_usec - start_time.tv_usec);
  fprintf(stderr, "Total CC time: %.3f seconds \n", totaltime / 1000000.0);

  if (FLAGS_tcc_dump_nodes > 0) {
    FILE* fh = fopen("CCT_dump","w");
    /*PDontRunOne(distance_map, {*/
      for (int i = 0; i < FLAGS_tcc_dump_nodes; ++i) {
        int d = (int)maxcomp_map->get(i);
        fprintf(fh, "%8d:\t%3d\n", i, d);
      }
    /*});*/
    fclose(fh);
  }
  return 0;
}
REGISTER_RUNNER(ShortestPathTrigger);
