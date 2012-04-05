#include "client/client.h"
#include "examples/examples.pb.h"

#include <sys/time.h>
#include <sys/resource.h>
#include <algorithm>
#include <libgen.h>

using namespace piccolo;
using namespace std;

static int NUM_WORKERS = 2;
#define MAXCOST RAND_MAX

DEFINE_int32(left_vertices, 100, "Number of left-side vertices");
DEFINE_int32(right_vertices, 100, "Number of right-side vertices");
DEFINE_double(edge_probability, 0.5, "Probability of edge between vertices");
DEFINE_bool(edge_costs, false, "Set to true to have edges have costs");

static TypedGlobalTable<int, vector<int> >* leftoutedges = NULL;
static TypedGlobalTable<int, int>* leftmatches = NULL;
static TypedGlobalTable<int, int>* leftattemptmatches = NULL;
static TypedGlobalTable<int, int>* rightmatches = NULL;
static TypedGlobalTable<string, string>* StatsTable = NULL;

//-----------------------------------------------
namespace piccolo {
struct RightMatchSet: public Accumulator<int> {
  virtual ~RightMatchSet() {}
  void Accumulate(int *i, const int &j) {
    if (*i == -1)
      *i = j;
    return;
  }
};

template<> struct Marshal<vector<int> > : MarshalBase {
  static void marshal(const vector<int>& t, string *out) {
    int i, j;
    int len = t.size();
    out->append((char*) &len, sizeof(int));
    for (i = 0; i < len; i++) {
      j = t[i];
      out->append((char*) &j, sizeof(int));
    }
  }
  static void unmarshal(const StringPiece &s, vector<int>* t) {
    int i, j;
    int len;
    memcpy(&len, s.data, sizeof(int));
    t->clear();
    for (i = 0; i < len; i++) {
      memcpy(&j, s.data + (i + 1) * sizeof(int), sizeof(int));
      t->push_back(j);
    }
  }
};
}

//-----------------------------------------------
class BPMKernel: public DSMKernel {
public:
  virtual ~BPMKernel() {}
  void InitTables() {
    vector<int> v; //for left nodes' neighbors

    v.clear();

    leftmatches->resize(FLAGS_left_vertices);
    leftattemptmatches->resize(FLAGS_left_vertices);
    leftoutedges->resize(FLAGS_left_vertices);
    rightmatches->resize(FLAGS_right_vertices);
    for (int i = 0; i < FLAGS_left_vertices; i++) {
      leftmatches->update(i, -1);
      leftattemptmatches->update(i, -1);
      leftoutedges->update(i, v);
    }
    for (int i = 0; i < FLAGS_right_vertices; i++) {
      rightmatches->update(i, -1);
    }

    //One-off status value
    StatsTable->resize(leftmatches->num_shards());
    for (int i = 0; i < leftmatches->num_shards(); i++) {
      char key[32];
      sprintf(key, "quiescent%d", i);
      StatsTable->update(key, "t");
    }
    StatsTable->SendUpdates();
  }

  void PopulateLeft() {
    srand(current_shard());
    TypedTableIterator<int, vector<int> > *it =
        leftoutedges->get_typed_iterator(current_shard());
    CHECK(it != NULL);
    for (; !it->done(); it->Next()) {
      vector<int> v = it->value();
      //vector<int> v2 = it2->value();
      for (int i = 0; i < FLAGS_right_vertices; i++) {
        if ((float) rand() / (float) RAND_MAX < FLAGS_edge_probability) {
          v.push_back(i); //add neighbor
          //cost = ((FLAGS_edge_costs)?rand():(RAND_MAX));
        }
      }
      //store list of neighboring edges
      leftoutedges->update(it->key(), v);
    }
  }

  //Set a random right neighbor of each left vertex to be
  //matched.  If multiple lefts set the same right, the triggers
  //will sort it out.
  void BPMRoundLeft() {
    bool quiescent = true;

    char qkey[32];
    sprintf(qkey, "quiescent%d", current_shard());

    TypedTableIterator<int, vector<int> > *it =
        leftoutedges->get_typed_iterator(current_shard());
    TypedTableIterator<int, int> *it3 = leftmatches->get_typed_iterator(
        current_shard());
    for (; !it->done() && !it3->done(); it->Next(), it3->Next()) {

      vector<int> v = it->value(); //leftoutedges

      //only try to match if this left node is unmatched and has candidate right nodes,
      //represented respectively by leftmatches == -1 and v/v2 having >0 items.
      if (v.size() <= 0 || it3->value() != -1)
        continue;

      //don't stop until nothing happens in a round
      quiescent = false;

      //try to find a random or best match
      int j;

      //all edges equal; pick one at random
      j = v.size() * ((float) rand() / (float) RAND_MAX);
      j = (j >= v.size()) ? v.size() - 1 : j;
      j = v[j];

      VLOG(2)
          << "Attempted match: left " << it->key() << " <--> right " << j
             ;

      rightmatches->update(j, it->key());
      leftattemptmatches->update(it->key(), j);
    }
    StatsTable->update(qkey, (quiescent ? "t" : "f"));
    VLOG(2)
        << "Shard " << current_shard() << " is quiescent? "
            << StatsTable->get(qkey).c_str();
  }

  void BPMRoundRight() {

    TypedTableIterator<int, int> *it = rightmatches->get_typed_iterator(
        current_shard());
    for (; !it->done(); it->Next()) {
      if (it->value() != -1)
        leftmatches->update(it->value(), it->key());
    }
  }

		void BPMRoundCleanupLeft() {
			int i=0;
			TypedTableIterator<int,int> *it =
				 leftmatches->get_typed_iterator(current_shard());
			TypedTableIterator<int,int> *it2 =
				 leftattemptmatches->get_typed_iterator(current_shard());
			TypedTableIterator<int,vector<int> > *it3 =
				 leftoutedges->get_typed_iterator(current_shard());
			for(; !it->done() && !it2->done() && !it3->done(); it->Next(), it2->Next(), it3->Next()) {
				CHECK_EQ(it->key(),it2->key());
				CHECK_EQ(it2->key(),it3->key());
				//tried to set but got denied
				if (it->value() == -1 && it2->value() != -1) {
					vector<int> v = it3->value();
					vector<int>::iterator it = find(v.begin(), v.end(), it2->value());
					if (it != v.end()) {
						v.erase(it);
						leftattemptmatches->update(it2->key(),-1);
						leftoutedges->update(it3->key(),v);
						i++;
					} else
						LOG(FATAL) << "Cleanup failed!";
				}
			}
			VLOG(2) << "Cleaned up " << i << " left nodes.";
		}
			

		void EvalPerformance() {
			int left_matched=0, right_matched=0;

    for (int i = 0; i < leftmatches->num_shards(); i++) {
      TypedTableIterator<int, int> *it = leftmatches->get_typed_iterator(
          current_shard());
      for (; !it->done(); it->Next()) {
        if (it->value() != -1) {
          left_matched++;
          right_matched++;
        }
      }
    }

    printf("Performance: [LEFT]  %d of %d matched.\n", left_matched,
        FLAGS_left_vertices);
    printf("Performance: [RIGHT] %d of %d matched.\n", right_matched,
        FLAGS_right_vertices);
  }
};

//-----------------------------------------------

REGISTER_KERNEL(BPMKernel);
REGISTER_METHOD(BPMKernel, InitTables);
REGISTER_METHOD(BPMKernel, PopulateLeft);
REGISTER_METHOD(BPMKernel, BPMRoundLeft);
REGISTER_METHOD(BPMKernel, BPMRoundRight);
REGISTER_METHOD(BPMKernel, BPMRoundCleanupLeft);
REGISTER_METHOD(BPMKernel, EvalPerformance);

int Bipartmatch(const ConfigData& conf) {

  leftoutedges = CreateTable(0, conf.num_workers(), new Sharding::Mod,
      new Accumulators<vector<int> >::Replace);
  leftmatches = CreateTable(1, conf.num_workers(), new Sharding::Mod,
      new Accumulators<int>::Replace);
  leftattemptmatches = CreateTable(2, conf.num_workers(), new Sharding::Mod,
      new Accumulators<int>::Replace);
  rightmatches = CreateTable(3, conf.num_workers(), new Sharding::Mod,
      new RightMatchSet);
  StatsTable = CreateTable(10000, 1, new Sharding::String,
      new Accumulators<string>::Replace); //CreateStatsTable();

  StartWorker(conf);
  Master m(conf);

  if (FLAGS_edge_costs)
    LOG(FATAL) << "Edges with costs not properly implemented";

  NUM_WORKERS = conf.num_workers();
  printf("---- Initializing Bipartmatch on %d workers ----\n", NUM_WORKERS);

  //Fill in all necessary keys
  m.run_one("BPMKernel", "InitTables", leftoutedges);
  //Populate edges left<->right
  m.run_all("BPMKernel", "PopulateLeft", leftoutedges);
  m.barrier();

  struct timeval start_time, end_time;
  gettimeofday(&start_time, NULL);

  bool unstable;
  int invocations = 0;
  do {
    unstable = false;
    m.run_all("BPMKernel", "BPMRoundLeft", leftoutedges);
    m.run_all("BPMKernel", "BPMRoundRight", rightmatches);
    m.run_all("BPMKernel", "BPMRoundCleanupLeft", leftattemptmatches);
    for (int i = 0; i < conf.num_workers(); i++) {
      char qkey[32];
      sprintf(qkey, "quiescent%d", i);
      if (0 == strcmp(StatsTable->get(qkey).c_str(), "f"))
        unstable = true;
    }
    invocations++;
  } while (unstable);
  cout << "---- Completed in " << invocations
      << " triples of kernel invocations ----";
  gettimeofday(&end_time, NULL);
  long long totaltime = (long long) (end_time.tv_sec - start_time.tv_sec)
      * 1000000 + (end_time.tv_usec - start_time.tv_usec);
  cout << "Total matching time: " << ((double) (totaltime) / 1000000.0)
      << " seconds";

  m.run_one("BPMKernel", "EvalPerformance", leftmatches);

  return 0;
}
REGISTER_RUNNER(Bipartmatch);
