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
//#define DEBUGOUT

DEFINE_int32(tleft_vertices, 400, "Number of left-side vertices");
DEFINE_int32(tright_vertices, 400, "Number of right-side vertices");
DEFINE_double(tedge_probability, 0.5, "Probability of edge between vertices");
DEFINE_bool(tedge_costs, false, "Set to true to have edges have costs");

static TypedGlobalTable<int, vector<int> >* leftoutedges = NULL;
static TypedGlobalTable<int, vector<int> >* leftoutcosts = NULL;
static TypedGlobalTable<int, int>* leftmatches = NULL;
static TypedGlobalTable<int, int>* rightmatches = NULL;
static TypedGlobalTable<int, int>* rightcosts = NULL;

//-----------------------------------------------
namespace piccolo {
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
    if (len < 0)
      LOG(FATAL) << "Unmarshalled vector of size < 0";
    t->clear();
    for (i = 0; i < len; i++) {
      memcpy(&j, s.data + (i + 1) * sizeof(int), sizeof(int));
      t->push_back(j);
    }
  }
};
}

//-----------------------------------------------

struct MatchRequestTrigger: public Trigger<int, int> {
public:
  void Fire(const int* key, int* value, const int& newvalue, bool* doUpdate,
      bool isNew) {
    int newcost = MAXCOST;

#ifdef DEBUGOUT
    /*
     if (rightmatches->get(key) != value) {
     cout << "KEY MISMATCH RIGHT IN TRIGGER: [" <<
     key << ",({" << value << " vs " << rightmatches->get(key)
     << "}, " << newvalue << ")]" <<  endl;
     exit(-1);
     }
     */
#endif

    if (newvalue != -1) {
    }
    if (*value != -1) {
#ifdef DEBUGOUT
      cout << "Denying match on " << *key << " from " << newvalue;
#endif
      leftmatches->enqueue_update(newvalue, -1);
      *doUpdate = false;
      return;
      //		return false;
    } else {
      //Else this match is acceptable.  Set new cost.
#ifdef DEBUGOUT
      cout << "Accepting match on " << *key << " from " << newvalue;
#endif
      rightcosts->enqueue_update(*key, newcost);
      leftmatches->enqueue_update(newvalue, *key);
    }
    *value = newvalue;
    *doUpdate = true;
    return;
//			return true;
  }

  //No longfire
  bool LongFire(const int key, bool) {
    volatile bool rv = false;
    return rv;
  }
};

struct LeftTrigger: public Trigger<int, int> {
public:
  void Fire(const int* key, int* value, const int& newvalue, bool* doUpdate,
      bool isNew) {

    //Sanity check: make sure the right side isn't trying to
    //break an already-agreed match or re-assign a left vertex
    //that's already linked.
#ifdef DEBUGOUT
    /*
     if (leftmatches->get(key) != value || value != -1) {
     cout << "KEY MISMATCH LEFT IN TRIGGER: [" <<
     key << ",(" << value << ", " << newvalue << ")]" <<  endl;
     exit(-1);
     }
     */
#endif

    //Don't store the denial!
    if (newvalue == -1) {

      //Denied: remove possible right match
      //Also, this is always a local get(), so no possible consistency issues
      vector<int> v = leftoutedges->get(*key);

      //vector<int>::iterator it = v.begin();

#ifdef DEBUGOUT
      cout << "Match on " << *key << " denied from " << *(v.end()-1);
#endif

      v.erase(v.end() - 1);

      int j = -1;
      if (v.size() != 0) {
        //try to find a random or best match
        //all edges equal; pick one at random
        j = v.size() * ((float) rand() / (float) RAND_MAX);
        j = (j >= v.size()) ? v.size() - 1 : j;
        int j2 = v[j];
        v.erase(v.begin() + j);
        v.push_back(j2);
        j = j2;
      }

      //Enqueue the removal
      leftoutedges->enqueue_update((int) *key, v);

      if (v.size() == 0) { //forget it if no more candidates
#ifdef DEBUGOUT
          cout << "Ran out of right candidates for " << *key;
#endif
        *value = newvalue;
        *doUpdate = true;
        return;
//					return true;
      }

      rightmatches->enqueue_update(j, *key);
      *value = j;
#ifdef DEBUGOUT
      cout << "Re-attempting from " << *key << " to " << j;
#endif
      *doUpdate = false;
      return;
//				return false;
    }

    //It was not a denial; store it.
    *value = newvalue;
#ifdef DEBUGOUT
    cout << "Storing accepting match on left " << *key << " from right " << value;
#endif
    *doUpdate = true;
    return;
//			return true;
  }

  //No longfire
  bool LongFire(const int key, bool) {
    volatile bool rv = false;
    return rv;
  }
};

//-----------------------------------------------

class BPMTKernel: public DSMKernel {
public:
  virtual ~BPMTKernel() {}
  void InitTables() {
    vector<int> v; //for left nodes' neighbors
    vector<int> v2; //for left nodes' edge costs

    v.clear();
    v2.clear();

    leftmatches->resize(FLAGS_tleft_vertices);
    rightmatches->resize(FLAGS_tright_vertices);
    leftoutedges->resize(FLAGS_tleft_vertices);
    leftoutcosts->resize(FLAGS_tleft_vertices);
    for (int i = 0; i < FLAGS_tleft_vertices; i++) {
      leftmatches->update(i, -1);
      leftoutedges->update(i, v);
      leftoutcosts->update(i, v2);
    }
    for (int i = 0; i < FLAGS_tright_vertices; i++) {
      rightmatches->update(i, -1);
      rightcosts->update(i, MAXCOST);
    }

  }

  void PopulateLeft() {
    srand(current_shard());
    TypedTableIterator<int, vector<int> > *it =
        leftoutedges->get_typed_iterator(current_shard());
    CHECK(it != NULL);
    TypedTableIterator<int, vector<int> > *it2 =
        leftoutcosts->get_typed_iterator(current_shard());
    CHECK(it2 != NULL);
    int cost = 0;
    for (; !it->done() && !it2->done(); it->Next(), it2->Next()) {

      if (leftmatches->get(it->key()) != -1) {
        LOG(FATAL) << "Uninitialized left match found!";
      }

      vector<int> v = it->value();
      vector<int> v2 = it2->value();
      for (int i = 0; i < FLAGS_tright_vertices; i++) {
        if ((float) rand() / (float) RAND_MAX < FLAGS_tedge_probability) {
          v.push_back(i); //add neighbor
          cost = ((FLAGS_tedge_costs) ? rand() : (RAND_MAX));
          v2.push_back(cost);
        }
      }

      //try to find a random or best match
      int j;
      if (FLAGS_tedge_costs) {
        //edges have associated costs
        vector<int>::iterator inner_it = v.begin();
        vector<int>::iterator inner_it2 = v2.begin();
        j = -1;
        if (inner_it == v.end() || inner_it2 == v2.end()) {
          float mincost = MAXCOST;
          int offset = -1;
          for (; inner_it != v.end() && inner_it2 != v2.end();
              inner_it++, inner_it2++) {
            if ((*inner_it2) < mincost) {
              mincost = *inner_it2;
              j = *inner_it;
              offset = inner_it - v.begin();
            }
          }
          v.erase(v.begin() + offset);
          v2.erase(v2.begin() + offset);
          v.push_back(j);
          v2.push_back(mincost);
        }
      } else {
        //all edges equal; pick one at random
        if (v.size() != 0) {
          j = v.size() * ((float) rand() / (float) RAND_MAX);
          j = (j >= v.size()) ? v.size() - 1 : j;
          int j2 = v[j];
          v.erase(v.begin() + j);
          v.push_back(j2);
          j = j2;
        } else {
          j = -1;
        }
      }
      //Note: the above code used to be in BeginBPMT.  It got
      //moved so that the trigger on leftoutedges wouldn't get
      //triggered when the best match was put at the end of each
      //vector.  CRM 4/12/2011

      leftoutedges->update(it->key(), v); //store list of neighboring edges
      leftoutcosts->update(it2->key(), v2); //store list of neighbor edge costs
      VLOG(2) << "Populated left vertex " << it->key();
    }
  }

  //Set a random right neighbor of each left vertex to be
  //matched.  If multiple lefts set the same right, the triggers
  //will sort it out.
  void BeginBPMT() {
    TypedTableIterator<int, vector<int> > *it =
        leftoutedges->get_typed_iterator(current_shard());
    TypedTableIterator<int, vector<int> > *it2 =
        leftoutcosts->get_typed_iterator(current_shard());
    for (; !it->done() && !it2->done(); it->Next(), it2->Next()) {
      vector<int> v = it->value();
      vector<int> v2 = it2->value();
      if (v.size() <= 0)
        continue;

      int j = *(v.end() - 1);
#ifdef DEBUGOUT
      cout << "Attempted match: left " << it->key() << " <--> right " << j;
#endif
      rightmatches->update(j, it->key());
      //leftmatches->update(it->key(),j);
    }
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

			printf("Performance: [LEFT]  %d of %d matched.\n",left_matched,FLAGS_tleft_vertices);
			printf("Performance: [RIGHT] %d of %d matched.\n",right_matched,FLAGS_tright_vertices);
		}

  void EnableTriggers() {
    leftmatches->swap_accumulator((Trigger<int, int>*) new LeftTrigger);
    rightmatches->swap_accumulator(
        (Trigger<int, int>*) new MatchRequestTrigger);
  }
  void DisableTriggers() {
    leftmatches->swap_accumulator(new Triggers<int, int>::NullTrigger);
    rightmatches->swap_accumulator(new Triggers<int, int>::NullTrigger);
  }
};

//-----------------------------------------------

REGISTER_KERNEL(BPMTKernel);
REGISTER_METHOD(BPMTKernel, InitTables);
REGISTER_METHOD(BPMTKernel, PopulateLeft);
REGISTER_METHOD(BPMTKernel, BeginBPMT);
REGISTER_METHOD(BPMTKernel, EvalPerformance);

REGISTER_METHOD(BPMTKernel, EnableTriggers);
REGISTER_METHOD(BPMTKernel, DisableTriggers);

int Bipartmatch_trigger(const ConfigData& conf) {

  leftoutedges = CreateTable(0, conf.num_workers(), new Sharding::Mod,
      new Accumulators<vector<int> >::Replace);
  leftmatches = CreateTable(1, conf.num_workers(), new Sharding::Mod,
      new Triggers<int, int>::NullTrigger);
  leftoutcosts = CreateTable(3, conf.num_workers(), new Sharding::Mod,
      new Accumulators<vector<int> >::Replace);
  rightmatches = CreateTable(2, conf.num_workers(), new Sharding::Mod,
      new Triggers<int, int>::NullTrigger); //NullTrigger accepts everything
  rightcosts = CreateTable(4, conf.num_workers(), new Sharding::Mod,
      new Accumulators<int>::Replace);

//	TriggerID matchreqid = rightmatches->register_trigger(new MatchRequestTrigger);
//	TriggerID lefttriggerid = leftmatches->register_trigger(new LeftTrigger);

  StartWorker(conf);
  Master m(conf);

  NUM_WORKERS = conf.num_workers();
  printf("---- Initializing Bipartmatch-trigger on %d workers ----\n",
      NUM_WORKERS);

  //Disable triggers
//	m.enable_trigger(matchreqid,2,false);
//	m.enable_trigger(lefttriggerid,1,false);

  if (!m.restore()) {
    //Fill in all necessary keys
    m.run_one("BPMTKernel", "InitTables", leftoutedges);
    //Populate edges left<->right
    m.run_all("BPMTKernel", "PopulateLeft", leftoutedges);
    m.barrier();

    //Enable triggers
    m.run_all("BPMTKernel", "EnableTriggers", leftmatches);
    //m.enable_trigger(matchreqid,2,true);
    //m.enable_trigger(lefttriggerid,1,true);
    m.barrier();

    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);

    RunDescriptor cp_rd("BPMTKernel", "BeginBPMT", leftoutedges);
    cp_rd.checkpoint_type = CP_CONTINUOUS;
    m.run_all(cp_rd);

    gettimeofday(&end_time, NULL);
    long long totaltime = (long long) (end_time.tv_sec - start_time.tv_sec)
        * 1000000 + (end_time.tv_usec - start_time.tv_usec);
    cout << "Total matching time: " << ((double) (totaltime) / 1000000.0)
        << " seconds";

    //Disable triggers
    m.run_all("BPMTKernel", "DisableTriggers", leftmatches);
  }

  m.run_one("BPMTKernel", "EvalPerformance", leftmatches);

  return 0;
}
REGISTER_RUNNER(Bipartmatch_trigger);
