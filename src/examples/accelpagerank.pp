// Accelerated PageRank Runner for Oolong/Piccolo
#include "examples/examples.h"
#include "external/webgraph/webgraph.h"

#include <algorithm>
#include <libgen.h>

using namespace piccolo;
using namespace std;

static float TOTALRANK = 0;
static int NUM_WORKERS = 2;

static const float kPropagationFactor = 0.8;
static const int kBlocksize = 1000;
static const char kTestPrefix[] = "/scratch/kerm/pr-graph.rec";

//Not an option for this runner
//DEFINE_bool(apr_memory_graph, false,
//            "If true, the web graph will be generated on-demand.");

DEFINE_string(apr_graph_prefix, kTestPrefix, "Path to web graph.");
DEFINE_int32(apr_nodes, 10000, "");
DEFINE_double(apr_tol, 1.5e-8, "threshold for updates");
DEFINE_double(apr_d, kPropagationFactor, "alpha/restart probability");
DEFINE_int32(ashow_top, 10, "number of top results to display");
DEFINE_int32(machines, 1, "number of physical filesystem(s), assuming round-robin MPI schedulinG");

#define PREFETCH 1024

DEFINE_string(apr_convert_graph, "", "Path to WebGraph .graph.gz database to convert");

static float powerlaw_random(float dmin, float dmax, float n) {
  float r = (float) random() / RAND_MAX;
  return pow((pow(dmax, n) - pow(dmin, n)) * pow(r, 3) + pow(dmin, n), 1.0 / n);
}

static float random_restart_seed() {
  return (1 - kPropagationFactor) * (TOTALRANK / FLAGS_apr_nodes);
}

// I'd like to use a pair here, but for some reason they fail to count
// as POD types according to C++.  Sigh.
struct APageId {
  int64_t site :32;
  int64_t page :32;
};

struct APageInfo {
  string url;
  vector<APageId> adj;
};

//-----------------------------------------------
// Marshalling for APageInfo type
//-----------------------------------------------
namespace piccolo {
template<> struct Marshal<APageInfo> : MarshalBase {
  static void marshal(const APageInfo& t, string *out) {
    int i;
    struct APageId j;
    int len = t.url.size();
    out->append((char*) &len, sizeof(int));
    out->append(t.url.c_str(), len);
    len = t.adj.size();
    out->append((char*) &len, sizeof(int));
    for (i = 0; i < len; i++) {
      j = t.adj[i];
      out->append((char*) &j, sizeof(struct APageId));
    }
  }
  static void unmarshal(const StringPiece &s, APageInfo* t) {
    int i;
    struct APageId j;
    int len, len2;
    memcpy(&len, s.data, sizeof(int));
    t->url.clear();
    t->url.append(s.data + 1 * sizeof(int), len);
    t->adj.clear();
    memcpy(&len2, s.data + 1 * sizeof(int) + len, sizeof(int));
    for (i = 0; i < len2; i++) {
      memcpy(&j, s.data + (2) * sizeof(int) + len + i * sizeof(struct APageId),
             sizeof(struct APageId));
      t->adj.push_back(j);
    }
  }
};
}

bool operator==(const APageId& a, const APageId& b) {
  return a.site == b.site && a.page == b.page;
}

std::ostream& operator<<(std::ostream& s, const APageId& a) {
  return s << a.site << ":" << a.page;
}

struct AccelPRStruct {
  int64_t L :32; //number of outgoing links
  float pr_int; //internal pagerank
  float pr_ext; //external pagerank
};

std::ostream& operator<<(std::ostream& s, const struct AccelPRStruct& a) {
  return s << "[" << a.L << "]<-(" << a.pr_int << "," << a.pr_ext << ")";
}

namespace std {
namespace tr1 {
template<>
struct hash<APageId> {
  size_t operator()(const APageId& p) const {
    //EVERY_N(100000, fprintf(stderr, "Hashing %d, %d\n", p.site, p.page));
    return SuperFastHash((const char*) &p, sizeof p);
  }
};
}
}

struct SiteSharding: public Sharder<APageId> {
  int operator()(const APageId& p, int nshards) {
    return p.site % nshards;
  }
};

struct APageIdBlockInfo: public BlockInfo<APageId> {
  APageId start(const APageId& k, int block_size) {
    APageId p = { k.site, k.page - (k.page % block_size) };
    return p;
  }

  int offset(const APageId& k, int block_size) {
    return k.page % block_size;
  }
};

TypedGlobalTable<APageId, AccelPRStruct> *prs;
TypedGlobalTable<APageId, APageInfo>* apages;
DiskTable<uint64_t, Page> *pagedb;

static vector<int> InitSites() {
  static vector<int> site_sizes;
  srand(0);
  for (int n = 0; n < FLAGS_apr_nodes;) {
    int c = powerlaw_random(
        1, min(50000, (int) (100000. * FLAGS_apr_nodes / 100e6)), 0.001);
    c = (c==0)?1:c;
    site_sizes.push_back(c);
    n += c;
  }
  return site_sizes;
}

static vector<int> site_sizes;

static void BuildGraph(int oldshard, int nshards, int nodes, int density) {
  char* d = strdup(FLAGS_apr_graph_prefix.c_str());
  File::Mkdirs(dirname(d));

  if (oldshard >= FLAGS_machines) return;
  for (int shard = 0; shard < nshards; shard += 1) {
    string target = StringPrintf("%s-%05d-of-%05d-N%05d", 
                                 FLAGS_apr_graph_prefix.c_str(), shard, nshards,
                                 nodes);

    if (File::Exists(target)) {
      continue;
    }

    srand(shard);
    Page n;
    RecordFile out(target, "w", RecordFile::NONE);

    for (int i = shard; i < site_sizes.size(); i += nshards) {
    PERIODIC(
        1,
        LOG(INFO) << "Working: Shard -- " << shard << " of " << nshards
            << "; site " << i << " with size " << site_sizes[i] << " of " << site_sizes.size());
    for (int j = 0; j < site_sizes[i]; ++j) {
      n.Clear();
      n.set_site(i);
      n.set_id(j);
      for (int k = 0; k < ((density>nodes)?nodes:density); k++) {
        int target_site =
            (random() % 10 != 0) ? i : (random() % site_sizes.size());
        n.add_target_site(target_site);
        n.add_target_id(random() % site_sizes[target_site]);
      }

        out.write(n);
      }
    }
  }
}

static void WebGraphAPageIds(WebGraph::Reader *wgr, vector<APageId> *out) {
  WebGraph::URLReader *r = wgr->newURLReader();

  map<string, struct APageId> hosts;
  map<string, struct APageId>::iterator it;
  struct APageId pid = { -1, -1 };
  string prev, url, host, prevhost;
  int i = 0;
  int maxsite = 0;

  out->reserve(wgr->nodes);

  while (r->readURL(&url)) {
    if (i++ % 100000 == 0)
      LOG(INFO) << "Reading URL " << i + 1 << " of " << wgr->nodes;

    // Get host part
    int hostLen = url.find('/', 8);
    CHECK(hostLen != url.npos) << "Failed to split host in URL " << url;
    ++hostLen;

    host = url.substr(7, hostLen - 7);

    if (0 == host.compare(prevhost)) {
      // Existing site.  Pid.site already set. it already set
      pid.page = it->second.page++;
    } else if ((it = hosts.find(host)) != hosts.end()) {
      // Existing site
      pid.page = it->second.page++;
      pid.site = it->second.site;
    } else {
      // Different site
      pid.page = 0;
      pid.site = maxsite;
      struct APageId newpid = { maxsite++, 1 };
      hosts.insert(pair<string, struct APageId>(host, newpid));
      //LOG(INFO) << "Host " << host << " is new site # " << pid.site;

      swap(prev, url);
    }

    out->push_back(pid);
  }

  delete r;

  LOG(INFO) << "URLReader: " << maxsite << " total sites read containing " << i
      << " nodes";
}

static void ConvertGraph(string path, int nshards) {
  WebGraph::Reader r(path);
  vector<APageId> APageIds;
  WebGraphAPageIds(&r, &APageIds);

  char* d = strdup(FLAGS_apr_graph_prefix.c_str());
  File::Mkdirs(dirname(d));

  RecordFile *out[nshards];
  for (int i = 0; i < nshards; ++i) {
    string target = StringPrintf("%s-%05d-of-%05d-N%05d",
                                 FLAGS_apr_graph_prefix.c_str(), i, nshards,
                                 r.nodes);
    out[i] = new RecordFile(target, "w", RecordFile::NONE);
  }

  // XXX Maybe we should take at most FLAGS_apr_nodes nodes
  const WebGraph::Node *node;
  Page n;
  while ((node = r.readNode())) {
    if (node->node % 100000 == 0)
      LOG(INFO) << "Reading node " << node->node << " of " << r.nodes;
    APageId src = APageIds.at(node->node);
    n.Clear();
    n.set_site(src.site);
    n.set_id(src.page);
    for (unsigned int i = 0; i < node->links.size(); ++i) {
      APageId dest = APageIds.at(node->links[i]);
      n.add_target_site(dest.site);
      n.add_target_id(dest.page);
    }
    out[src.site % nshards]->write(n);
  }

  for (int i = 0; i < nshards; ++i)
    delete out[i];
}

namespace piccolo {
struct AccelPRTrigger: public HybridTrigger<APageId, AccelPRStruct> {
public:
  bool Accumulate(AccelPRStruct* a, const AccelPRStruct& b) {
    //LOG(INFO) << "Accumulating " << b << " into " << *a;
    a->pr_int += (FLAGS_apr_d * b.pr_int); // / (float)b.L;
    a->pr_ext += b.pr_ext;
    return (b.pr_ext == 0);		//don't propagate if we're just setting our ext = int
  }
  bool LongFire(const APageId key, bool lastrun) {
    //if (!lastrun) {
      AccelPRStruct value = prs->get(key);
      if (abs(value.pr_int-value.pr_ext) >= FLAGS_apr_tol) { //BAD || newvalue.pr_ext == 0.0) {
        //VLOG(2) << "LongFire propagate on key " << key.site << ":" << key.page;
        // Get neighbors
        APageInfo p = apages->get(key);
        struct AccelPRStruct updval = { 0, (value.pr_int
            - value.pr_ext)/((float)value.L), 0 };

        //Tell everyone about our delta PR
        vector<APageId>::iterator it = p.adj.begin();
        for (; it != p.adj.end(); it++) {
          struct APageId neighbor = { it->site, it->page };
          //prs->enqueue_update(neighbor, updval);
          if (!(neighbor == key))
            prs->update(neighbor, updval);
        }

        //Update our own external PR
        updval.pr_int = 0;	//therefore we don't have to do anything about L.
        updval.pr_ext = value.pr_int-value.pr_ext;
        prs->update(key,updval);
      }
    
    return false;
  }
};
}
;

//fake kernel to force a checkpoint
class aprcp_kern: public DSMKernel {
public:
  void aprcp_kern_cp() {
    volatile int i = 0;
    i++;
  }
};

REGISTER_KERNEL(aprcp_kern);
REGISTER_METHOD(aprcp_kern,aprcp_kern_cp);
int AccelPagerank(const ConfigData& conf) {
  site_sizes = InitSites();

  NUM_WORKERS = conf.num_workers();
  TOTALRANK = FLAGS_apr_nodes;

  prs = CreateTable(0, FLAGS_shards, new SiteSharding,
                    (Trigger<APageId, AccelPRStruct>*) new AccelPRTrigger, 1);

  //no RecordTable option in this runner, MemoryTable only
  apages = CreateTable(1, FLAGS_shards, new SiteSharding,
                       new Accumulators<APageInfo>::Replace);

  //Also need to load pages
  if (FLAGS_apr_convert_graph.empty() && !FLAGS_build_graph) {
    LOG(INFO) << "Loading page database";
    pagedb = CreateRecordTable<Page>(2, FLAGS_apr_graph_prefix + "*", false, FLAGS_shards);
  }

  StartWorker(conf);
  Master m(conf);
  

  //Graph construction functions
  if (!FLAGS_apr_convert_graph.empty()) {
    ConvertGraph(FLAGS_apr_convert_graph, FLAGS_shards);					//convert a real graph
    return 0;
  } else if (FLAGS_build_graph) {
    LOG(INFO) << "Building graph with " << FLAGS_shards << " shards; "		//build a simulated graph
              << FLAGS_apr_nodes << " nodes.";
    PRunAll(prs, {
         BuildGraph(current_shard(), FLAGS_shards, FLAGS_apr_nodes, 15);
    });
    return 0;
  }

  PRunAll(prs, {
        prs->swap_accumulator(new Accumulators<AccelPRStruct>::Replace);
      });

  bool restored = m.restore(); //Restore checkpoint, if it exists.  Useful for stopping the process and modifying the graph.
  fprintf(stderr, "%successfully restore%s previous checkpoint.\n",
          (restored ? "S" : "Did not s"), (restored ? "d" : ""));

	PRunAll(pagedb, {
	  apages->resize(FLAGS_apr_nodes);
	  prs->resize(FLAGS_apr_nodes);
      RecordTable<Page>::Iterator *it =
      pagedb->get_typed_iterator(current_shard());
      for(; !it->done(); it->Next()) {
        Page n = it->value();
        struct APageId p = {n.site(), n.id()};
        struct APageInfo info;
        info.adj.clear();
        for(int i=0; i<n.target_site_size(); i++) {
          struct APageId neigh = {n.target_site(i), n.target_id(i)};
          info.adj.push_back(neigh);
        }
        apages->update(p,info); //some/all of these are wasteful when restoring from checkpoint
      }
      });

  struct timeval start_time, end_time;
  gettimeofday(&start_time, NULL);

  //Do initialization! But don't start processing yet, otherwise some of the vertices
  //will exist because of trigger propagation before they get their initialization value.
  PRunAll(pagedb, {

    RecordTable<Page>::Iterator *it =
      pagedb->get_typed_iterator(current_shard());
    int i=0;

    for(; !it->done(); it->Next()) {
      Page n = it->value();
      struct APageId p = { n.site(), n.id() };
      if (!prs->contains(p)) {
        struct AccelPRStruct initval = { 
           n.target_site_size(), 
           (1.0f-(float)FLAGS_apr_d)/((float)FLAGS_apr_nodes),
	       0.0};
        prs->update(p, initval);
        i++;
      }
    }
    fprintf(stderr,"Shard %d: %d new vertices initialized.\n",current_shard(),i);
  });

  PRunAll(prs, {
        prs->swap_accumulator((HybridTrigger<APageId, AccelPRStruct>*)new AccelPRTrigger);
      });

  //Start it all up by poking the thresholding process with a "null" update on the newly-initialized nodes.
  PRunAll(prs, {
    TypedTableIterator<APageId, AccelPRStruct> *it =
      prs->get_typed_iterator(current_shard());
    float initval = (1-(float)FLAGS_apr_d)/((float)FLAGS_apr_nodes);
    int i=0;
    for(; !it->done(); it->Next()) {
      if (it->value().pr_int == initval && it->value().pr_ext == 0.0f) {
        struct AccelPRStruct initval = { 1, 0, 0 };
        prs->update(it->key(),initval);
        i++;
      }
    }
    fprintf(stderr,"Shard %d: Driver kickstarted %d vertices.\n",current_shard(),i);
  });

  gettimeofday(&end_time, NULL);
  long long totaltime = (long long) (end_time.tv_sec - start_time.tv_sec)
      * 1000000 + (end_time.tv_usec - start_time.tv_usec);
  cout << "Total PageRank time: " << ((double) (totaltime) / 1000000.0)
      << " seconds" << endl;

  //Perform checkpoint
  vector<int> cptables;
  cptables.push_back(0);
  cptables.push_back(1);
  RunDescriptor cp_rd("aprcp_kern", "aprcp_kern_cp", prs, cptables);
  m.run_all(cp_rd);

  PRunOne(prs, {
    fprintf(stdout,"PageRank complete, tabulating results...\n");
    float pr_min = 1, pr_max = -1, pr_sum = 0;
    struct APageId toplist[FLAGS_ashow_top];
    float topscores[FLAGS_ashow_top];
    int totalpages = 0;

    //Initialize top scores
    for(int i=0; i<FLAGS_ashow_top; i++)
      topscores[i] = -999999.9999;

    //Find top [FLAGS_ashow_top] pages
    for(int shard=0; shard < prs->num_shards(); shard++) {
      TypedTableIterator<APageId, AccelPRStruct> *it = prs->get_typed_iterator(shard,PREFETCH);

      for(; !it->done(); it->Next()) {
        totalpages++;
        float thisval = it->value().pr_ext;
        if (thisval > pr_max)
          pr_max = it->value().pr_ext;
        //If it's at least better than the worst of the top list, then replace the worst with
        //this one, and propagate it upwards through the list until it's in place
        if (thisval > topscores[FLAGS_ashow_top-1]) {
          topscores[FLAGS_ashow_top-1] = thisval;
          toplist[FLAGS_ashow_top-1] = it->key();
          for(int i=FLAGS_ashow_top-2; i>=0; i--) {
            if (topscores[i] < topscores[i+1]) {
              float a = topscores[i];				//swap!
              struct APageId b = toplist[i];
              topscores[i] = topscores[i+1];
              toplist[i] = toplist[i+1];
              topscores[i+1] = a;
              toplist[i+1] = b;
            } else {
              break;
            }
          }
        }
        if (it->value().pr_ext < pr_min)
          pr_min = thisval;
        pr_sum += thisval;
      }
    }
    float pr_avg = pr_sum/totalpages;
    fprintf(stdout,"RESULTS: min=%e, max=%e, sum=%f, avg=%f [%d pages in %d shards]\n",pr_min,pr_max,pr_sum,pr_avg,totalpages,prs->num_shards());
    fprintf(stdout,"Top Pages:\n");
    for(int i=0;i<FLAGS_ashow_top;i++) {
      fprintf(stdout,"%d\t%f\t%ld-%ld\n",i+1,topscores[i],toplist[i].site,toplist[i].page);
    }
  });

  return 0;
}
REGISTER_RUNNER(AccelPagerank);
