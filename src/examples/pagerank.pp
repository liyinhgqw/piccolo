#include "examples/examples.h"
#include "external/webgraph/webgraph.h"

#include <algorithm>
#include <libgen.h>

using namespace piccolo;
using namespace std;

#define PREFETCH 512
static float TOTALRANK = 0;
static int NUM_WORKERS = 2;

static const float kPropagationFactor = 0.8;
static const int kBlocksize = 1000;
static const char kTestPrefix[] = "/scratch/kerm/pr-graph.rec";

DEFINE_bool(memory_graph, false,
    "If true, the web graph will be generated on-demand.");

DEFINE_string(graph_prefix, kTestPrefix, "Path to web graph.");
DEFINE_int32(nodes, 10000, "");
DEFINE_int32(show_top, 10, "number of top results to display");
DEFINE_double(tol,0.0, "convergence tolerance (0 to use iteration count)");

DEFINE_string(convert_graph, "", "Path to WebGraph .graph.gz database to convert");

static float powerlaw_random(float dmin, float dmax, float n) {
  float r = (float) random() / RAND_MAX;
  return pow((pow(dmax, n) - pow(dmin, n)) * pow(r, 3) + pow(dmin, n), 1.0 / n);
}

static float random_restart_seed() {
  return (1 - kPropagationFactor) * (TOTALRANK / FLAGS_nodes);
}

// I'd like to use a pair here, but for some reason they fail to count
// as POD types according to C++.  Sigh.
struct PageId {
  int64_t site :32;
  int64_t page :32;
};

bool operator==(const PageId& a, const PageId& b) {
  return a.site == b.site && a.page == b.page;
}

std::ostream& operator<<(std::ostream& s, const PageId& a) {
  return s << a.site << ":" << a.page;
}

namespace std {
namespace tr1 {
template<>
struct hash<PageId> {
  size_t operator()(const PageId& p) const {
    return SuperFastHash((const char*) &p, sizeof p);
  }
};
}
}

struct SiteSharding: public Sharder<PageId> {
  int operator()(const PageId& p, int nshards) {
    return p.site % nshards;
  }
};

struct PageIdBlockInfo: public BlockInfo<PageId> {
  PageId start(const PageId& k, int block_size) {
    PageId p = { k.site, k.page - (k.page % block_size) };
    return p;
  }

  int offset(const PageId& k, int block_size) {
    return k.page % block_size;
  }

};

static vector<int> InitSites() {
  static vector<int> site_sizes;
  srand(0);
  for (int n = 0; n < FLAGS_nodes;) {
    int c = powerlaw_random(1,
                            min(50000, (int) (100000. * FLAGS_nodes / 100e6)),
                            0.001);
    c = (c==0)?1:c;
    site_sizes.push_back(c);
    n += c;
  }
  return site_sizes;
}

static vector<int> site_sizes;

static void BuildGraph(int shard, int nshards, int nodes, int density) {
  char* d = strdup(FLAGS_graph_prefix.c_str());
  File::Mkdirs(dirname(d));

  string target = StringPrintf("%s-%05d-of-%05d-N%05d",
                               FLAGS_graph_prefix.c_str(), shard, nshards,
                               nodes);

  if (File::Exists(target)) {
    return;
  }

  srand(shard);
  Page n;
  RecordFile out(target, "w", RecordFile::NONE);
  // Only sites with site_id % nshards == shard are in this shard.
  for (int i = shard; i < site_sizes.size(); i += nshards) {
    PERIODIC(
        1,
        LOG(INFO) << "Working: Shard -- " << shard << " of " << nshards
            << "; site " << i << " of " << site_sizes.size());
    for (int j = 0; j < site_sizes[i]; ++j) {
      n.Clear();
      n.set_site(i);
      n.set_id(j);
      for (int k = 0; k < density; k++) {
        int target_site =
            (random() % 10 != 0) ? i : (random() % site_sizes.size());
        n.add_target_site(target_site);
        n.add_target_id(random() % site_sizes[target_site]);
      }

      out.write(n);
    }
  }
}

static void WebGraphPageIds(WebGraph::Reader *wgr, vector<PageId> *out) {
  WebGraph::URLReader *r = wgr->newURLReader();

  map<string, struct PageId> hosts;
  map<string, struct PageId>::iterator it;
  struct PageId pid = { -1, -1 };
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
      struct PageId newpid = { maxsite++, 1 };
      hosts.insert(pair<string, struct PageId>(host, newpid));
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
  vector<PageId> pageIds;
  WebGraphPageIds(&r, &pageIds);

  char* d = strdup(FLAGS_graph_prefix.c_str());
  File::Mkdirs(dirname(d));

  RecordFile *out[nshards];
  for (int i = 0; i < nshards; ++i) {
    string target = StringPrintf("%s-%05d-of-%05d-N%05d",
                                 FLAGS_graph_prefix.c_str(), i, nshards,
                                 r.nodes);
    out[i] = new RecordFile(target, "w", RecordFile::NONE);
  }

  // XXX Maybe we should take at most FLAGS_nodes nodes
  const WebGraph::Node *node;
  Page n;
  LOG(INFO) << "Beginning ConvertGraph..." << endl;
  int i = 0;
  while ((node = r.readNode())) {
    if (i++ % 100000 == 0)
      LOG(INFO) << "Reading node " << 1 + node->node << " of " << r.nodes;
    PageId src = pageIds.at(node->node);
    n.Clear();
    n.set_site(src.site);
    n.set_id(src.page);
    for (unsigned int i = 0; i < node->links.size(); ++i) {
      PageId dest = pageIds.at(node->links[i]);
//      LOG(INFO) << "Translating neighbor node "<<i<<" of "<<node->links.size()<<" @offset="<< node->links[i] <<": site=" << dest.site << ", page=" << dest.page << endl;
      n.add_target_site(dest.site);
      n.add_target_id(dest.page);
    }
    out[src.site % nshards]->write(n);
  }

  for (int j = 0; j < nshards; ++j)
    delete out[j];
  LOG(INFO) << "ConvertGraph: " << i << " nodes read";
}

// Generate a graph on-demand rather then reading from disk.
class InMemoryTable: public DiskTable<uint64_t, Page> {
public:
  struct Iterator: public TypedTableIterator<uint64_t, Page> {
    Iterator(int shard, int num_shards) :
        shard_(shard), site_(shard), site_pos_(0), num_shards_(num_shards) {
      srand(shard);
    }

    Page p_;
    uint64_t k_;

    const uint64_t& key() {
      k_ = 0;
      return k_;
    }
    Page& value() {
      return p_;
    }

    void Next() {
      if (site_pos_ >= site_sizes[site_]) {
        site_ += num_shards_;
        site_pos_ = 0;
      }

      p_.Clear();
      p_.set_site(site_);
      p_.set_id(site_pos_);

//      k_ = P(site_, site_pos_);

      for (int k = 0; k < 15; k++) {
        int target_site =
            (random() % 10 != 0) ? site_ : (random() % site_sizes.size());
        p_.add_target_site(target_site);
        p_.add_target_id(random() % site_sizes[target_site]);
      }

      ++site_pos_;
    }

    bool done() {
      return (site_ >= site_sizes.size());
    }
  private:
    int shard_;
    int site_;
    int site_pos_;
    int num_shards_;
  };

  InMemoryTable(int num_shards) :
      DiskTable<uint64_t, Page>("", 0), num_shards_(num_shards) {
  }
  Iterator *get_iterator(int shard, unsigned int fetch_num) {
    return new Iterator(shard, num_shards_);
  }

private:
  int num_shards_;
};

//Tables in use
TypedGlobalTable<PageId, float>* curr_pr;
TypedGlobalTable<PageId, float>* next_pr;
DiskTable<uint64_t, Page> *pages;
TypedGlobalTable<int, float>* maxtol;
static TypedGlobalTable<string, string>* StatsTable = NULL;

// This kernel contains awkward functions that need to directly play with the RunDescriptor,
// and therefore can't be run via the abstracted P RunOne/P RunAll/P Map interfaces.
class prcp_kern: public DSMKernel {
public:
  // Fake little method to cause a checkpoint
  void prcp_kern_cp() {
    volatile int i = 0;
    i++;
  }

  // Perform prescaling of a loaded checkpoint's PR values
  void prcp_kern_prescale() {
    const string& psfs("psf");
    int psf = get_arg<int>(psfs);
    TypedTableIterator<PageId, float>* it = curr_pr->get_typed_iterator(
        current_shard());
    for (; !it->done(); it->Next()) {
      //negative zeroes it, fraction scales it.
      float newpr = -it->value() + (it->value() / psf);
      curr_pr->update(it->key(), newpr);
    }
    it = next_pr->get_typed_iterator(current_shard());
    for (; !it->done(); it->Next()) {
      float newpr = -it->value() + (it->value() / psf);
      next_pr->update(it->key(), newpr);
    }
  }
};

REGISTER_KERNEL(prcp_kern);
REGISTER_METHOD(prcp_kern,prcp_kern_cp);
REGISTER_METHOD(prcp_kern,prcp_kern_prescale);

//Main PageRank driver
int Pagerank(const ConfigData& conf) {
  site_sizes = InitSites();

  NUM_WORKERS = conf.num_workers();
  TOTALRANK = FLAGS_nodes;

  TableDescriptor* pr_desc = new TableDescriptor(0, FLAGS_shards);

  pr_desc->partition_factory = new SparseTable<PageId, float>::Factory;
  pr_desc->block_size = 1000;
  pr_desc->block_info = new PageIdBlockInfo;
  pr_desc->sharder = new SiteSharding;
  pr_desc->accum = new Accumulators<float>::Sum;

  curr_pr = CreateTable<PageId, float>(pr_desc);
  pr_desc->table_id = 1;
  next_pr = CreateTable<PageId, float>(pr_desc);

  if (FLAGS_build_graph) {
    if (rpc::NetworkThread::Get()->id() == 0) {
      LOG(INFO) << "Building graph with " << FLAGS_shards << " shards; "
          << FLAGS_nodes << " nodes.";
      for (int i = 0; i < FLAGS_shards; ++i) {
        BuildGraph(i, FLAGS_shards, FLAGS_nodes, 15);
      }
    }
    return 0;
  }

  if (FLAGS_memory_graph) {
    pages = new InMemoryTable(FLAGS_shards);
    TableRegistry::Get()->tables().insert(make_pair(2, pages));
  } else if (FLAGS_convert_graph.empty()) {
    pages = CreateRecordTable<Page>(2, FLAGS_graph_prefix + "*", false, FLAGS_shards);
  } //else we're doing a conversion

  maxtol = CreateTable(3, FLAGS_shards, new Sharding::Mod,
                       new Accumulators<float>::Replace);
  StatsTable = CreateTable(10000, 1, new Sharding::String,
                           new Accumulators<string>::Replace);
  maxtol->resize(FLAGS_shards * 2); //first FLAGS_shards nodes for max tol, second FLAGS_shards nodes for PR sums
  StatsTable->resize(1); //pass quiescence state

  StartWorker(conf);
  Master m(conf);

  if (!FLAGS_convert_graph.empty()) {
    ConvertGraph(FLAGS_convert_graph, FLAGS_shards);
    return 0;
  }

  PRunAll(curr_pr, {
        next_pr->resize((int)(2 * FLAGS_nodes));
        curr_pr->resize((int)(2 * FLAGS_nodes));
      });

  bool restored = m.restore(); //Restore checkpoint, if it exists and restore flag is true.  Useful for stopping the process and modifying the graph.
  fprintf(stderr, "%s restore%s previous checkpoint.\n",
          (restored ? "Successfully" : "Did not successfully"),
          (restored ? "d" : ""));

  int &i = m.get_cp_var<int>("iteration", 0); //actual iteration count
  int i2 = 0; //number of iterations since this particular run started

  //If necessary, do some pre-scaling
  if (i != 0) {
    RunDescriptor prescale_rd("prcp_kern", "prcp_kern_prescale", curr_pr);
    prescale_rd.params.put("psf", i);
    m.run_all(prescale_rd);
  }

  bool done = (i >= FLAGS_iterations) && FLAGS_tol == 0.0;

  struct timeval start_time, end_time;
  gettimeofday(&start_time, NULL);

  while (!done) {
    PRunAll(pages, {
          DiskTable<uint64_t, Page>::Iterator *it = pages->get_typed_iterator(current_shard());
          int pcount=0;
          for (; !it->done(); it->Next()) {
            Page& n = it->value();
            struct PageId p = {n.site(), n.id()};
            next_pr->update(p, random_restart_seed());

            float v = 0;
            if (curr_pr->contains(p)) {
              v = curr_pr->get_local(p);
            }

            float contribution = kPropagationFactor * v / n.target_site_size();
            for (int i = 0; i < n.target_site_size(); ++i) {
              PageId target = {n.target_site(i), n.target_id(i)};
              if (!(target == p)) {
                next_pr->update(target, contribution);
              }
            }
            pcount++;
          }
          delete it;
        });

    // Find per-shard max delta
    PRunAll(curr_pr, {
          TypedTableIterator<PageId,float>* it = curr_pr->get_typed_iterator(current_shard());
          float diff = -999999.9;
          double sum = 0;
          for(; !it->done(); it->Next()) {
            float thispr = it->value();
            CHECK(next_pr->contains(it->key())) << "Missing key from next_pr! site " << it->key().site << ", page " << it->key().page;
            diff = max(diff,(next_pr->get(it->key())-thispr)-random_restart_seed());
            sum += thispr;
          }
          maxtol->update(current_shard(),diff); //first num_shards items are per-shard max deltas
          maxtol->update(current_shard()+curr_pr->num_shards(),sum);//next num_shard items are per-shard PR sums
        });

    // Find overall max delta, establish quiescence state
    PRunOne(maxtol, {
          float maxdiff = -99999.9;
          double sum = 0;
          for(int shard = 0; shard < curr_pr->num_shards(); shard++) {
            maxdiff = max(maxdiff,maxtol->get(shard));
            sum += maxtol->get(shard+curr_pr->num_shards());
          }
          fprintf(stderr, "Maximum PR delta=%e, sum=%f, scaled max=%e\n",maxdiff,sum,maxdiff/sum); //scaling necessary because of PR method used
          StatsTable->update("q",((maxdiff/sum)<FLAGS_tol)?"q":"u");//_q_uiescent or _u_nstable based on tolerance vs. max delta PR
        });

    PageId pzero = { 0, 0 };
    fprintf(stderr, "Iteration %d; PR %.3f\n", i,
            curr_pr->contains(pzero) ? curr_pr->get(pzero) : 0);

    // Move the values computed from the last iteration into the current table.
    curr_pr->swap(next_pr);
    next_pr->clear();

    i++;
    i2++;
    if (FLAGS_tol == 0.0) { //0.0 == use strict iteration count
      done = (i >= FLAGS_iterations);
    } else {
      done = ((i2 > 1) && (0 == strcmp("q", StatsTable->get("q").c_str()))); //otherwise check if tolerance is reached
      //why the i2 check? Because other restarting with a modified/augmented
      //graph fails, since it thinks it converged on the first iter
    }
  }

  gettimeofday(&end_time, NULL);
  long long totaltime = (long long) (end_time.tv_sec - start_time.tv_sec)
      * 1000000 + (end_time.tv_usec - start_time.tv_usec);
  cout << "Total PageRank time: " << ((double) (totaltime) / 1000000.0)
      << " seconds" << endl;

  //Perform checkpoint
  vector<int> cptables;
  cptables.push_back(0);
  cptables.push_back(1);
  cptables.push_back(3);
  RunDescriptor cp_rd("prcp_kern", "prcp_kern_cp", curr_pr, cptables);
  m.run_all(cp_rd);

  //Final evaluation
  PRunOne(curr_pr, {
        fprintf(stdout,"PageRank complete, tabulating results...\n");
        float pr_min = 1, pr_max = 0, pr_sum = 0;
        struct PageId toplist[FLAGS_show_top]; //Array of top-scoring page IDs
        float topscores[FLAGS_show_top];//Array of top-scoring page ranks
        int totalpages = 0;
        for(int i=0; i<FLAGS_show_top; i++) {
          topscores[i] = -99999.9999;
        }

        for(int shard=0; shard < curr_pr->num_shards(); shard++) {
          TypedTableIterator<PageId, float> *it = curr_pr->get_typed_iterator(shard,PREFETCH);

          for(; !it->done(); it->Next()) {
            totalpages++;
            if (it->value() > pr_max)
            pr_max = it->value();
            if (it->value() > topscores[FLAGS_show_top-1]) {
              topscores[FLAGS_show_top-1] = it->value();
              toplist[FLAGS_show_top-1] = it->key();
              for(int i=FLAGS_show_top-2; i>=0; i--) {
                if (topscores[i] < topscores[i+1]) {
                  float a = topscores[i];
                  struct PageId b = toplist[i];
                  topscores[i] = topscores[i+1];
                  toplist[i] = toplist[i+1];
                  topscores[i+1] = a;
                  toplist[i+1] = b;
                } else {
                  break;
                }
              }
            }
            if (it->value() < pr_min)
            pr_min = it->value();
            pr_sum += it->value();
          }
        }
        if (0 >= totalpages) {LOG(FATAL) << "No pages found in output table!" << endl;}
        float pr_avg = pr_sum/totalpages;
        fprintf(stdout,"RESULTS: min=%f, max=%f, sum=%f, avg=%f [%d pages in %d shards]\n",pr_min,pr_max,pr_sum,pr_avg,totalpages,curr_pr->num_shards());
        fprintf(stdout,"Top Pages:\n");
        for(int i=0;i<FLAGS_show_top;i++) {
          fprintf(stdout,"%d\t%e\t%ld-%ld\n",i+1,topscores[i],toplist[i].site,toplist[i].page);
        }
      });

  return 0;
}
REGISTER_RUNNER(Pagerank);
