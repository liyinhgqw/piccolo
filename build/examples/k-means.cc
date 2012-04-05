
#include "client/client.h"

using namespace piccolo;

#line 1 "../src/examples/k-means.pp"
#include "client/client.h"

DEFINE_int64(num_clusters, 2, "");
DEFINE_int64(num_points, 100, "");
DEFINE_bool(dump_results, false, "");

struct Point {
  float x, y;
  float min_dist;
  int source;
};

struct Cluster {
  float x, y;
};

static TypedGlobalTable<int32_t, Point> *points;
static TypedGlobalTable<int32_t, Cluster> *clusters;
static TypedGlobalTable<int32_t, Cluster> *actual;

Cluster random_cluster() {
  Cluster c = { 0.5 - rand_double(), 0.5 - rand_double() };
  return c;
}

struct ClusterAccum: public Accumulator<Cluster> {
  void Accumulate(Cluster* c1, const Cluster& c2) {
    Cluster o = { c1->x + c2.x, c1->y + c2.y };
    *c1 = o;
  }
};

static int KMeans(const ConfigData& conf) {
  const int num_shards = conf.num_workers() * 4;
  clusters = CreateTable(0, num_shards, new Sharding::Mod, new ClusterAccum);
  points = CreateTable(1, num_shards, new Sharding::Mod,
                       new Accumulators<Point>::Replace);
  actual = CreateTable(2, num_shards, new Sharding::Mod,
                       new Accumulators<Cluster>::Replace);

  clusters->mutable_info()->max_stale_time = 10.;

  if (!StartWorker(conf)) {
    Master m(conf);
    if (!m.restore()) {
      // Initialize the cluster centers that we are attempting to find, and
      // our initial guesses.
      m.run_one("___src_examples_k_means_ppRunKernel1", "run", clusters);
    }

  // Initialize points: points are created in a small cloud around each cluster center.
    m.run_all("___src_examples_k_means_ppRunKernel2", "run", points);

    for (int i = 0; i < FLAGS_iterations; i++) {
      // Compute the closest cluster to each point.
      m.run_all("P_P_P_srcP_examplesP_kP_meansP_ppMapKernel3", "map", points);

      // Reset cluster positions.  If a cluster has no points, assign it the
      // position of a random point instead.
      m.run_all("P_P_P_srcP_examplesP_kP_meansP_ppMapKernel4", "map", clusters);

      // Average the contribution of each point to it's assigned cluster.
      m.run_all("P_P_P_srcP_examplesP_kP_meansP_ppMapKernel5", "map", points);


      m.checkpoint();
    }
  }
  return 0;
}
REGISTER_RUNNER(KMeans);

class ___src_examples_k_means_ppRunKernel1 : public DSMKernel {
public:
  virtual ~___src_examples_k_means_ppRunKernel1 () {}
  void run() {
#line 48 "../src/examples/k-means.pp"
      
        for (int i = 0; i < FLAGS_num_clusters; ++i) {
          actual->update(i, random_cluster());
          clusters->update(i, random_cluster());
        };
  }
};

REGISTER_KERNEL(___src_examples_k_means_ppRunKernel1);
REGISTER_METHOD(___src_examples_k_means_ppRunKernel1, run);


class ___src_examples_k_means_ppRunKernel2 : public DSMKernel {
public:
  virtual ~___src_examples_k_means_ppRunKernel2 () {}
  void run() {
#line 57 "../src/examples/k-means.pp"
      
          points->resize(FLAGS_num_points);

          const int num_shards = points->num_shards();
          for (int64_t i = current_shard(); i < FLAGS_num_points; i += num_shards) {
            Cluster c = actual->get(i % FLAGS_num_clusters);
            Point p = {c.x + 0.1 * (rand_double() - 0.5), c.y + 0.1 * (rand_double() - 0.5), -1, 0};
            points->update(i, p);
          };
  }
};

REGISTER_KERNEL(___src_examples_k_means_ppRunKernel2);
REGISTER_METHOD(___src_examples_k_means_ppRunKernel2, run);


class P_P_P_srcP_examplesP_kP_meansP_ppMapKernel3 : public DSMKernel {
public:
  virtual ~P_P_P_srcP_examplesP_kP_meansP_ppMapKernel3() {}
  template <class K, class Value0>
  void run_iter(const K& k, Value0 &p) {
#line 70 "../src/examples/k-means.pp"
    
            p.min_dist = 2;
            for (int i = 0; i < FLAGS_num_clusters; ++i) {
              const Cluster& c = clusters->get(i);
              double d_squared = pow(p.x - c.x, 2) + pow(p.y - c.y, 2);
              if (d_squared < p.min_dist) {
                p.min_dist = d_squared;
                p.source = i;
              }};
  }
  
  template <class TableA>
  void run_loop(TableA* a) {
    typename TableA::Iterator *it =  a->get_typed_iterator(current_shard());
    for (; !it->done(); it->Next()) {
      run_iter(it->key(), it->value());
    }
    delete it;
  }
  
  void map() {
      run_loop(points);
  }
};

REGISTER_KERNEL(P_P_P_srcP_examplesP_kP_meansP_ppMapKernel3);
REGISTER_METHOD(P_P_P_srcP_examplesP_kP_meansP_ppMapKernel3, map);


class P_P_P_srcP_examplesP_kP_meansP_ppMapKernel4 : public DSMKernel {
public:
  virtual ~P_P_P_srcP_examplesP_kP_meansP_ppMapKernel4() {}
  template <class K, class Value0>
  void run_iter(const K& k, Value0 &c) {
#line 84 "../src/examples/k-means.pp"
    
            if (c.x == 0 && c.y == 0) {
              Point p = points->get(random() % FLAGS_num_points);
              c.x = p.x; c.y = p.y;
            }else {
              c.x = 0; c.y = 0;
            };
  }
  
  template <class TableA>
  void run_loop(TableA* a) {
    typename TableA::Iterator *it =  a->get_typed_iterator(current_shard());
    for (; !it->done(); it->Next()) {
      run_iter(it->key(), it->value());
    }
    delete it;
  }
  
  void map() {
      run_loop(clusters);
  }
};

REGISTER_KERNEL(P_P_P_srcP_examplesP_kP_meansP_ppMapKernel4);
REGISTER_METHOD(P_P_P_srcP_examplesP_kP_meansP_ppMapKernel4, map);


class P_P_P_srcP_examplesP_kP_meansP_ppMapKernel5 : public DSMKernel {
public:
  virtual ~P_P_P_srcP_examplesP_kP_meansP_ppMapKernel5() {}
  template <class K, class Value0>
  void run_iter(const K& k, Value0 &p) {
#line 94 "../src/examples/k-means.pp"
    
            Cluster c = {p.x * FLAGS_num_clusters / FLAGS_num_points,
              p.y * FLAGS_num_clusters / FLAGS_num_points};

            clusters->update(p.source, c);
          ;
  }
  
  template <class TableA>
  void run_loop(TableA* a) {
    typename TableA::Iterator *it =  a->get_typed_iterator(current_shard());
    for (; !it->done(); it->Next()) {
      run_iter(it->key(), it->value());
    }
    delete it;
  }
  
  void map() {
      run_loop(points);
  }
};

REGISTER_KERNEL(P_P_P_srcP_examplesP_kP_meansP_ppMapKernel5);
REGISTER_METHOD(P_P_P_srcP_examplesP_kP_meansP_ppMapKernel5, map);


