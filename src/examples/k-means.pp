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
      PRunOne(clusters, {
        for (int i = 0; i < FLAGS_num_clusters; ++i) {
          actual->update(i, random_cluster());
          clusters->update(i, random_cluster());
        }
      });
    }

  // Initialize points: points are created in a small cloud around each cluster center.
    PRunAll(points, {
          points->resize(FLAGS_num_points);

          const int num_shards = points->num_shards();
          for (int64_t i = current_shard(); i < FLAGS_num_points; i += num_shards) {
            Cluster c = actual->get(i % FLAGS_num_clusters);
            Point p = {c.x + 0.1 * (rand_double() - 0.5), c.y + 0.1 * (rand_double() - 0.5), -1, 0};
            points->update(i, p);
          }
        });

    for (int i = 0; i < FLAGS_iterations; i++) {
      // Compute the closest cluster to each point.
      PMap( {p : points}, {
            p.min_dist = 2;
            for (int i = 0; i < FLAGS_num_clusters; ++i) {
              const Cluster& c = clusters->get(i);
              double d_squared = pow(p.x - c.x, 2) + pow(p.y - c.y, 2);
              if (d_squared < p.min_dist) {
                p.min_dist = d_squared;
                p.source = i;
              }
            }
          });

      // Reset cluster positions.  If a cluster has no points, assign it the
      // position of a random point instead.
      PMap( {c : clusters}, {
            if (c.x == 0 && c.y == 0) {
              Point p = points->get(random() % FLAGS_num_points);
              c.x = p.x; c.y = p.y;
            } else {
              c.x = 0; c.y = 0;
            }
          });

      // Average the contribution of each point to it's assigned cluster.
      PMap( {p : points}, {
            Cluster c = {p.x * FLAGS_num_clusters / FLAGS_num_points,
              p.y * FLAGS_num_clusters / FLAGS_num_points};

            clusters->update(p.source, c);
          });

      m.checkpoint();
    }
  }
  return 0;
}
REGISTER_RUNNER(KMeans);
