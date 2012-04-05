#include "client/client.h"

using namespace piccolo;

DEFINE_int32(table_size, 100000, "");

static TypedGlobalTable<int, int>* min_hash = NULL;
static TypedGlobalTable<int, int>* max_hash = NULL;
static TypedGlobalTable<int, int>* sum_hash = NULL;
static TypedGlobalTable<int, int>* replace_hash = NULL;
static TypedGlobalTable<int, string>* string_hash = NULL;

//static TypedGlobalTable<int, Pair>* pair_hash = NULL;

class TableKernel : public DSMKernel {
public:
  void TestPut() {
    for (int i = 0; i < FLAGS_table_size; ++i) {
      LOG_EVERY_N(INFO, 100000) << "Writing... " << LOG_OCCURRENCES;
      min_hash->update(i, i);
      max_hash->update(i, i);
      sum_hash->update(i, 1);
      replace_hash->update(i, i);
      string_hash->update(i, StringPrintf("%d", i));
//      p.set_key(StringPrintf("%d", i));
//      p.set_value(StringPrintf("%d", i));
//      pair_hash->update(i, p);
    }
  }

  void TestGet() {
    int num_shards = min_hash->num_shards();

    for (int i = 0; i < FLAGS_table_size; ++i) {
      LOG_EVERY_N(INFO, 100) << "Fetching... " << i;
      CHECK_EQ(min_hash->get(i), i) << " i= " << i;
      CHECK_EQ(max_hash->get(i), i) << " i= " << i;
      CHECK_EQ(replace_hash->get(i), i) << " i= " << i;
      CHECK_EQ(sum_hash->get(i), num_shards) << " i= " << i;
      CHECK_EQ(string_hash->get(i), StringPrintf("%d", i)) << " i= " << i;
//      CHECK_EQ(pair_hash->get(i).value(), StringPrintf("%d", i));
    }
  }

  void TestGetLocal() {
    TypedTableIterator<int, int> *it = min_hash->get_typed_iterator(current_shard());
    int num_shards = min_hash->num_shards();

    while (!it->done()) {
      const int& k = it->key();
      CHECK_EQ(min_hash->get_shard(k), current_shard());
      CHECK_EQ(min_hash->get(k), k) << " k= " << k;
      CHECK_EQ(max_hash->get(k), k) << " k= " << k;
      CHECK_EQ(replace_hash->get(k), k) << " k= " << k;
      CHECK_EQ(sum_hash->get(k), num_shards) << " k= " << k;
      CHECK_EQ(string_hash->get(k), StringPrintf("%d", k)) << " i= " << k;
//      CHECK_EQ(pair_hash->get(k).value(), StringPrintf("%d", k));
      it->Next();
    }
  }

  void TestIterator() {
    int n = min_hash->num_shards();
    for (int k = 0; k < n; k++) {
      TypedTableIterator<int, int> *it = min_hash->get_typed_iterator(k);
      int i = 0;
      while (i < 100 && !it->done()) {
        assert(it->key()==it->value());
        i++;
        it->Next();
      }
    }
  }
};

REGISTER_KERNEL(TableKernel);
REGISTER_METHOD(TableKernel, TestPut);
REGISTER_METHOD(TableKernel, TestGet);
REGISTER_METHOD(TableKernel, TestGetLocal);
REGISTER_METHOD(TableKernel, TestIterator);

static int TestTables(const ConfigData &conf) {
  min_hash = CreateTable(0, FLAGS_shards, new Sharding::Mod, new Accumulators<int>::Min);
  max_hash = CreateTable(1, FLAGS_shards, new Sharding::Mod, new Accumulators<int>::Max);
  sum_hash = CreateTable(2, FLAGS_shards, new Sharding::Mod, new Accumulators<int>::Sum);
  replace_hash = CreateTable(3, FLAGS_shards, new Sharding::Mod, new Accumulators<int>::Replace);
  string_hash = CreateTable(4, FLAGS_shards, new Sharding::Mod, new Accumulators<string>::Replace);

  if (!StartWorker(conf)) {
    Master m(conf);
    m.run_all("TableKernel", "TestPut",  min_hash);
    //m.checkpoint();

    // wipe all the tables and then restore from the previous checkpoint.
    // m.run_all(Master::RunDescriptor::C("TableKernel", "TestClear", 0, 0));

    //m.restore();
    m.run_all("TableKernel", "TestGetLocal",  min_hash);

    //m.checkpoint();
    m.run_all("TableKernel", "TestGet",  min_hash);

		m.run_one("TableKernel", "TestIterator",  min_hash);
  }
  return 0;
}
REGISTER_RUNNER(TestTables);
