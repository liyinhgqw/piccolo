#include "client/client.h"

using namespace piccolo;

DEFINE_int32(table_size2, 100000, "");

static TypedGlobalTable<int, int>* min_hash = NULL;
static TypedGlobalTable<int, int>* max_hash = NULL;
static TypedGlobalTable<int, int>* sum_hash = NULL;
static TypedGlobalTable<int, int>* replace_hash = NULL;
static TypedGlobalTable<int, string>* string_hash = NULL;

//static TypedGlobalTable<int, Pair>* pair_hash = NULL;

#define PREFETCH 512

class TableKernel2: public DSMKernel {
public:
  virtual ~TableKernel2() {
  }
  void TestPut2() {
    for (int i = 0; i < FLAGS_table_size2; ++i) {
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

  void TestGet2() {
    int num_shards = min_hash->num_shards();

    for (int i = 0; i < FLAGS_table_size2; ++i) {
      LOG_EVERY_N(INFO, 100) << "Get Fetching... " << i;
      CHECK_EQ(min_hash->get(i), i) << " i= " << i;
      CHECK_EQ(max_hash->get(i), i) << " i= " << i;
      CHECK_EQ(replace_hash->get(i), i) << " i= " << i;
      CHECK_EQ(sum_hash->get(i), num_shards) << " i= " << i;
      CHECK_EQ(string_hash->get(i), StringPrintf("%d", i)) << " i= " << i;
//      CHECK_EQ(pair_hash->get(i).value(), StringPrintf("%d", i));
    }
  }

  void TestGetIterator2() {
    int k = current_shard();
    int totalrows = 0;
    int num_shards = min_hash->num_shards();
    for (int j = 0; j < num_shards; j++) {
      TypedTableIterator<int, int> *it_min = min_hash->get_typed_iterator(j,
          PREFETCH);
      TypedTableIterator<int, int> *it_max = max_hash->get_typed_iterator(j,
          PREFETCH);
      TypedTableIterator<int, int> *it_replace =
          replace_hash->get_typed_iterator(j, PREFETCH);
      TypedTableIterator<int, int> *it_sum = sum_hash->get_typed_iterator(j,
          PREFETCH);
      TypedTableIterator<int, string> *it_string =
          string_hash->get_typed_iterator(j, PREFETCH);
      int i = 0;
      for (;
          !it_min->done() && !it_max->done() && !it_replace->done()
              && !it_sum->done() && !it_string->done();) {
        LOG_EVERY_N(INFO, 10000) << "Iter Fetching Shard " << j << " (local "
              << k << ")... " << i;
        CHECK_EQ(it_min->value(), it_min->key()) << " i= " << i;
        it_min->Next();
        CHECK_EQ(it_max->value(), it_max->key()) << " i= " << i;
        it_max->Next();
        CHECK_EQ(it_replace->value(), it_replace->key()) << " i= " << i;
        it_replace->Next();
        CHECK_EQ(it_sum->value(), num_shards) << " i= " << i;
        it_sum->Next();
        CHECK_EQ(it_string->value(), StringPrintf("%d", it_string->key()))
            << " i= " << i;
        it_string->Next();

        i++;
        totalrows++;
      }
    }
    printf("Total of %d rows read\n", totalrows);
  }

  void TestGetLocal2() {
    TypedTableIterator<int, int> *it = min_hash->get_typed_iterator(
        current_shard());
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

  void TestIterator2() {
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

REGISTER_KERNEL(TableKernel2);
REGISTER_METHOD(TableKernel2, TestPut2);
REGISTER_METHOD(TableKernel2, TestGet2);
REGISTER_METHOD(TableKernel2, TestGetIterator2);
REGISTER_METHOD(TableKernel2, TestGetLocal2);
REGISTER_METHOD(TableKernel2, TestIterator2);

static int TestTables2(const ConfigData &conf) {
  min_hash = CreateTable(0, FLAGS_shards, new Sharding::Mod,
      new Accumulators<int>::Min);
  max_hash = CreateTable(1, FLAGS_shards, new Sharding::Mod,
      new Accumulators<int>::Max);
  sum_hash = CreateTable(2, FLAGS_shards, new Sharding::Mod,
      new Accumulators<int>::Sum);
  replace_hash = CreateTable(3, FLAGS_shards, new Sharding::Mod,
      new Accumulators<int>::Replace);
  string_hash = CreateTable(4, FLAGS_shards, new Sharding::Mod,
      new Accumulators<string>::Replace);

  if (!StartWorker(conf)) {
    Master m(conf);
    m.run_all("TableKernel2", "TestPut2", min_hash);
    //m.checkpoint();

    // wipe all the tables and then restore from the previous checkpoint.
    // m.run_all(Master::RunDescriptor::C("TableKernel", "TestClear", 0, 0));

    //m.restore();
    m.run_all("TableKernel2", "TestGetLocal2", min_hash);

    //m.checkpoint();
    m.run_all("TableKernel2", "TestGetIterator2", min_hash);

//		m.run_one("TableKernel", "TestIterator",  min_hash);
  }
  return 0;
}
REGISTER_RUNNER(TestTables2);
