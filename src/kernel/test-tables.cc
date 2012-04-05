#include "util/common.h"
#include "kernel/sparse-table.h"
#include "kernel/dense-table.h"
#include "kernel/global-table.h"

#include "util/static-initializers.h"
#include <gflags/gflags.h>

using std::tr1::unordered_map;
using namespace piccolo;

int optimizer_hack;

DEFINE_int32(test_table_size, 100000, "");
#define START_TEST_PERF { Timer t; for (int i = 0; i < FLAGS_test_table_size; ++i) {

#define END_TEST_PERF(name)\
  }\
  fprintf(stderr, "%s: %d ops in %.3f seconds; %.0f/s %.0f cycles\n",\
          #name, FLAGS_test_table_size, t.elapsed(), t.rate(FLAGS_test_table_size), t.cycle_rate(FLAGS_test_table_size)); }

#define TEST_PERF(name, op)\
    START_TEST_PERF \
    op; \
    END_TEST_PERF(name)

namespace {
struct MapTestRGB {
  uint16_t r;
  uint16_t g;
  uint16_t b;
};

template <class T>
static T* GetTable() {
  TableDescriptor td(0, 1);
  td.accum = new Accumulators<int>::Replace();
  td.key_marshal = new Marshal<int>;
  td.value_marshal = new Marshal<int>;
  td.block_size = 500;
  td.block_info = new IntBlockInfo;

  T *t = new T;
  t->Init(&td);
  t->resize(100);
  return t;
}

template <class T>
static void TestPut() {
  T* t = GetTable<T>();

  for (int i = 0; i < 10000; ++i) {
    t->put(100 * i, 1);
  }

  for (int i = 0; i < 10000; ++i) {
    CHECK(t->contains(100 * i));
    CHECK_EQ(t->get(100 * i), 1);
  }
}

template <class T>
static void TestIterate() {
  T* t = GetTable<T>();

  // Dense tables create entries rounded up to the size of a block, so let's do that here
  // also to avoid spurious errors (for entries that are default initialized)
  for (int i = 0; i < (1 + (10000 / t->info().block_size)) * t->info().block_size; ++i) {
    t->put(i, 1);
  }

  std::tr1::unordered_map<int, int> check;
  typename T::Iterator *i = (typename T::Iterator*)t->get_iterator();
  while (!i->done()) {
    CHECK_EQ(t->contains(i->key()), true);
    CHECK_EQ(i->value(), 1) << i->key() << " : " << i->value();

    check[i->key()] = 1;
    i->Next();
  }

  for (int i = 0; i < 10000; ++i) {
    CHECK(check.find(i) != check.end());
  }
}

template <class T>
static void TestUpdate() {
  T* t = GetTable<T>();

  for (int i = 0; i < 10000; ++i) {
    t->put(100 * i, 1);
  }

  for (int i = 0; i < 10000; ++i) {
    CHECK(t->contains(100 * i));
    t->update(100 * i, 2);
  }

  for (int i = 0; i < 10000; ++i) {
    CHECK(t->contains(100 * i));
    CHECK_EQ(t->get(100 * i), 2);
  }
}


template <class T>
static void TestSerialize() {
  T* t = GetTable<T>();

  static const int kBlockSize = t->info().block_size;
  static const int kTestSize = (1 + (10000 / kBlockSize)) * kBlockSize;
  for (int i = 0; i < kTestSize; ++i) {
    t->put(i, 1);
  }

  CHECK_EQ(t->size(), kTestSize);

  TableData tdata;
  T* t2 = GetTable<T>();

  ProtoTableCoder c(&tdata);
  t->Serialize(&c);

  //The following replaces the old t2->ApplyUpdate() <CRM>
/*
  DecodeIterator* it = t2->partitions_[t2->req.shard()]->DecodeUpdates(&c);
  for(;!it->done(); it->Next()) {
    t2->update(it->key(),it->value());
  }
*/

  LOG(INFO) << "Serialized table to: " << tdata.ByteSize() << " bytes.";

  CHECK_EQ(t->size(), t2->size());

  TableIterator *i1 = t->get_iterator();
  TableIterator *i2 = t2->get_iterator();

  int count = 0;
  string k1, k2, v1, v2;

  while (!i1->done()) {
    CHECK_EQ(i2->done(), false);

    i1->key_str(&k1); i1->value_str(&v1);
    i2->key_str(&k2); i2->value_str(&v2);

    CHECK_EQ(k1, k2);
    CHECK_EQ(v1, v2);

    i1->Next();
    i2->Next();
    ++count;
  }

  CHECK_EQ(i2->done(), true);
  CHECK_EQ(count, t->size());
}

typedef DenseTable<int, int> DTInt;
typedef SparseTable<int, int> STInt;

REGISTER_TEST(DenseTablePut, TestPut<DTInt>());
REGISTER_TEST(SparseTablePut, TestPut<STInt>());

REGISTER_TEST(DenseTableUpdate, TestUpdate<DTInt>());
REGISTER_TEST(SparseTableUpdate, TestUpdate<STInt>());

REGISTER_TEST(DenseTableSerialize, TestSerialize<DTInt>());
REGISTER_TEST(SparseTableSerialize, TestSerialize<STInt>());

REGISTER_TEST(DenseTableIterate, TestIterate<DTInt>());
REGISTER_TEST(SparseTableIterate, TestIterate<STInt>());

static void TestMapPerf() {
  vector<int> source(FLAGS_test_table_size);
  for (int i = 0; i < source.size(); ++i) {
    source[i] = random() % FLAGS_test_table_size;
  }

  SparseTable<int, int> *h = GetTable<SparseTable<int, int> >();
  TEST_PERF(SparseTablePut, h->put(source[i], i));
  TEST_PERF(SparseTableGet, h->get(source[i]));

  DenseTable<int, int> *d = GetTable<DenseTable<int, int> >();
  TEST_PERF(DenseTablePut, d->put(source[i], i));
  TEST_PERF(DenseTableGet, d->get(source[i]));

  vector<int> array_test(FLAGS_test_table_size * 2);
  std::tr1::hash<int> hasher;

  // Need to have some kind of side effect or this gets eliminated entirely.
  optimizer_hack = 0;
  TEST_PERF(ArrayPut, optimizer_hack += array_test[hasher(i) % FLAGS_test_table_size]);
}
REGISTER_TEST(SparseTablePerf, TestMapPerf());

}
