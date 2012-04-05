#include "client/client.h"
#include "examples/examples.h"
#include <algorithm>

using namespace piccolo;
typedef uint32_t KeyType;
typedef Bucket ValueType;

struct BucketMerge : public Accumulator<Bucket> {
  void Accumulate(Bucket *l, const Bucket &r) {
    l->MergeFrom(r);
  }
};

struct KeyGen {
  KeyGen() : x_(314159625), a_(1220703125) {}
  KeyType next() {
    uint64_t n = a_ * x_ % (2ll << 46);
    x_ = n;
    return x_;
  }

  uint64_t x_;
  uint64_t a_;
};

static std::vector<int> src;
static TypedGlobalTable<KeyType, ValueType> *dst = NULL;

DEFINE_int64(sort_size, 1000000, "");

class SortKernel : public DSMKernel {
public:
  void Init() {
    KeyGen k;
    for (int i = 0; i < FLAGS_sort_size / dst->num_shards(); ++i) {
      src.push_back(k.next());
    }
  }

  void Partition() {
    Bucket b;
    b.mutable_value()->Add(0);
    for (int i = 0; i < src.size(); ++i) {
      PERIODIC(1.0, LOG(INFO) << "Partitioning...." << 100. * i / src.size());
      b.set_value(0, src[i]);
      dst->put(src[i] & 0xffff, b);
    }
  }

  void Sort() {
    TypedTableIterator<KeyType, ValueType> *i = dst->get_typed_iterator(current_shard());
    while (!i->done()) {
      Bucket b = i->value();
      uint32_t* t = b.mutable_value()->mutable_data();
      std::sort(t, t + b.value_size());
      i->Next();
    }
  }

};
REGISTER_KERNEL(SortKernel);
REGISTER_METHOD(SortKernel, Init);
REGISTER_METHOD(SortKernel, Partition);
REGISTER_METHOD(SortKernel, Sort);

int IntegerSort(const ConfigData& conf) {
  dst = CreateTable(0, conf.num_workers(),
      new Sharding::UintMod, new BucketMerge);

  if (!StartWorker(conf)) {
    Master m(conf);
    m.run_all("SortKernel", " Init",  dst);
    m.run_all("SortKernel", " Partition",  dst);
    m.run_all("SortKernel", " Sort",  dst);
  }
  return 0;
}
REGISTER_RUNNER(IntegerSort);
