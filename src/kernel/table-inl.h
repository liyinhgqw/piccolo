#ifndef TABLE_INL_H_
#define TABLE_INL_H_

#include "kernel/table.h"

namespace piccolo {

struct TableDescriptor {
public:
  TableDescriptor() {
    Reset();
  }

  TableDescriptor(int id, int shards) {
    Reset();
    table_id = id;
    num_shards = shards;
  }

  void Reset() {
    table_id = -1;
    num_shards = -1;
    block_size = 500;
    max_stale_time = 0.;
    helper = NULL;
    partition_factory = NULL;
    block_info = NULL;
    accum = NULL;
    sharder = NULL;
    //triggers.clear();
  }

  void swap_accumulator(AccumulatorBase* newaccum) {
    //delete accum;
    accum = newaccum;
    return;
  }

  int table_id;
  int num_shards;

  // For local tables, the shard of the global table they represent.
  int shard;
  int default_shard_size;

//  vector<TriggerBase*> triggers;

  AccumulatorBase *accum;
  SharderBase *sharder;

  // For global tables, factory for constructing new partitions.
  TableFactory *partition_factory;

  // For dense tables, information on block layout and size.
  int block_size;
  BlockInfoBase *block_info;

  // For global tables, the maximum amount of time to cache remote values
  double max_stale_time;

  // For global tables, reference to the local worker.  Used for passing
  // off remote access requests.
  TableHelper *helper;
};


// Methods common to both global and local table views.
class TableBase: public Table {
public:
  typedef TableIterator Iterator;
  virtual void Init(const TableDescriptor * info) {
    info_ = *info;
  }

  const TableDescriptor & info() const {
    return info_;
  }

  TableDescriptor* mutable_info() {
    return &info_;
  }

  int id() const {
    return info().table_id;
  }

  int num_shards() const {
    return info().num_shards;
  }

  TableHelper *helper() {
    return info().helper;
  }
  int helper_id() {
    return helper()->id();
  }

  void set_helper(TableHelper *w) {
    info_.helper = w;
  }

protected:
  TableDescriptor info_;
};

// Added for the sake of triggering on remote updates/puts <CRM>
template<typename K, typename V>
struct DecodeIterator: public DecodeIteratorBase,
    public TypedTableIterator<K, V> {

  DecodeIterator() {
    clear();
    rewind();
  }

  void append(K k, V v) {
    kvpair thispair(k, v);
    decode_queue_.push_back(thispair);
  }

  void clear() {
    decode_queue_.clear();
  }

  void rewind() {
    queue_iter_ = decode_queue_.begin();
  }

  bool done() {
    return queue_iter_ == decode_queue_.end();
  }

  void Next() {
    queue_iter_++;
  }

  const K& key() {
    return queue_iter_->first;
  }

  V& value() {
    return queue_iter_->second;
  }

private:
  typedef std::pair<K, V> kvpair;
  std::vector<kvpair> decode_queue_;
  typename std::vector<kvpair>::iterator queue_iter_;
};

}

#endif /* TABLE_INL_H_ */
