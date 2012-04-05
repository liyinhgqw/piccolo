#ifndef SPARSE_MAP_H_
#define SPARSE_MAP_H_

#include "util/common.h"
#include "worker/worker.pb.h"
#include "kernel/table.h"
#include "kernel/local-table.h"
#include <boost/noncopyable.hpp>
#include <boost/dynamic_bitset.hpp>

namespace piccolo {

static const double kLoadFactor = 0.4;

template <class K, class V>
class SparseTable :
  public LocalTable,
  public TypedTable<K, V>,
  private boost::noncopyable {
private:
#pragma pack(push, 1)
  struct Bucket {
    K k;
    V v;
    bool in_use;
  };
#pragma pack(pop)

public:
  typedef DecodeIterator<K, V> UpdateDecoder;

  struct Iterator : public TypedTableIterator<K, V> {
    Iterator(SparseTable<K, V>& parent) : pos(-1), parent_(parent) {
      Next();
    }

    void Next() {
      do {
        ++pos;
      } while (pos < parent_.size_ && !parent_.buckets_[pos].in_use);
    }

    bool done() {
      return pos == parent_.size_;
    }

    const K& key() { return parent_.buckets_[pos].k; }
    V& value() { return parent_.buckets_[pos].v; }

    int pos;
    SparseTable<K, V> &parent_;
  };

  struct Factory : public TableFactory {
    TableBase* New() { return new SparseTable<K, V>(); }
  };

  // Construct a SparseTable with the given initial size; it will be expanded as necessary.
  SparseTable(int size=1);
  ~SparseTable() {}

  void Init(const TableDescriptor * td) {
    TableBase::Init(td);
  }

  V get(const K& k);
  bool contains(const K& k);
  void put(const K& k, const V& v);
  void update(const K& k, const V& v);
  void remove(const K& k) {
    LOG(FATAL) << "Not implemented.";
  }

  boost::dynamic_bitset<uint32_t>* bitset_getbitset(void);
  const K bitset_getkeyforbit(unsigned long int bit_offset);
  int bitset_epoch(void);

  void resize(int64_t size);

  bool empty() { return size() == 0; }
  int64_t size() { return entries_; }
  int64_t capacity() { return size_; }

  void clear() {
    for (int i = 0; i < size_; ++i) { buckets_[i].in_use = 0; }
    entries_ = 0;
    trigger_flags_.reset();
  }

  TableIterator *get_iterator() {
      return new Iterator(*this);
  }

  void Serialize(TableCoder *out, bool tryOptimize = false);
  void Deserialize(TableCoder *in, bool tryOptimize = false);

  void DecodeUpdates(TableCoder *in, DecodeIteratorBase *itbase);

private:
  uint32_t bucket_idx(K k) {
    return hashobj_(k) % size_;
  }

  int bucket_for_key(const K& k) {
    int start = bucket_idx(k);
    int b = start;
    int tries = 0;

    do {
      ++tries;
      if (buckets_[b].in_use) {
        if (buckets_[b].k == k) {
//		  EVERY_N(10000, fprintf(stderr, "ST:: Tries = %d\n", tries))
          return b;
        }
      } else {
//		  EVERY_N(10000, fprintf(stderr, "ST:: Tries = %d\n", tries))
        return -1;
      }

       b = (b + 1) % size_;
    } while (b != start);

//		  EVERY_N(10000, fprintf(stderr, "ST:: Tries = %d\n", tries))
    return -1;
  }

  std::vector<Bucket> buckets_;
  boost::dynamic_bitset<uint32_t> trigger_flags_;		//Retrigger flags

  int64_t entries_;
  int64_t size_;
  int bitset_epoch_;

  std::tr1::hash<K> hashobj_;
};

template <class K, class V>
SparseTable<K, V>::SparseTable(int size)
  : buckets_(0), entries_(0), size_(0), bitset_epoch_(0) {
  clear();

  trigger_flags_.resize(size);        //Retrigger flags
  trigger_flags_.reset();
  resize(size);
}

template <class K, class V>
void SparseTable<K, V>::Serialize(TableCoder *out, bool tryOptimize) {
  if (tryOptimize && boost::is_pod<K>::value && (boost::is_pod<V>::value)) {
    //optimize!
    StringPiece k("rawdump");
    VLOG(1) << "Optimized dump of " << (buckets_.size()*sizeof(Bucket)) << " bytes of data in " <<
               buckets_.size() << " buckets of " << sizeof(Bucket) << " bytes each.";
    StringPiece v((const char*)&(buckets_[0]),buckets_.size()*sizeof(Bucket));
    out->WriteEntry(k,v);
  } else {
    Iterator *i = (Iterator*)get_iterator();
    string k, v;
    while (!i->done()) {
      k.clear(); v.clear();
      marshal(i->key(), &k);
      marshal(i->value(), &v);
      out->WriteEntry(k, v);
      i->Next();
    }
    delete i;
  }
}

template <class K, class V>
void SparseTable<K, V>::Deserialize(TableCoder *in, bool tryOptimize) {
  string k,v;
  if (tryOptimize && boost::is_pod<K>::value && boost::is_pod<V>::value) {
    //optimize!
    CHECK_EQ(true,in->ReadEntry(&k,&v)) << "Failed to read raw table dump!";
    VLOG(1) << "Optimized restore of " << v.length() << " bytes of data";
    memcpy(&(buckets_[0]),v.c_str(),v.length());
  } else {
    while (in->ReadEntry(&k, &v)) {
      update_str(k, v);
    }
  }
}

template <class K, class V>
void SparseTable<K, V>::DecodeUpdates(TableCoder *in, DecodeIteratorBase *itbase) {
  UpdateDecoder* it = static_cast<UpdateDecoder*>(itbase);
  K k;
  V v;
  string kt, vt;

  it->clear();
  while (in->ReadEntry(&kt, &vt)) {
    unmarshal(kt, &k);
    unmarshal(vt, &v);
    it->append(k, v);
  }
  it->rewind();
  return;
}

template <class K, class V>
void SparseTable<K, V>::resize(int64_t size) {
  CHECK_GT(size, 0);
  if (size_ == size)
    return;

  boost::recursive_mutex::scoped_lock sl(TypedTable<K,V>::rt_bitset_mutex());	//prevent a bunch of nasty resize side-effects

  std::vector<Bucket> old_b = buckets_;

  size_t setbits = trigger_flags_.count();
  boost::dynamic_bitset<uint32_t> old_ltflags;
  old_ltflags.resize(trigger_flags_.size());
  for(int i=0; i<trigger_flags_.size(); i++)
    old_ltflags.set(i,trigger_flags_.test(i));

  int old_entries = entries_;

  VLOG(2) << "Rehashing... " << entries_ << " : " << size_ << " -> " << size;

  buckets_.resize(size);
  size_ = size;
  clear();

  trigger_flags_.resize(size_);
  trigger_flags_.reset();
  bitset_epoch_++;						//prevent resize side effects for scanning long thread

  for (size_t i = 0; i < old_b.size(); ++i) {
    if (old_b[i].in_use) {
      put(old_b[i].k, old_b[i].v);
      trigger_flags_[bucket_for_key(old_b[i].k)] = old_ltflags[i];
    }
  }

  CHECK_EQ(old_entries, entries_);
  CHECK_EQ(setbits,trigger_flags_.count()) << "Pre-resize had " << setbits << " bits set, but post-resize has " << trigger_flags_.count() << " bits set.";

  //VLOG(2) << "Rehashed " << entries_ << " : " << size_ << " -> " << size << " with " << setbits << " bits set on";
}

template <class K, class V>
bool SparseTable<K, V>::contains(const K& k) {
  return bucket_for_key(k) != -1;
}

template <class K, class V>
V SparseTable<K, V>::get(const K& k) {
  int b = bucket_for_key(k);
  //The following key display is a hack hack hack and only yields valid
  //results for ints.  It will display nonsense for other types.
  CHECK_NE(b, -1) << "No entry for requested key <" << *((int*)&k) << ">";

  return buckets_[b].v;
}

template <class K, class V>
void SparseTable<K, V>::update(const K& k, const V& v) {
  int b = bucket_for_key(k);

  if (b != -1) {
    if (info_.accum->type() == AccumulatorBase::ACCUMULATOR) {
      ((Accumulator<V>*)info_.accum)->Accumulate(&buckets_[b].v, v);
    } else if (info_.accum->type() == AccumulatorBase::HYBRID) {
      trigger_flags_[b] |= ((HybridTrigger<K,V>*)info_.accum)->Accumulate(&buckets_[b].v, v);

      //VLOG(2) << "HYBRID Setting (maybe = " << trigger_flags_.test(b) << ") bit " << b << " for key " << k
      //        << ", now " << trigger_flags_.count() << " bits set.";

    } else if (info_.accum->type() == AccumulatorBase::TRIGGER) {
      V v2 = buckets_[b].v;
      bool doUpdate = false;
      //LOG(INFO) << "Executing Trigger [sparse]";
      ((Trigger<K,V>*)info_.accum)->Fire(&k,&v2,v,&doUpdate,false);	//isNew=false
      if (doUpdate) {
        buckets_[b].v = v2;
        trigger_flags_.set(b);
        //VLOG(1) << "TRIGGER Setting bit " << b << " for key " << k;
      }
    } else {
      LOG(FATAL) << "update() called with neither TRIGGER nor ACCUMULATOR nor HYBRID";
    }
  } else {
    if (info_.accum->type() == AccumulatorBase::TRIGGER) {
      bool doUpdate = true;
      V v2 = v;
      //LOG(INFO) << "Executing Trigger [sparse] [new]";
      ((Trigger<K,V>*)info_.accum)->Fire(&k,&v2,v,&doUpdate,true); //isNew=true
      if (doUpdate) {
        put(k, v2);
        //VLOG(1) << "TRIGGER Setting bit " << b << " for key " << k;
      }
    } else {
      put(k, v);
    }
  }
}

template <class K, class V>
void SparseTable<K, V>::put(const K& k, const V& v) {
  int start = bucket_idx(k);
  int b = start;
  bool found = false;

  do {
    if (!buckets_[b].in_use) {
      break;
    }

    if (buckets_[b].k == k) {
      found = true;
      break;
    }

    b = (b + 1) % size_;
  } while(b != start);

  // Inserting a new entry:
  if (!found) {
    if (entries_ > size_ * kLoadFactor) {
      resize((int)(1 + size_ * 2));
      put(k, v);
    } else {
      buckets_[b].in_use = 1;
      buckets_[b].k = k;
      buckets_[b].v = v;
      if (info_.accum->type() == AccumulatorBase::HYBRID ||
          info_.accum->type() == AccumulatorBase::TRIGGER) {
        trigger_flags_[b] = true;
      }
      ++entries_;
    }
  } else {
    // Replacing an existing entry
    buckets_[b].v = v;
  }
}

template <class K, class V>
boost::dynamic_bitset<uint32_t>* SparseTable<K, V>::bitset_getbitset(void) {
  // boost::recursive_mutex::scoped_lock sl(TypedTable<K,V>::rt_bitset_mutex());	//prevent a bunch of nasty resize side-effects
  //VLOG(2) << "First set bit is " << trigger_flags_.find_first() << " where size " << trigger_flags_.size() << "==" << size_;
  return &trigger_flags_;
}

template <class K, class V>
const K SparseTable<K, V>::bitset_getkeyforbit(unsigned long int bit_offset) {
  //boost::recursive_mutex::scoped_lock sl(TypedTable<K,V>::rt_bitset_mutex());	//prevent a bunch of nasty resize side-effects
  CHECK_GT(size_,(int64_t) bit_offset);
  K k = buckets_[bit_offset].k;
  //VLOG(2) << "Getting bit " << bit_offset << " for key " << k;
  return k;
}

template <class K, class V>
int SparseTable<K, V>::bitset_epoch() {
  //boost::recursive_mutex::scoped_lock sl(TypedTable<K,V>::rt_bitset_mutex());	//prevent a bunch of nasty resize side-effects
  return bitset_epoch_;
}

}	/* namespace piccolo */

#endif /* SPARSE_MAP_H_ */
