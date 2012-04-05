#ifndef ACCUMULATOR_H
#define ACCUMULATOR_H

#define FETCH_NUM 2048

#include "util/common.h"
#include "util/file.h"
#include "util/marshal.h"
#include "worker/worker.pb.h"
#include <boost/thread.hpp>
#include <boost/dynamic_bitset.hpp>

namespace piccolo {

struct Table;
struct TableBase;

class TableData;
class TableDescriptor;

// This interface is used by global tables to communicate with the outside
// world and determine the current state of a computation.
struct TableHelper {
  virtual int id() const = 0;
  virtual int epoch() const = 0;
  virtual int peer_for_shard(int table, int shard) const = 0;
  virtual void HandlePutRequest() = 0;
};

struct SharderBase {
};

struct AccumulatorBase {
  enum Type {
    ACCUMULATOR, TRIGGER, HYBRID
  };

  virtual Type type() = 0;
};

struct BlockInfoBase {
};

struct DecodeIteratorBase {
};

typedef int TriggerID;
struct TriggerBase {
  Table *table;
  TableHelper *helper;
  TriggerID triggerid;

  TriggerBase() {
    enabled_ = true;
  }

  virtual void enable(bool enabled) {
    enabled_ = enabled;
  }

  virtual bool enabled() {
    return enabled_;
  }

private:
  bool enabled_;
};

#ifndef SWIG

// Each table is associated with a single accumulator.  Accumulators are
// applied whenever an update is supplied for an existing key-value cell.
template<class V>
struct Accumulator: public AccumulatorBase {
  Type type() {
    return AccumulatorBase::ACCUMULATOR;
  }
  virtual void Accumulate(V* a, const V& b) = 0;
};

template<class K, class V>
struct Trigger: public AccumulatorBase {
  Type type() {
    return AccumulatorBase::TRIGGER;
  }

  virtual void Fire(const K* key, V* value, const V& updateval, bool* doUpdate,
                    bool isNew) = 0;
  virtual bool LongFire(const K key, bool) = 0;
};

template<class K, class V>
struct HybridTrigger: public AccumulatorBase {
  Type type() {
    return AccumulatorBase::HYBRID;
  }

  virtual bool Accumulate(V* a, const V& b) = 0;
  virtual bool LongFire(const K key, bool) = 0;
};

template<class K>
struct Sharder: public SharderBase {
  virtual int operator()(const K& k, int shards) = 0;
};

// Commonly-used trigger operators.
template<class K, class V>
struct Triggers {
  struct NullTrigger: public Trigger<K, V> {
    void Fire(const K* key, V* value, const V& updateval, bool* doUpdate,
              bool isNew) {
      *value = updateval;
      *doUpdate = true;
      return;
    }
    bool LongFire(const K key, bool lastrun) {
      return false;
    }
  };
  struct ReadOnlyTrigger: public Trigger<K, V> {
    void Fire(const K* key, V* value, const V& updateval, bool* doUpdate,
              bool isNew) {
      *doUpdate = false;
      return;
    }
    bool LongFire(const K key, bool lastrun) {
      return false;
    }
  };
};

// Commonly used accumulation and sharding operators.
template<class V>
struct Accumulators {
  struct Min: public Accumulator<V> {
    void Accumulate(V* a, const V& b) {
      *a = std::min(*a, b);
    }
  };

  struct Max: public Accumulator<V> {
    void Accumulate(V* a, const V& b) {
      *a = std::max(*a, b);
    }
  };

  struct Sum: public Accumulator<V> {
    void Accumulate(V* a, const V& b) {
      *a = *a + b;
    }
  };

  struct Replace: public Accumulator<V> {
    void Accumulate(V* a, const V& b) {
      *a = b;
    }
  };
};

struct Sharding {
  struct String: public Sharder<string> {
    int operator()(const string& k, int shards) {
      return StringPiece(k).hash() % shards;
    }
  };

  struct Mod: public Sharder<int> {
    int operator()(const int& key, int shards) {
      return key % shards;
    }
  };

  struct UintMod: public Sharder<uint32_t> {
    int operator()(const uint32_t& key, int shards) {
      return key % shards;
    }
  };
};
#endif /* SWIG */

struct TableFactory {
  virtual TableBase* New() = 0;
};

class TableIterator;

struct Table {
  virtual const TableDescriptor& info() const = 0;
  virtual TableDescriptor* mutable_info() = 0;
  virtual int id() const = 0;
  virtual int num_shards() const = 0;
};

struct UntypedTable {
  virtual bool contains_str(const StringPiece& k) = 0;
  virtual string get_str(const StringPiece &k) = 0;
  virtual void update_str(const StringPiece &k, const StringPiece &v) = 0;
  virtual boost::dynamic_bitset<uint32_t>* bitset_getbitset(void) = 0;
  virtual boost::recursive_mutex& rt_bitset_mutex() = 0;
};

struct TableIterator {
  virtual void key_str(string *out) = 0;
  virtual void value_str(string *out) = 0;
  virtual bool done() = 0;
  virtual void Next() = 0;
};

// Key/value typed interface.
template<class K, class V>
class TypedTable: virtual public UntypedTable {
public:
  virtual bool contains(const K &k) = 0;
  virtual V get(const K &k) = 0;
  virtual void put(const K &k, const V &v) = 0;
  virtual void update(const K &k, const V &v) = 0;
  virtual void remove(const K &k) = 0;

  virtual boost::dynamic_bitset<uint32_t>* bitset_getbitset(void) = 0;
  virtual const K bitset_getkeyforbit(unsigned long int bit_offset) = 0;
  virtual int bitset_epoch(void) = 0;
  boost::recursive_mutex& rt_bitset_mutex() {
    return rt_bitset_m_;
  }
  boost::recursive_mutex rt_bitset_m_;


  // Default specialization for untyped methods
  virtual bool contains_str(const StringPiece& s) {
    return contains(unmarshal<K>(s));
  }

  virtual string get_str(const StringPiece &s) {
    return marshal(get(unmarshal<K>(s)));
  }

  virtual void update_str(const StringPiece& kstr, const StringPiece &vstr) {
    update(unmarshal<K>(kstr), unmarshal<V>(vstr));
  }

  virtual ~TypedTable() {
  }
protected:
};

template<class K, class V>
struct TypedTableIterator: public TableIterator {
  virtual const K& key() = 0;
  virtual V& value() = 0;

  virtual void key_str(string *out) {
    marshal<K>(key(), out);
  }
  virtual void value_str(string *out) {
    marshal<V>(value(), out);
  }

  virtual ~TypedTableIterator() {
  }

protected:
};

template<class K, class V> struct TypedTableIterator;
template<class K, class V> struct Trigger;
template<class K, class V> struct PythonTrigger;

// Checkpoint and restoration.
class Checkpointable {
public:
  virtual void start_checkpoint(const string& f, bool deltaOnly) = 0;
  virtual void write_delta(const TableData& put) = 0;
  virtual void finish_checkpoint() = 0;
  virtual void restore(const string& f) = 0;
};

// Interface for serializing tables, either to disk or for transmitting via RPC.
class LocalTable;
struct TableCoder {
  virtual void WriteEntry(StringPiece k, StringPiece v) = 0;
  virtual bool ReadEntry(string* k, string *v) = 0;
  virtual void WriteBitMap(boost::dynamic_bitset<uint32_t>*, int64_t capacity) = 0;
  virtual bool ReadBitMap(boost::dynamic_bitset<uint32_t>*, LocalTable* table) = 0;
};

class Serializable {
public:
  virtual void DecodeUpdates(TableCoder *in, DecodeIteratorBase *it) = 0;
  virtual void Serialize(TableCoder* out, bool tryOptimize = false) = 0;
  virtual void Deserialize(TableCoder* in, bool tryOptimize = false) = 0;
};
}

#endif
