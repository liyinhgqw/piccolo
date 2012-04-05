#ifndef GLOBALTABLE_H_
#define GLOBALTABLE_H_

#include "kernel/table-inl.h"
#include "kernel/local-table.h"

#include "util/file.h"
#include "util/marshal.h"
#include "util/rpc.h"
#include "util/timer.h"
#include "util/tuple.h"

#include <queue>
#include <tr1/unordered_map>

//#define GLOBAL_TABLE_USE_SCOPEDLOCK

#define RETRIGGER_SCAN_INTERVAL 0.01

namespace piccolo {

class Worker;
class Master;

// Encodes table entries using the passed in TableData protocol buffer.
struct ProtoTableCoder: public TableCoder {
  ProtoTableCoder(const TableData* in);
  ~ProtoTableCoder();
  virtual void WriteEntry(StringPiece k, StringPiece v);
  virtual bool ReadEntry(string* k, string *v);
  virtual void WriteBitMap(boost::dynamic_bitset<uint32_t>*, int64_t capacity);
  virtual bool ReadBitMap(boost::dynamic_bitset<uint32_t>*, LocalTable* table);

  int read_pos_;
  TableData *t_;
};

struct PartitionInfo {
  PartitionInfo() :
      dirty(false), tainted(false) {
  }
  bool dirty;
  bool tainted;
  ShardInfo sinfo;
};

class GlobalTable: virtual public TableBase {
public:
  virtual ~GlobalTable() {
  }

  virtual void UpdatePartitions(const ShardInfo& sinfo) = 0;
  virtual TableIterator* get_iterator(int shard, unsigned int fetch_num =
                                          FETCH_NUM) = 0;

  virtual bool is_local_shard(int shard) = 0;
  virtual bool is_local_key(const StringPiece &k) = 0;

  virtual PartitionInfo* get_partition_info(int shard) = 0;
  virtual LocalTable* get_partition(int shard) = 0;

  virtual bool tainted(int shard) = 0;
  virtual int owner(int shard) = 0;

protected:
  friend class Worker;
  friend class Master;

  // Fill in a response from a remote worker for the given key.
  virtual void handle_get(const HashGet& req, TableData* resp) = 0;
  virtual int64_t shard_size(int shard) = 0;
};

class MutableGlobalTable: virtual public GlobalTable {
public:
  MutableGlobalTable() :
      applyingupdates(false) {
  }

  // Handle updates from the master or other workers.
  virtual void SendUpdates() = 0;
  virtual void SendUpdates(int* count) = 0;
  virtual void ApplyUpdates(const TableData& req) = 0;
  virtual void HandlePutRequests() = 0;
  virtual unsigned int KernelFinalize() = 0;
  virtual void retrigger_start() = 0;
  virtual unsigned int retrigger_stop() = 0;

  virtual int pending_write_bytes() = 0;
  virtual int clearUpdateQueue() = 0;

  virtual void clear() = 0;
  virtual void resize(int64_t new_size) = 0;

  // Exchange the content of this table with that of table 'b'.
  virtual void swap(GlobalTable *b) = 0;
protected:
  friend class Worker;
  virtual void local_swap(GlobalTable *b) = 0;

  bool applyingupdates;
};

class GlobalTableBase: virtual public GlobalTable {
public:
  virtual ~GlobalTableBase();

  void Init(const TableDescriptor *tinfo);

  void UpdatePartitions(const ShardInfo& sinfo);

  virtual TableIterator* get_iterator(int shard, unsigned int fetch_num =
                                          FETCH_NUM) = 0;

  virtual bool is_local_shard(int shard);
  virtual bool is_local_key(const StringPiece &k);

  int64_t shard_size(int shard);

  // Fill in a response from a remote worker for the given key.
  void handle_get(const HashGet& req, TableData* resp);

  PartitionInfo* get_partition_info(int shard) {
    return &partinfo_[shard];
  }
  LocalTable* get_partition(int shard) {
    return partitions_[shard];
  }

  bool tainted(int shard) {
    return get_partition_info(shard)->tainted;
  }
  int owner(int shard) {
    return get_partition_info(shard)->sinfo.owner();
  }
protected:
  virtual int shard_for_key_str(const StringPiece& k) = 0;

  // Fetch the given key, using only local information.
  void get_local(const StringPiece &k, string *v);

  // Fetch key k from the node owning it.  Returns true if the key exists.
  bool get_remote(int shard, const StringPiece &k, string* v);

  int worker_id_;

  // partitions_ for buffering remote writes to non-Trigger tables,
  // or writebufs_ for Trigger tables
  std::vector<LocalTable*> partitions_;
  std::vector<TableData*> writebufs_;
  std::vector<ProtoTableCoder*> writebufcoders_;

  std::vector<LocalTable*> cache_;

  boost::recursive_mutex& mutex() {
    return m_;
  }

  boost::recursive_mutex& updatequeue_mutex() {
    return m_uq_;
  }

  boost::recursive_mutex& trigger_mutex() {
    return m_trig_;
  }

  boost::mutex& retrigger_mutex() {
    return m_retrig_;
  }

  boost::recursive_mutex m_;
  boost::recursive_mutex m_uq_;
  boost::recursive_mutex m_trig_;
  boost::mutex m_retrig_;

  std::vector<PartitionInfo> partinfo_;

  struct CacheEntry {
    double last_read_time;
    string value;
  };

  std::tr1::unordered_map<StringPiece, CacheEntry> remote_cache_;
};

class MutableGlobalTableBase: virtual public GlobalTableBase,
    virtual public MutableGlobalTable,
    virtual public Checkpointable {
public:
  MutableGlobalTableBase() :
      pending_writes_(0) {
  }

  void SendUpdates();
  void SendUpdates(int* count);
  virtual void ApplyUpdates(const TableData& req) = 0;
  virtual void retrigger_start() = 0;
  virtual unsigned int retrigger_stop() = 0;
  virtual unsigned int KernelFinalize() = 0;
  void HandlePutRequests();

  int pending_write_bytes();

  void clear();
  void resize(int64_t new_size);

  void start_checkpoint(const string& f, bool deltaOnly);
  void write_delta(const TableData& d);
  void finish_checkpoint();
  void restore(const string& f);

  void swap(GlobalTable *b);

protected:
  int64_t pending_writes_;
  void local_swap(GlobalTable *b);
};

template<class K, class V>
class TypedGlobalTable: virtual public GlobalTable,
    public MutableGlobalTableBase,
    public TypedTable<K, V>,
    private boost::noncopyable {
public:
  //update queuing support
  typedef std::pair<K, V> KVPair;
  std::deque<KVPair> update_queue;
  bool clearingUpdateQueue;

  //long trigger support
  unsigned int KernelFinalize();
  int retrigger_threadcount_; // number of long trigger threads for this table
  std::vector<boost::thread::id> retrigger_threadids_; //IDs of retrigger threads, for future use
  bool retrigger_terminate_; // set to instruct retrigger threads to clear tables
  bool retrigger_updates_; // used for what triggers do in a flush/apply loop
  unsigned int retrigger_termthreads_; // number of "terminated" threads

  typedef TypedTableIterator<K, V> Iterator;
  typedef DecodeIterator<K, V> UpdateDecoder;
  virtual void Init(const TableDescriptor *tinfo, int retrigt_count) {
    GlobalTableBase::Init(tinfo);
    for (size_t i = 0; i < partitions_.size(); ++i) {
      // For non-triggered tables that allow remote accumulation
      partitions_[i] = create_local(i);

      // For triggered tables that do not allow remote accumulation
      writebufs_[i] = new TableData;
      writebufcoders_[i] = new ProtoTableCoder(writebufs_[i]);
    }

    //Clear the update queue, just in case
    clearingUpdateQueue = false;
    update_queue.clear();

    //Clear the long/retrigger map, just in case, and
    //then start up the retrigger thread(s)
    retrigger_threadids_.clear();
    retrigger_terminate_ = false;
    retrigger_threadcount_ = 0;

    //Don't create retrigger threads for classic Piccolo apps
    if (retrigt_count) {
      VLOG(1) << "Retrigger threadcount > 0, creating long trigger threads";
      for (size_t i = 0; i < partitions_.size(); ++i) {
        //Only create retrigger threads for local shards
        //if (is_local_shard(i)) {
          retrigger_threadids_.push_back(
            boost::thread(
              boost::bind(&TypedGlobalTable<K, V>::retrigger_thread,
                          this,i)).get_id());
          VLOG(1) << "Bound thread " << retrigger_threadcount_ <<
                       " to shard " << i << " on table " << tinfo->table_id;
          retrigger_threadcount_++;
        //}
      }
    }
  }

  int get_shard(const K& k);
  V get_local(const K& k);

  // Store the given key-value pair in this hash. If 'k' has affinity for a
  // remote thread, the application occurs immediately on the local host,
  // and the update is queued for transmission to the owner.
  void put(const K &k, const V &v);
  void update(const K &k, const V &v);
  void enqueue_update(K k, V v);
  void swap_accumulator(Accumulator<V>* newaccum);
  void swap_accumulator(Trigger<K, V>* newaccum);
  void swap_accumulator(HybridTrigger<K, V>* newaccum);
  int clearUpdateQueue();

  // Provide a mechanism to enable a long trigger / retrigger, as well as
  // a function from which to create a retrigger thread, and one to instruct
  // retrigger threads to clear themselves of existing items.
  void enable_retrigger(K k);
  void retrigger_start(void);
  unsigned int retrigger_stop(void);
  void retrigger_thread(int shard_id);

  // Lock for retrigger bitsets, and the two functions
  boost::dynamic_bitset<uint32_t>* bitset_getbitset(void);
  const K bitset_getkeyforbit(unsigned long int bit_offset);
  int bitset_epoch(void);

  // Return the value associated with 'k', possibly blocking for a remote fetch.
  V get(const K &k);
  bool contains(const K &k);
  void remove(const K &k);
  TableIterator* get_iterator(int shard, unsigned int fetch_num = FETCH_NUM);
  TypedTable<K, V>* partition(int idx) {
    return dynamic_cast<TypedTable<K, V>*>(partitions_[idx]);
  }
  ProtoTableCoder* writebufcoder(int idx) {
    return writebufcoders_[idx];
  }

  virtual TypedTableIterator<K, V>* get_typed_iterator(int shard,
                                                       unsigned int fetch_num =
                                                           FETCH_NUM) {
    return static_cast<TypedTableIterator<K, V>*>(get_iterator(shard, fetch_num));
  }

  void ApplyUpdates(const piccolo::TableData& req) {
    boost::recursive_mutex::scoped_lock sl(mutex());
    if (applyingupdates == true) {
      VLOG(2) << "Avoiding recursive ApplyUpdate().";
      return; //prevent recursive applyupdates
    }

    applyingupdates = true;
    VLOG(3) << "Performing non-recursive ApplyUpdate().";
    if (!is_local_shard(req.shard())) {
      LOG_EVERY_N(INFO, 1000)
        << "Forwarding push request from: " << MP(id(), req.shard()) << " to "
            << owner(req.shard());
    }

    // Changes to support locality of triggers <CRM>
    ProtoTableCoder c(&req);
    UpdateDecoder it;
    partitions_[req.shard()]->DecodeUpdates(&c, &it);
    for (; !it.done(); it.Next()) {
      update(it.key(), it.value());
    }
    applyingupdates = false;
  }

protected:
  int shard_for_key_str(const StringPiece& k);
  virtual LocalTable* create_local(int shard);

};

static const int kWriteFlushCount = 1000000;

template<class K, class V>
class RemoteIterator: public TypedTableIterator<K, V> {
public:
  RemoteIterator(TypedGlobalTable<K, V> *table, int shard
                 , unsigned int fetch_num = FETCH_NUM) :
      owner_(table), shard_(shard), done_(false), fetch_num_(fetch_num) {
    request_.set_table(table->id());
    request_.set_shard(shard_);
    request_.set_row_count(fetch_num_);
    int target_worker = table->owner(shard);

    // << CRM 2011-01-18 >>
    while (!cached_results.empty())
      cached_results.pop();

    VLOG(3)
        << "Created RemoteIterator on table " << table->id() << ", shard "
            << shard << " @" << this;
    rpc::NetworkThread::Get()->Call(target_worker + 1, MTYPE_ITERATOR, request_,
                                    &response_);
    for (size_t i = 1; i <= response_.row_count(); i++) {
      std::pair<string, string> row;
      row = make_pair(response_.key(i - 1), response_.value(i - 1));
      cached_results.push(row);
    }

    request_.set_id(response_.id());
  }

  void key_str(string *out) {
    if (!cached_results.empty())
    VLOG(4) << "Pulling first of " << cached_results.size() << " results";
    if (!cached_results.empty()) *out = cached_results.front().first;
  }

  void value_str(string *out) {
    if (!cached_results.empty())
    VLOG(4) << "Pulling first of " << cached_results.size() << " results";
    if (!cached_results.empty()) *out = cached_results.front().second;
  }

  bool done() {
    return response_.done() && cached_results.empty();
  }

  void Next() {
    int target_worker = dynamic_cast<GlobalTable*>(owner_)->owner(shard_);
    if (!cached_results.empty()) cached_results.pop();
    if (cached_results.empty()) {
      if (response_.done()) {
        return;
      }
      rpc::NetworkThread::Get()->Call(target_worker + 1, MTYPE_ITERATOR,
                                      request_, &response_);
      if (response_.row_count() < 1 && !response_.done())
        LOG(ERROR) << "Call to server requesting " << request_.row_count()
            << " rows returned " << response_.row_count() << " rows.";
      for (size_t i = 1; i <= response_.row_count(); i++) {
        std::pair<string, string> row;
        row = make_pair(response_.key(i - 1), response_.value(i - 1));
        cached_results.push(row);
      }
    } else {
      VLOG(4) << "[PREFETCH] Using cached key for Next()";
    }
    ++index_;
  }

  const K& key() {
    if (cached_results.empty())
    LOG(FATAL) << "Cache miss on key!";
    unmarshal<K>(cached_results.front().first, &key_);
    return key_;
  }

  V& value() {
    if (cached_results.empty())
    LOG(FATAL) << "Cache miss on key!";
    unmarshal<V>((cached_results.front().second), &value_);
    return value_;
  }

private:
  TableBase* owner_;
  IteratorRequest request_;
  IteratorResponse response_;
  int id_;

  int shard_;
  int index_;
  K key_;
  V value_;
  bool done_;

  // << CRM 2011-01-18 >>
  std::queue<std::pair<string, string> > cached_results;
  unsigned int fetch_num_;
};

template<class K, class V>
int TypedGlobalTable<K, V>::get_shard(const K& k) {
  DCHECK(this != NULL);
  DCHECK(this->info().sharder != NULL);

  Sharder<K> *sharder = (Sharder<K>*) (this->info().sharder);
  int shard = (*sharder)(k, this->info().num_shards);
  DCHECK_GE(shard, 0);
  DCHECK_LT(shard, this->num_shards());
  return shard;
}

template<class K, class V>
int TypedGlobalTable<K, V>::shard_for_key_str(const StringPiece& k) {
  return get_shard(unmarshal<K>(k));
}

template<class K, class V>
V TypedGlobalTable<K, V>::get_local(const K& k) {
  int shard = this->get_shard(k);

  CHECK(is_local_shard(shard)) << " non-local for shard: " << shard;

  return partition(shard)->get(k);
}

// Store the given key-value pair in this hash. If 'k' has affinity for a
// remote thread, the application occurs immediately on the local host,
// and the update is queued for transmission to the owner.
template<class K, class V>
void TypedGlobalTable<K, V>::put(const K &k, const V &v) {
  LOG(FATAL) << "Need to implement.";
  int shard = this->get_shard(k);

#ifdef GLOBAL_TABLE_USE_SCOPEDLOCK
  boost::recursive_mutex::scoped_lock sl(mutex());
#endif
  partition(shard)->put(k, v);

  if (!is_local_shard(shard)) {
    ++pending_writes_;
  }

  if (pending_writes_ > kWriteFlushCount) {
    SendUpdates();
  }

  PERIODIC(0.1, {this->HandlePutRequests();});
}

template<class K, class V>
void TypedGlobalTable<K, V>::update(const K &k, const V &v) {
  int shard = this->get_shard(k);

#ifdef GLOBAL_TABLE_USE_SCOPEDLOCK
  boost::mutex::scoped_lock sl(trigger_mutex());
  boost::recursive_mutex::scoped_lock sl(mutex());
#endif

  if (is_local_shard(shard)) {

    partition(shard)->update(k, v);

  } else {

    if (this->info().accum->type() != AccumulatorBase::TRIGGER) {
      //No triggers, remote accumulation is OK
      partition(shard)->update(k, v);

    } else {
      //Triggers, no remote accumulation allowed
      string sk, sv;
      marshal(k, &sk);
      marshal(v, &sv);
      writebufcoder(shard)->WriteEntry(sk, sv);

    }
    ++pending_writes_;
    if (pending_writes_ > kWriteFlushCount) {
      SendUpdates();
    }

    PERIODIC(0.1, {this->HandlePutRequests();});
  }

  //Deal with updates enqueued inside triggers
  clearUpdateQueue();
}

template<class K, class V>
int TypedGlobalTable<K, V>::clearUpdateQueue(void) {

  int i = 0;
  std::deque<KVPair> removed_items;
  {
    boost::recursive_mutex::scoped_lock sl(updatequeue_mutex());
    if (clearingUpdateQueue) return 0;
    clearingUpdateQueue = true; //turn recursion into iteration

  }
  int lastqueuesize = 0;
  do {
    {
      boost::recursive_mutex::scoped_lock sl(mutex());
      //Swap queue with an empty queue so we don't recurse way down
      //VLOG(3) << "clearing update queue for table " << this->id() << " of " << update_queue.size() << " items" << endl;

      removed_items.clear();
      update_queue.swap(removed_items);
      lastqueuesize = removed_items.size();
    }

    while (!removed_items.empty()) {
      KVPair thispair(removed_items.front());
      VLOG(3)
          << "Removed pair (" << (i - removed_items.size()) << " of " << i
              << ")";
      update(thispair.first, thispair.second);
      removed_items.pop_front();
      i++;
    }
  } while (lastqueuesize != 0);
  {
    boost::recursive_mutex::scoped_lock sl(updatequeue_mutex());
    clearingUpdateQueue = false; //turn recursion into iteration
  }
  VLOG(3) << "Cleared update queue of " << i << " items";
  return i;
}

template<class K, class V>
unsigned int TypedGlobalTable<K, V>::KernelFinalize() {
  unsigned int i = retrigger_stop();
  retrigger_start();
  return i;
}

template<class K, class V>
void TypedGlobalTable<K, V>::enable_retrigger(K k) {
  boost::mutex::scoped_lock sl(retrigger_mutex());

  //Set bit in table instead of in retrigger map	TODO
  LOG(FATAL) << "enable_retrigger temporarily deprecated";

  return;
}

template<class K, class V>
void TypedGlobalTable<K, V>::retrigger_start(void) {
  {
    boost::mutex::scoped_lock sl(retrigger_mutex());		//Saves us a bit of pain here
    retrigger_terminate_ = false;
    retrigger_termthreads_ = 0;
  }
  while (retrigger_termthreads_ < retrigger_threadcount_) {
    Sleep(RETRIGGER_SCAN_INTERVAL);
  }
  return;
}

template<class K, class V>
unsigned int TypedGlobalTable<K, V>::retrigger_stop(void) {
  if (retrigger_terminate_)
    return 0;
  if (!retrigger_threadcount_) {
    for (size_t i = 0; i < partitions_.size(); ++i) {
      //Only create retrigger threads for local shards
      if (is_local_shard(i)) {
        (partition(i)->bitset_getbitset())->reset();		//reset all bits in bitset
      }
    }
    return 0;
  } else {
    boost::mutex::scoped_lock sl(retrigger_mutex());		//Saves us a bit of pain here
    retrigger_termthreads_ = 0;
    retrigger_updates_ = 0;

    retrigger_terminate_ = true;
  }
  while (retrigger_termthreads_ < retrigger_threadcount_) {
    Sleep(RETRIGGER_SCAN_INTERVAL);
  }
  return retrigger_updates_;	//if non-zero, let's assume there are issues.
}

template<class K, class V>
boost::dynamic_bitset<uint32_t>* TypedGlobalTable<K,V>::bitset_getbitset(void) {
  LOG(FATAL) << "bitset_getbitset called in TypedGlobalTable";
  return NULL;
}

template<class K, class V>
const K TypedGlobalTable<K,V>::bitset_getkeyforbit(unsigned long int bit_offset) {
  LOG(FATAL) << "bitset_getkeyforbit called in TypedGlobalTable";
  K k;
  return k;
}
template<class K, class V>
int TypedGlobalTable<K,V>::bitset_epoch(void) {
  LOG(FATAL) << "bitset_epoch called in TypedGlobalTable";
  return -1;
}

template<class K, class V>
void TypedGlobalTable<K, V>::retrigger_thread(int shard_id) {
  int bitset_epoch_ = 0;		//used to handle things like resizes to prevent problems
  while (1) {
    bool terminated = false;
    unsigned int updates = 0;
    while (!terminated) {
      if (is_local_shard(shard_id)) {
        //     boost::mutex::scoped_lock sl(retrigger_mutex());
        //LOG(FATAL) << "Retrigger mode 1 not yet implemented!";

      boost::recursive_mutex::scoped_lock sl(TypedTable<K,V>::rt_bitset_mutex());
      bitset_scan_restart:
        bitset_epoch_ = partition(shard_id)->bitset_epoch();
        bool terminating = retrigger_terminate_;

        boost::dynamic_bitset<uint32_t>* ltflags = partition(shard_id)->bitset_getbitset();
        size_t bititer = ltflags->find_first();

        while(bititer != ltflags->npos) {
          //VLOG(2) << "Key on bit " << bititer << " in table " << info_.table_id << " set.";
          K key = partition(shard_id)->bitset_getkeyforbit(bititer);
          bool dorestart = false;
          {
            //boost::recursive_mutex::scoped_lock sl(TypedTable<K,V>::rt_bitset_mutex());
            if (bitset_epoch_ != partition(shard_id)->bitset_epoch()) {
              dorestart = true;
            } else {
              CHECK_EQ(true,ltflags->test(bititer)) << "Bititer flag " << bititer << " not actually set!";
              ltflags->reset(bititer);
            }
          }

          if (dorestart) {	//in case someone else resized/rearranged the hashmap
            VLOG(2) << "Tainted bitset_epoch_!";
            goto bitset_scan_restart;
          }

          bool retain = false;
          if (info_.accum->type() != AccumulatorBase::ACCUMULATOR) {
            updates++;				//this is for the Flush/Apply finalization
            retain = ((Trigger<K, V>*) info_.accum)->LongFire(		//retain temporarily removed
                     key, terminating);
            if (retain)
              enable_retrigger(key);
          }
          bititer = ltflags->find_next(bititer);
        }

        terminated = terminating;

        if (terminated && ltflags->any()) {		//if we didn't actually shut down here...
          VLOG(2) << "Terminating retrigger thread suspended pending at least " <<
                     partition(shard_id)->bitset_getbitset()->count() << " bits set";
          terminated = false;
        } else if (terminated) {
          VLOG(2) << "Terminated retrigger thread iteration with " << updates << " updates.";
          boost::mutex::scoped_lock sl(retrigger_mutex());
          retrigger_termthreads_++; //increment terminated thread count
          retrigger_updates_ += updates; //not actually terminated if more updates happened.
        }
      } else {
        if (retrigger_terminate_) {
          boost::mutex::scoped_lock sl(retrigger_mutex());
          retrigger_termthreads_++; //increment terminated thread count
          terminated = true;
        }
        //Sleep(10*RETRIGGER_SCAN_INTERVAL);
      }
      //VLOG(2) << "retrigger iteration accrued " << updates << " updates; next pass should have >=" <<
      //           partition(shard_id)->bitset_getbitset()->count() << " bits set";
      Sleep(RETRIGGER_SCAN_INTERVAL);
    }
    while(retrigger_terminate_) {
      Sleep(RETRIGGER_SCAN_INTERVAL);
    } 
 //   VLOG(2) << "Waiting for re-enable lock";
    {
      boost::mutex::scoped_lock sl(retrigger_mutex());
      retrigger_termthreads_++; //increment terminated thread count
      terminated = false;
    }
  } //now a kernel is live again or we're going through a flush/apply loop.
}

template<class K, class V>
void TypedGlobalTable<K, V>::enqueue_update(K k, V v) {
  boost::recursive_mutex::scoped_lock sl(mutex());

  const KVPair thispair(k, v);
  update_queue.push_back(thispair);
  VLOG(3)
      << "Enqueing table id " << this->id() << " update ("
          << update_queue.size() << " pending pairs)";
}

template<class K, class V>
void TypedGlobalTable<K, V>::swap_accumulator(Accumulator<V>* newaccum) {
  this->mutable_info()->swap_accumulator((AccumulatorBase*) newaccum);
  for (int i = 0; i < partitions_.size(); ++i) {
    partitions_[i]->mutable_info()->swap_accumulator(
        (AccumulatorBase*) newaccum);
  }
}
template<class K, class V>
void TypedGlobalTable<K, V>::swap_accumulator(Trigger<K, V>* newaccum) {
  this->mutable_info()->swap_accumulator((AccumulatorBase*) newaccum);
  for (size_t i = 0; i < partitions_.size(); ++i) {
    partitions_[i]->mutable_info()->swap_accumulator(
        (AccumulatorBase*) newaccum);
  }
}
template<class K, class V>
void TypedGlobalTable<K, V>::swap_accumulator(HybridTrigger<K, V>* newaccum) {
  this->mutable_info()->swap_accumulator((AccumulatorBase*) newaccum);
  for (size_t i = 0; i < partitions_.size(); ++i) {
    partitions_[i]->mutable_info()->swap_accumulator(
        (AccumulatorBase*) newaccum);
  }
}

// Return the value associated with 'k', possibly blocking for a remote fetch.
template<class K, class V>
V TypedGlobalTable<K, V>::get(const K &k) {
  int shard = this->get_shard(k);

  // If we received a request for this shard; but we haven't received all of the
  // data for it yet. Continue reading from other workers until we do.
  // New for triggers: be sure to not recursively apply updates.
  if (tainted(shard)) {
    boost::recursive_mutex::scoped_lock sl(mutex());
    if (applyingupdates == false) {
      applyingupdates = true;
      while (tainted(shard)) {
        this->HandlePutRequests();
        sched_yield();
      }
      applyingupdates = false;
    }
  }

  PERIODIC(0.1, this->HandlePutRequests());

  if (is_local_shard(shard)) {
#ifdef GLOBAL_TABLE_USE_SCOPEDLOCK
    boost::recursive_mutex::scoped_lock sl(mutex());
#endif
    return partition(shard)->get(k);
  }

  string v_str;
  get_remote(shard, marshal(k), &v_str);
  return unmarshal<V>(v_str);
}

template<class K, class V>
bool TypedGlobalTable<K, V>::contains(const K &k) {
  int shard = this->get_shard(k);

  // If we received a request for this shard; but we haven't received all of the
  // data for it yet. Continue reading from other workers until we do.
  // New for triggers: be sure to not recursively apply updates.
  if (tainted(shard)) {
    boost::recursive_mutex::scoped_lock sl(mutex());
    if (applyingupdates == false) {
      applyingupdates = true;
      while (tainted(shard)) {
        this->HandlePutRequests();
        sched_yield();
      }
      applyingupdates = false;
    }
  }

  if (is_local_shard(shard)) {
#ifdef GLOBAL_TABLE_USE_SCOPEDLOCK
    boost::recursive_mutex::scoped_lock sl(mutex());
#endif
    return partition(shard)->contains(k);
  }

  string v_str;
  return get_remote(shard, marshal(k), &v_str);
}

template<class K, class V>
void TypedGlobalTable<K, V>::remove(const K &k) {
  LOG(FATAL) << "Not implemented!";
}

template<class K, class V>
LocalTable* TypedGlobalTable<K, V>::create_local(int shard) {
  TableDescriptor *linfo = new TableDescriptor(info());
  linfo->shard = shard;
  LocalTable* t = (LocalTable*) info_.partition_factory->New();
  t->Init(linfo);

  return t;
}

template<class K, class V>
TableIterator* TypedGlobalTable<K, V>::get_iterator(int shard,
                                                    unsigned int fetch_num) {
  if (this->is_local_shard(shard)) {
    return (TypedTableIterator<K, V>*) partitions_[shard]->get_iterator();
  } else {
    return new RemoteIterator<K, V>(this, shard, fetch_num);
  }
}

}

#endif /* GLOBALTABLE_H_ */
