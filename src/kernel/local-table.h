#ifndef LOCALTABLE_H_
#define LOCALTABLE_H_

#include "kernel/table.h"
#include "kernel/table-inl.h"
#include "util/file.h"
#include "util/rpc.h"

namespace piccolo {

// Represents a single shard of a partitioned global table.
class LocalTableCoder;

class LocalTable :
  public TableBase,
  virtual public UntypedTable,
  public Checkpointable,
  public Serializable {
public:
  LocalTable() : delta_file_(NULL) {}
  virtual ~LocalTable() {}
  bool empty() { return size() == 0; }

  void start_checkpoint(const string& f, bool deltaOnly);
  void finish_checkpoint();
  void restore(const string& f);
  void write_delta(const TableData& put);

  virtual int64_t size() = 0;
  virtual int64_t capacity() = 0;
  virtual void clear() = 0;
  virtual void resize(int64_t size) = 0;

  virtual TableIterator* get_iterator() = 0;

  int shard() { return info_.shard; }

protected:
  friend class GlobalTable;
  LocalTableCoder *delta_file_;
};

}

#endif /* LOCALTABLE_H_ */
