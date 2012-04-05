#ifndef DISKTABLE_H_
#define DISKTABLE_H_

// DiskTables present a globally accessible, read-only interface to a set of
// files stored in a GFS.
//
// RecordTables wrap the RecordFile interface, and provide convenient access for
// streaming and random-access reads. TextTables read values from newline-delimited files;
// keys are simply the position of a line within it's shard.
//
// All DiskTables types can optionally sub-divide large files into smaller chunks.  These
// are presented to the user as separate input shards which can be mapped over.

#include "kernel/table.h"
#include "kernel/global-table.h"

#include <vector>

namespace google {
namespace protobuf {
class Message;
}
}

using google::protobuf::Message;

namespace piccolo {

struct FilePartition {
  File::Info info;
  uint64_t start_pos;
  uint64_t end_pos;
};

template<class K, class V>
class DiskTable: public GlobalTableBase {
public:
  typedef TypedTableIterator<K, V> Iterator;

  DiskTable(StringPiece filepattern, uint64_t split_files_at);
  void Init(const TableDescriptor *tinfo);

  virtual TableIterator* get_iterator(int shard, unsigned int fetch_num =
                                          FETCH_NUM) = 0;
  virtual TypedTableIterator<K, V>* get_typed_iterator(int shard) {
    return (TypedTableIterator<K, V>*) get_iterator(shard); //TODO?????
  }

  int64_t shard_size(int shard);
  int shard_for_key_str(const StringPiece &k) {
    return 0;
  }
protected:
  std::vector<FilePartition*> pinfo_;
};

TypedTableIterator<uint64_t, Message>* CreateRecordIterator(FilePartition info,
                                                            Message* msg);

template<class MessageClass>
class RecordTable: public DiskTable<uint64_t, MessageClass> {
public:
  typedef TypedTableIterator<uint64_t, MessageClass> Iterator;
  RecordTable(StringPiece filepattern, uint64_t split_files_at = 0) :
      DiskTable<uint64_t, MessageClass>(filepattern, split_files_at) {
  }

  Iterator *get_iterator(int shard, unsigned int fetch_num) {
    return (Iterator*) CreateRecordIterator(*this->pinfo_[shard],
                                            new MessageClass);
  }
private:
};

class TextTable: public DiskTable<uint64_t, string> {
public:
  typedef TypedTableIterator<uint64_t, string> Iterator;

  TextTable(StringPiece filepattern, uint64_t split_files_at = 0) :
      DiskTable<uint64_t, string>(filepattern, split_files_at) {
  }
  TypedTableIterator<uint64_t, string> *get_iterator(int shard,
                                                     unsigned int fetch_num =
                                                         FETCH_NUM);
};

template<class K, class V>
DiskTable<K, V>::DiskTable(StringPiece file_pattern, uint64_t split_files_at) {
  std::vector<File::Info> files = File::MatchingFileinfo(file_pattern);
  CHECK(!files.empty())
      << "DiskTable failed to find any matching files for pattern: "
          << file_pattern.AsString();

  if (split_files_at == 0) {
    split_files_at = ULONG_MAX;
  }

  for (size_t i = 0; i < files.size(); ++i) {
    File::Info fi = files[i];
    VLOG(3)
        << "DiskTable populating from file " << (i + 1) << " of "
            << files.size() << " <" << fi.name << ">";
    for (uint64_t j = 0; j < fi.stat.st_size; j += split_files_at) {
      VLOG(3) << j << " of " << fi.stat.st_size << " with " << split_files_at;
      FilePartition*p = new FilePartition();
      p->info = fi;
      p->start_pos = j;
      p->end_pos = std::min(j + split_files_at, (uint64_t) fi.stat.st_size);
      pinfo_.push_back(p);
    }
  }
}

template<class K, class V>
int64_t DiskTable<K, V>::shard_size(int shard) {
  return pinfo_[shard]->end_pos - pinfo_[shard]->start_pos;
}

template<class K, class V>
void DiskTable<K, V>::Init(const TableDescriptor *tinfo) {
  ((TableDescriptor*) tinfo)->num_shards = pinfo_.size();
  GlobalTableBase::Init(tinfo);
}

}

#endif /* DISKTABLE_H_ */
