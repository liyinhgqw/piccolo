#include "disk-table.h"
#include "util/file.h"

using google::protobuf::Message;
namespace piccolo {

struct RecordIterator : public TypedTableIterator<uint64_t, Message> {
  virtual ~RecordIterator () {}
  RecordIterator(const FilePartition& p, Message *msg) : p_(p), r_(p.info.name, "r") {
    r_.seek(p.start_pos);
    data_ = msg;
    Next();
  }

  const uint64_t& key() { return pos_; }
  Message& value() { return *data_; }

  void key_str(string *out) { kmarshal_.marshal(pos_, out); }
  void value_str(string *out) { vmarshal_.marshal(*data_, out); }

  bool done() {
//    LOG(INFO) << "RecordIterator: done()" << p_.info.name << " : " << (pos_ > p_.end_pos) << ":: " << done_;
    return done_ || pos_ > p_.end_pos;
  }

  void Next() {
    done_ = !r_.read(data_);
    pos_ = r_.fp->tell();
//    LOG(INFO) << "RecordIterator: Next()" << p_.info.name << " : " << (pos_ > p_.end_pos) << ":: " << done_;
  }

  uint64_t pos_;
  bool done_;
  Message *data_;
  FilePartition p_;
  RecordFile r_;

  Marshal<uint64_t> kmarshal_;
  Marshal<Message> vmarshal_;
};

TypedTableIterator<uint64_t, Message>* CreateRecordIterator(FilePartition p, Message *msg) {
  return new RecordIterator(p, msg);
}

struct TextIterator : public TypedTableIterator<uint64_t, string> {
  TextIterator(const FilePartition& p) : p_(p), f_(p.info.name, "r") {
    f_.seek(p.start_pos);
    done_ = false;
    Next();
  }

  const uint64_t& key() { return pos_; }
  string& value() { return line_; }
  void key_str(string *out) { return kmarshal_.marshal(pos_, out); }
  void value_str(string *out) { vmarshal_.marshal(line_, out); }

  bool done() { return done_ || f_.eof() || f_.tell() >= p_.end_pos; }

  void Next() {
    if (!f_.read_line(&line_)) { done_ = true; }
  }

  bool done_;
  uint64_t pos_;
  string line_;
  FilePartition p_;
  LocalFile f_;

  Marshal<uint64_t> kmarshal_;
  Marshal<string> vmarshal_;
};

TypedTableIterator<uint64_t, string> *TextTable::get_iterator(int shard,unsigned int fetch_num) {
  return new TextIterator(*pinfo_[shard]);
}

}


