#include "kernel/table.h"
#include "kernel/local-table.h"
#include "util/stats.h"
#include "util/timer.h"
#include <boost/dynamic_bitset.hpp>

namespace piccolo {

// Encodes or decodes table entries, reading and writing from the
// specified file.
struct LocalTableCoder : public TableCoder {
  LocalTableCoder(const string &f, const string& mode, int64_t* capacity);
  virtual ~LocalTableCoder();

  virtual void WriteEntry(StringPiece k, StringPiece v);
  virtual bool ReadEntry(string* k, string *v);

  virtual void WriteBitMap(boost::dynamic_bitset<uint32_t>*, int64_t tablesize);
  virtual bool ReadBitMap(boost::dynamic_bitset<uint32_t>*,LocalTable* table);

  RecordFile *f_;
};

void LocalTable::start_checkpoint(const string& f, bool deltaOnly) {
  VLOG(1) << "Start checkpoint " << f;
  Timer t;
  int64_t table_capacity;

  if (!deltaOnly) {
    Marshal<int64_t>  m_int64;

    // Do a full checkpoint
    table_capacity = capacity();
    LocalTableCoder c(f, "w", &table_capacity);
    Serialize(&c, true);		//tryOptimize = true

    // Set up for checkpointing bitset
    LocalTableCoder d(f + ".bitmap", "w", &table_capacity);

    {
      boost::recursive_mutex::scoped_lock sl(rt_bitset_mutex()); //prevent a bunch of nasty resize side-effects
      d.WriteBitMap(bitset_getbitset(),capacity());
    }
  }

  delta_file_ = new LocalTableCoder(f + ".delta", "w", &table_capacity);
  VLOG(1) << "Flushed to disk in: " << t.elapsed();
}

void LocalTable::finish_checkpoint() {
  if (delta_file_) {
	VLOG(1) << "Closing delta log.";
    delete delta_file_;
    delta_file_ = NULL;
  }
}

void LocalTable::restore(const string& f) {
  string k, v;
  int64_t table_capacity;
  if (!File::Exists(f)) {
    //this is no longer a return-able condition because there might
    //be epochs that are just deltas for continuous checkpointing
    VLOG(2) << "Skipping full restore of non-existent shard " << f;
  } else {

    VLOG(2) << "Restoring full snapshot " << f;
    //TableData p;
    Timer t;

    //Resize the table as needed
    LocalTableCoder rf(f, "r", &table_capacity);
    resize(table_capacity);

    Deserialize(&rf,true);	//tryOptimize = true
	VLOG(1) << "Restored full snapshot '" << f << "' in " << t.elapsed() << " sec";
  }

  if (!File::Exists(f + ".bitmap")) {
    VLOG(2) << "Skipping full restore of missing bitmap " << f << ".bitmap";
  } else {
    //return the bitmap
    VLOG(2) << "Restoring full snapshot bitmap" << f + ".bitmap";
    LocalTableCoder rfbm(f + ".bitmap", "r", &table_capacity);
    resize(table_capacity);
    rfbm.ReadBitMap(bitset_getbitset(),this);
  }

  if (!File::Exists(f + ".delta")) {
    VLOG(2) << "Skipping delta restore of missing delta " << f << ".delta";
  } else {
    // Replay delta log.
    LocalTableCoder df(f + ".delta", "r", &table_capacity);
    while (df.ReadEntry(&k, &v)) {
      update_str(k, v);
    }
  }
}

//Dummy stub
//void LocalTable::DecodeUpdates(TableCoder *in, DecodeIteratorBase *itbase) { return; }

void LocalTable::write_delta(const TableData& put) {
  for (int i = 0; i < put.kv_data_size(); ++i) {
    delta_file_->WriteEntry(put.kv_data(i).key(), put.kv_data(i).value());
  }
}

LocalTableCoder::LocalTableCoder(const string& f, const string &mode, int64_t *capacity) :
  f_(new RecordFile(f, mode, RecordFile::LZO)) {

  Marshal<int64_t> m_int64;

  //Record or return table size
  string tablesize_s;
  if (mode == "w") {
    m_int64.marshal(*capacity,&tablesize_s);
    f_->writeChunk(StringPiece(tablesize_s));
  } else {
    string tablesize_s;
    f_->readChunk(&tablesize_s);
    m_int64.unmarshal(StringPiece(tablesize_s),capacity);
  }
    
}

LocalTableCoder::~LocalTableCoder() {
  delete f_;
}

bool LocalTableCoder::ReadEntry(string *k, string *v) {
  if (f_->readChunk(k)) {
    f_->readChunk(v);
    return true;
  }

  return false;
}

void LocalTableCoder::WriteEntry(StringPiece k, StringPiece v) {
  f_->writeChunk(k);
  f_->writeChunk(v);
}

enum BitMapPackMode {
  BITMAP_DENSE, BITMAP_SPARSE
};

void LocalTableCoder::WriteBitMap(boost::dynamic_bitset<uint32_t>* bitset, int64_t capacity) {
  Marshal<int>      m_int;
  Marshal<uint32_t> m_int32;
  Marshal<int64_t>  m_int64;

/*
  string tablesize_s;
  m_int64.marshal(capacity,&tablesize_s);
  f_->writeChunk(StringPiece(tablesize_s));
*/

  //Figure out what method to use
  string packmode_s;
  uint64_t saved = 0;
  uint64_t count = bitset->count();
  if (((sizeof(int)+sizeof(uint64_t))*bitset->count()) < bitset->size()/32) {
    // Sparse is worthwhile
    m_int.marshal(BITMAP_SPARSE,&packmode_s);
    f_->writeChunk(StringPiece(packmode_s));

    size_t bititer = bitset->find_first();
    while(bititer != bitset->npos) {
      string bitindex_s;
      m_int64.marshal((int64_t)bititer,&bitindex_s);
      f_->writeChunk(StringPiece(bitindex_s));
      bititer = bitset->find_next(bititer);
      saved++;
    }
  } else {
    // Dense is better
    m_int.marshal(BITMAP_DENSE,&packmode_s);
    f_->writeChunk(StringPiece(packmode_s));

    uint32_t fullbyte = 0;
    int64_t offset = 0;
    while(offset < bitset->size()) {
      fullbyte |= ((bitset->test(offset))<<(offset%32));
      offset++;
      if ((offset%32 == 0) || (offset >= bitset->size())) {
        string fullbyte_s;
        m_int32.marshal(fullbyte,&fullbyte_s);
        f_->writeChunk(StringPiece(fullbyte_s));
        fullbyte = 0;
      }
    }
    saved = offset;
  }
  VLOG(1) << "Saved " << saved << " bits from retrigger bitset containing " << count << " set and " << bitset->size() << " total bits.";
  return;
}

bool LocalTableCoder::ReadBitMap(boost::dynamic_bitset<uint32_t>* bitset, LocalTable* table) {
  Marshal<int>      m_int;
  Marshal<uint32_t> m_int32;
  Marshal<int64_t>  m_int64;

/*
  string tablesize_s;
  int64_t tablesize;
  f_->readChunk(&tablesize_s);
  m_int64.unmarshal(StringPiece(tablesize_s),&tablesize);
  if (tablesize <= 0) {
    LOG(ERROR) << "Restoration of table bitmap requested from table of size " << tablesize;
    return false;
  }
  VLOG(1) << "Resizing table to size " << tablesize << " to match bitset";
  table->resize(tablesize);		//important to make bitmaps match properly!
*/
  
  string packmode_s;
  int packmode;
  f_->readChunk(&packmode_s);
  m_int.unmarshal(StringPiece(packmode_s),&packmode);
  uint64_t restored = 0;
  if (packmode == BITMAP_SPARSE) {
    //Indices of set bits are packed into the string
    string bittoset_s;
    while(f_->readChunk(&bittoset_s)) {
      int64_t bittoset;
      m_int64.unmarshal(StringPiece(bittoset_s),&bittoset);
      bitset->set(bittoset);
    }
  } else if (packmode == BITMAP_DENSE) {
    //Bits are packed 32 per byte, unpack them into the bitset
    string fullbyte_s;
    uint32_t fullbyte;
    int64_t offset = 0;
    while(f_->readChunk(&fullbyte_s) && offset < bitset->size()) {
      m_int32.unmarshal(StringPiece(fullbyte_s),&fullbyte);
      for(int bit = 0; bit < 32 && offset < bitset->size(); bit++) {
        bitset->set(offset++,fullbyte&(1<<bit));
      }
    }
    restored = offset;
  } else {
    LOG(FATAL) << "Unknown BitMap packing mode!";
  }
  VLOG(1) << "Restored " << restored << " bits to retrigger bitset.";
  return true;
}

}	// namespace piccolo {
