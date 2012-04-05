#include "kernel/global-table.h"

namespace piccolo {

void GlobalTableBase::UpdatePartitions(const ShardInfo& info) {
  partinfo_[info.shard()].sinfo.CopyFrom(info);
}

GlobalTableBase::~GlobalTableBase() {
  for (int i = 0; i < partitions_.size(); ++i) {
    delete partitions_[i];
    delete writebufs_[i];
    delete writebufcoders_[i];
  }
}

TableIterator* GlobalTableBase::get_iterator(int shard, unsigned int fetch_num) {
  return partitions_[shard]->get_iterator();
}

bool GlobalTableBase::is_local_shard(int shard) {
  if (!helper())
    return false;
  return owner(shard) == helper_id();
}

bool GlobalTableBase::is_local_key(const StringPiece &k) {
  return is_local_shard(shard_for_key_str(k));
}

void GlobalTableBase::Init(const TableDescriptor *info) {
  TableBase::Init(info);
  partitions_.resize(info->num_shards);
  partinfo_.resize(info->num_shards);
  writebufs_.resize(info->num_shards);
  writebufcoders_.resize(info->num_shards);
}

int64_t GlobalTableBase::shard_size(int shard) {
  if (is_local_shard(shard)) {
    return partitions_[shard]->size();
  } else {
    return partinfo_[shard].sinfo.entries();
  }
}

void MutableGlobalTableBase::resize(int64_t new_size) {
  for (int i = 0; i < partitions_.size(); ++i) {
    if (is_local_shard(i)) {
      partitions_[i]->resize(new_size / partitions_.size());
    }
  }
}

bool GlobalTableBase::get_remote(int shard, const StringPiece& k, string* v) {
  {
	VLOG(3) << "Entering get_remote";
    boost::recursive_mutex::scoped_lock sl(mutex());
	VLOG(3) << "Entering get_remote and locked";
    if (remote_cache_.find(k) != remote_cache_.end()) {
      CacheEntry& c = remote_cache_[k];
      if ((Now() - c.last_read_time) < info().max_stale_time) {
        *v = c.value;
        return true;
      } else {
        remote_cache_.erase(k);
      }
    }
  }

  HashGet req;
  TableData resp;

  req.set_key(k.AsString());
  req.set_table(info().table_id);
  req.set_shard(shard);

  if (!helper())
	LOG(FATAL) << "get_remote() failed: helper() undefined.";
  int peer = helper()->peer_for_shard(info().table_id, shard);

  DCHECK_GE(peer, 0);
  DCHECK_LT(peer, rpc::NetworkThread::Get()->size() - 1);

  VLOG(2) << "Sending get request to: " << MP(peer, shard);
  rpc::NetworkThread::Get()->Call(peer + 1, MTYPE_GET, req, &resp);

  if (resp.missing_key()) {
    return false;
  }

  *v = resp.kv_data(0).value();

  if (info().max_stale_time > 0) {
    boost::recursive_mutex::scoped_lock sl(mutex());
    CacheEntry c = { Now(), *v };
    remote_cache_[k] = c;
  }
  return true;
}

void MutableGlobalTableBase::swap(GlobalTable *b) {
  SwapTable req;

  req.set_table_a(this->id());
  req.set_table_b(b->id());
  VLOG(2) << StringPrintf("Sending swap request (%d <--> %d)", req.table_a(), req.table_b());

  rpc::NetworkThread::Get()->SyncBroadcast(MTYPE_SWAP_TABLE, req);
}

void MutableGlobalTableBase::clear() {
  ClearTable req;

  req.set_table(this->id());
  VLOG(2) << StringPrintf("Sending clear request (%d)", req.table());

  rpc::NetworkThread::Get()->SyncBroadcast(MTYPE_CLEAR_TABLE, req);
}


void MutableGlobalTableBase::start_checkpoint(const string& f, bool deltaOnly) {
  for (int i = 0; i < partitions_.size(); ++i) {
    LocalTable *t = partitions_[i];

    if (is_local_shard(i)) {
      t->start_checkpoint(f + StringPrintf(".%05d-of-%05d", i, partitions_.size()), deltaOnly);
    }
  }
}

void MutableGlobalTableBase::finish_checkpoint() {
  for (int i = 0; i < partitions_.size(); ++i) {
    LocalTable *t = partitions_[i];

    if (is_local_shard(i)) {
      t->finish_checkpoint();
    }
  }
}

void MutableGlobalTableBase::write_delta(const TableData& d) {
  if (!is_local_shard(d.shard())) {
    LOG_EVERY_N(INFO, 1000) << "Ignoring delta write for forwarded data";
    return;
  }

  partitions_[d.shard()]->write_delta(d);
}



void MutableGlobalTableBase::restore(const string& f) {
  for (int i = 0; i < partitions_.size(); ++i) {
    LocalTable *t = partitions_[i];

    if (is_local_shard(i)) {
      t->restore(f + StringPrintf(".%05d-of-%05d", i, partitions_.size()));
    } else {
      t->clear();
    }
  }
}

void GlobalTableBase::handle_get(const HashGet& get_req, TableData *get_resp) {
  boost::recursive_mutex::scoped_lock sl(mutex());

  int shard = get_req.shard();
  if (!is_local_shard(shard)) {
    LOG_EVERY_N(WARNING, 1000) << "Not local for shard: " << shard;
  }

  UntypedTable *t = dynamic_cast<UntypedTable*>(partitions_[shard]);
  if (!t->contains_str(get_req.key())) {
    get_resp->set_missing_key(true);
  } else {
    Arg *kv = get_resp->add_kv_data();
    kv->set_key(get_req.key());
    kv->set_value(t->get_str(get_req.key()));
  }
}

void MutableGlobalTableBase::HandlePutRequests() {

    helper()->HandlePutRequest();
}

ProtoTableCoder::ProtoTableCoder(const TableData *in) : read_pos_(0), t_(const_cast<TableData*>(in)) {}
ProtoTableCoder::~ProtoTableCoder() {}

bool ProtoTableCoder::ReadEntry(string *k, string *v) {
  if (read_pos_ < t_->kv_data_size()) {
    k->assign(t_->kv_data(read_pos_).key());
    v->assign(t_->kv_data(read_pos_).value());
    ++read_pos_;
    return true;
  }

  return false;
}

void ProtoTableCoder::WriteEntry(StringPiece k, StringPiece v) {
  Arg *a = t_->add_kv_data();
  a->set_key(k.data, k.len);
  a->set_value(v.data, v.len);
}

void ProtoTableCoder::WriteBitMap(boost::dynamic_bitset<uint32_t>*, int64_t capacity) {
  return;
}

bool ProtoTableCoder::ReadBitMap(boost::dynamic_bitset<uint32_t>*, LocalTable* table) {
  return true;
}


void MutableGlobalTableBase::SendUpdates() {
  int i;
  MutableGlobalTableBase::SendUpdates(&i);
  return;
}

void MutableGlobalTableBase::SendUpdates(int* count) {
  TableData put;
  boost::recursive_mutex::scoped_lock sl(mutex());
  for (int i = 0; i < partitions_.size(); ++i) {
    LocalTable *t = partitions_[i];
    ProtoTableCoder *ptc = writebufcoders_[i];

    if (!is_local_shard(i) && (
        get_partition_info(i)->dirty || !t->empty() || ptc->t_->kv_data_size() > 0)) {
      // Always send at least one chunk, to ensure that we clear taint on
      // tables we own.
      do {
        put.Clear();

        if (!t->empty()) {
          VLOG(3) << "Sending update from non-trigger table ";
          ProtoTableCoder c(&put);
          t->Serialize(&c);
          t->clear();
        } else {
          VLOG(3) << "Sending update from trigger table with " << ptc->t_->kv_data_size() << " pairs.";
          put.CopyFrom(*(ptc->t_));
          ptc->t_->Clear();
        }
        put.set_shard(i);
        put.set_source(helper()->id());
        put.set_table(id());
        put.set_epoch(helper()->epoch());

        put.set_done(true);

        VLOG(3) << "Sending update for " << MP(t->id(), t->shard()) << " to " << owner(i) << " size " << put.kv_data_size();

        *count += put.kv_data_size();
        rpc::NetworkThread::Get()->Send(owner(i) + 1, MTYPE_PUT_REQUEST, put);
      } while(!t->empty());

      VLOG(3) << "Done with update for " << MP(t->id(), t->shard());
      t->clear();
    }
  }

  pending_writes_ = 0;
}

int MutableGlobalTableBase::pending_write_bytes() {
  int64_t s = 0;
  for (int i = 0; i < partitions_.size(); ++i) {
    LocalTable *t = partitions_[i];
    if (!is_local_shard(i)) {
      s += t->size();
    }
  }

  return s;
}

void GlobalTableBase::get_local(const StringPiece &k, string* v) {
  int shard = shard_for_key_str(k);
  CHECK(is_local_shard(shard));

  UntypedTable *h = (UntypedTable*)partitions_[shard];
  v->assign(h->get_str(k));
}

void MutableGlobalTableBase::local_swap(GlobalTable *b) {
  CHECK(this != b);

  MutableGlobalTableBase *mb = dynamic_cast<MutableGlobalTableBase*>(b);
  std::swap(partinfo_, mb->partinfo_);
  std::swap(partitions_, mb->partitions_);
  std::swap(cache_, mb->cache_);
  std::swap(pending_writes_, mb->pending_writes_);
}
}
