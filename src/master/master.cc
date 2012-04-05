#include "kernel/global-table.h"
#include "kernel/local-table.h"
#include "kernel/table.h"
#include "master/master.h"
#include "util/static-initializers.h"

#include <set>

using std::map;
using std::vector;
using std::set;

DEFINE_string(dead_workers, "", "For failure testing; comma delimited list of workers to pretend have died.");
DEFINE_bool(work_stealing, true, "Enable work stealing to load-balance tasks between machines.");
DEFINE_bool(checkpoint, false, "If true, enable checkpointing.");
DEFINE_bool(restore, false, "If true, enable restore.");
DEFINE_int32(mintime_ccp_delta, 10, "Minimum seconds for each continuous checkpoint delta to last");
DEFINE_int32(mintime_ccp_full, 100, "Minimum interval for each continuous checkpoint full snapshot");

DECLARE_string(checkpoint_write_dir);
DECLARE_string(checkpoint_read_dir);
DECLARE_double(sleep_time);

namespace piccolo {

static std::set<int> dead_workers;

struct Taskid {
  int table;
  int shard;

  Taskid(int t, int s) : table(t), shard(s) {}

  bool operator<(const Taskid& b) const {
    return table < b.table || (table == b.table && shard < b.shard);
  }
};

struct TaskState : private boost::noncopyable {
  enum Status {
    PENDING  = 0,
    ACTIVE   = 1,
    FINISHED  = 2
  };

  TaskState(Taskid id, int64_t size)
    : id(id), status(PENDING), size(size), stolen(false) {}

  static bool IdCompare(TaskState *a, TaskState *b) {
    return a->id < b->id;
  }

  static bool WeightCompare(TaskState *a, TaskState *b) {
    if (a->stolen && !b->stolen) {
      return true;
    }
    return a->size < b->size;
  }

  Taskid id;
  int status;
  int size;
  bool stolen;
};

typedef map<Taskid, TaskState*> TaskMap;
typedef std::set<Taskid> ShardSet;
struct WorkerState : private boost::noncopyable {
  WorkerState(int w_id) : id(w_id) {
    last_ping_time = Now();
    last_task_start = 0;
    total_runtime = 0;
    checkpointing = false;
  }

  TaskMap work;

  // Table shards this worker is responsible for serving.
  ShardSet shards;

  double last_ping_time;

  int status;
  int id;

  double last_task_start;
  double total_runtime;

  bool checkpointing;

  // Order by number of pending tasks and last update time.
  static bool PendingCompare(WorkerState *a, WorkerState* b) {
//    return (a->pending_size() < b->pending_size());
    return a->num_pending() < b->num_pending();
  }

  bool alive() const {
    return dead_workers.find(id) == dead_workers.end();
  }

  bool is_assigned(Taskid id) {
    return work.find(id) != work.end();
  }

  void ping() {
    last_ping_time = Now();
  }

  double idle_time() {
    // Wait a little while before stealing work; should really be
    // using something like the standard deviation, but this works
    // for now.
    if (num_finished() != work.size())
      return 0;

    return Now() - last_ping_time;
  }

  void assign_shard(int shard, bool should_service) {
    TableRegistry::Map &tables = TableRegistry::Get()->tables();
    for (TableRegistry::Map::iterator i = tables.begin(); i != tables.end(); ++i) {
      if (shard < i->second->num_shards()) {
        Taskid t(i->first, shard);
        if (should_service) {
          shards.insert(t);
        } else {
          shards.erase(shards.find(t));
        }
      }
    }
  }

  bool serves(Taskid id) const {
    return shards.find(id) != shards.end();
  }

  void assign_task(TaskState *s) {
    work[s->id] = s;
  }

  void remove_task(TaskState* s) {
    work.erase(work.find(s->id));
  }

  void clear_tasks() {
    work.clear();
  }

  void set_finished(const Taskid& id) {
    CHECK(work.find(id) != work.end());
    TaskState *t = work[id];
    CHECK(t->status == TaskState::ACTIVE);
    t->status = TaskState::FINISHED;
  }

#define COUNT_TASKS(name, type)\
  size_t num_ ## name() const {\
    int c = 0;\
    for (TaskMap::const_iterator i = work.begin(); i != work.end(); ++i)\
      if (i->second->status == TaskState::type) { ++c; }\
    return c;\
  }\
  int64_t name ## _size() const {\
      int64_t c = 0;\
      for (TaskMap::const_iterator i = work.begin(); i != work.end(); ++i)\
        if (i->second->status == TaskState::type) { c += i->second->size; }\
      return c;\
  }\
  vector<TaskState*> name() const {\
    vector<TaskState*> out;\
    for (TaskMap::const_iterator i = work.begin(); i != work.end(); ++i)\
      if (i->second->status == TaskState::type) { out.push_back(i->second); }\
    return out;\
  }

  COUNT_TASKS(pending, PENDING)
  COUNT_TASKS(active, ACTIVE)
  COUNT_TASKS(finished, FINISHED)
#undef COUNT_TASKS

  int num_assigned() const { return work.size(); }
  int64_t total_size() const {
    int64_t out = 0;
    for (TaskMap::const_iterator i = work.begin(); i != work.end(); ++i) {
      out += 1 + i->second->size;
    }
    return out;
  }

  // Order pending tasks by our guess of how large they are
  bool get_next(const RunDescriptor& r, KernelRequest* msg) {
    vector<TaskState*> p = pending();

    if (p.empty()) {
      return false;
    }

    TaskState* best = *max_element(p.begin(), p.end(), &TaskState::WeightCompare);

    msg->set_kernel(r.kernel);
    msg->set_method(r.method);
    msg->set_table(r.table->id());
    msg->set_shard(best->id.shard);

    best->status = TaskState::ACTIVE;
    last_task_start = Now();

    return true;
  }
};

Master::Master(const ConfigData &conf) :
  tables_(TableRegistry::Get()->tables()){
  config_.CopyFrom(conf);
  kernel_epoch_ = 0;
  finished_ = dispatched_ = 0;

  //Checkpointing
  last_checkpoint_ = Now();
  prev_ccp_full_ = Now();
  checkpoint_epoch_ = 0;
  checkpointing_ = false;

  network_ = rpc::NetworkThread::Get();
  shards_assigned_ = false;

  CHECK_GT(network_->size(), 1) << "At least one master and one worker required!";

  for (int i = 0; i < config_.num_workers(); ++i) {
    workers_.push_back(new WorkerState(i));
  }

  for (int i = 0; i < config_.num_workers(); ++i) {
    RegisterWorkerRequest req;
    int src = 0;
    network_->Read(rpc::ANY_SOURCE, MTYPE_REGISTER_WORKER, &req, &src);
    VLOG(1) << "Registered worker " << src - 1 << "; " << config_.num_workers() - 1 - i << " remaining.";
  }

  LOG(INFO) << "All workers registered; starting up.";

  vector<StringPiece> bits = StringPiece::split(FLAGS_dead_workers, ",");
//  LOG(INFO) << "dead workers: " << FLAGS_dead_workers;
  for (size_t i = 0; i < bits.size(); ++i) {
    LOG(INFO) << MP(i, bits[i].AsString());
    dead_workers.insert(strtod(bits[i].AsString().c_str(), NULL));
  }
}

Master::~Master() {
  LOG(INFO) << "Total runtime: " << runtime_.elapsed();

  LOG(INFO) << "Worker execution time:";
  for (size_t i = 0; i < workers_.size(); ++i) {
    WorkerState& w = *workers_[i];
    if (i % 10 == 0) {
      fprintf(stderr, "\n%zu: ", i);
    }
    fprintf(stderr, "%.3f ", w.total_runtime);
  }
  fprintf(stderr, "\n");

  LOG(INFO) << "Kernel stats: ";
  for (MethodStatsMap::iterator i = method_stats_.begin(); i != method_stats_.end(); ++i) {
     LOG(INFO) << i->first << "--> " << i->second.ShortDebugString();
  }

  LOG(INFO) << "Shutting down workers.";
  EmptyMessage msg;
  for (int i = 1; i < network_->size(); ++i) {
    network_->Send(i, MTYPE_WORKER_SHUTDOWN, msg);
  }
}

void Master::start_checkpoint() {
  if (checkpointing_) {
    return;
  }

  Timer cp_timer;

  checkpoint_epoch_ += 1;
  checkpointing_ = true;

  LOG(INFO) << "Starting new checkpoint: epoch" << checkpoint_epoch_;

  File::Mkdirs(StringPrintf("%s/epoch_%05d/",
                            FLAGS_checkpoint_write_dir.c_str(), checkpoint_epoch_));

  if (current_run_.checkpoint_type == CP_NONE) {
    current_run_.checkpoint_type = CP_TASK_COMMIT;
  }

  for (size_t i = 0; i < workers_.size(); ++i) {
    start_worker_checkpoint(i, current_run_);
  }
  int finished = 0;
  EmptyMessage resp;
  int worker_id = 0;
  while(finished < workers_.size()) {
    VLOG(3) << "Waiting for checkpoint started responses (" << finished << " received)" << endl;
    if (network_->TryRead(rpc::ANY_SOURCE, MTYPE_START_CHECKPOINT_DONE, &resp, &worker_id)) {
      finished++;
      VLOG(2) << "Received checkpoint start done " << finished << " of " << workers_.size() << endl;

      workers_[worker_id-1]->checkpointing = true;//(current_run_.checkpoint_type == CP_CONTINUOUS || 
                                              //current_run_.checkpoint_type == CP_INTERVAL);
    } else {
      Sleep(FLAGS_sleep_time);
    }
  }
  if (current_run_.checkpoint_type == CP_CONTINUOUS)
    start_deltacheckpoint_ = Now();
}

void Master::start_worker_checkpoint(int worker_id, const RunDescriptor &r) {
  start_checkpoint();

  if (workers_[worker_id]->checkpointing) {
    return;
  }

  VLOG(1) << "Starting checkpoint on worker: " << worker_id;

  workers_[worker_id]->checkpointing = true;

  CheckpointRequest req;
  req.set_epoch(checkpoint_epoch_);
  req.set_checkpoint_type(r.checkpoint_type);

  for (size_t i = 0; i < r.checkpoint_tables.size(); ++i) {
    req.add_table(r.checkpoint_tables[i]);
  }

  if (r.checkpoint_type == CP_CONTINUOUS) {
    network_->Send(1 + worker_id, MTYPE_START_CHECKPOINT_ASYNC, req);
  } else {
    network_->Send(1 + worker_id, MTYPE_START_CHECKPOINT, req);
  }
}

void Master::finish_worker_checkpoint(int worker_id, const RunDescriptor& r, bool deltaOnly) {
  CHECK_EQ(workers_[worker_id]->checkpointing, true);
  VLOG(1) << " Instructing worker " << worker_id << " to finish checkpoint.";

  if (r.checkpoint_type == CP_TASK_COMMIT || r.checkpoint_type == CP_CONTINUOUS) {
    CheckpointFinishRequest req;
    req.set_next_delta_only(deltaOnly);
    network_->Send(1 + worker_id, (r.checkpoint_type == CP_CONTINUOUS)?MTYPE_FINISH_CHECKPOINT_ASYNC:MTYPE_FINISH_CHECKPOINT, req);
  }
}

//write the checkpoint.finished file and its contents
void Master::finish_checkpoint_writefile(int epoch) {
  Args *params = current_run_.params.ToMessage();
  Args *cp_vars = cp_vars_.ToMessage();

  RecordFile rf(StringPrintf("%s/epoch_%05d/checkpoint.finished",
                            FLAGS_checkpoint_write_dir.c_str(), epoch), "w");

  CheckpointInfo cinfo;
  cinfo.set_checkpoint_epoch(checkpoint_epoch_);
  cinfo.set_kernel_epoch(kernel_epoch_);

  rf.write(cinfo);
  rf.write(*params);
  rf.write(*cp_vars);
  rf.sync();

  delete params;
  delete cp_vars;
}

void Master::finish_checkpoint() {
  //if continuous and we've recently done a full, just do deltas
  bool deltaOnly = ((current_run_.checkpoint_type == CP_CONTINUOUS) && (!(prev_ccp_full_ + FLAGS_mintime_ccp_full < Now())));
  for (size_t i = 0; i < workers_.size(); ++i) {
    finish_worker_checkpoint(i, current_run_, deltaOnly);
  }

  int finished = 0;
  EmptyMessage resp;
  int worker_id;
  while(finished < workers_.size()) {
    VLOG(3) << "Waiting for checkpoint finished responses (" << finished << " received)" << endl;
    if (network_->TryRead(rpc::ANY_SOURCE, MTYPE_FINISH_CHECKPOINT_DONE, &resp, &worker_id)) {
      finished++;
      VLOG(2) << "Received checkpoint finish done " << finished << " of " << workers_.size() << endl;

      if (current_run_.checkpoint_type != CP_CONTINUOUS)
        workers_[worker_id-1]->checkpointing = false;
    } else {
      Sleep(FLAGS_sleep_time);
    }
  }

  for (size_t i = 0; i < workers_.size(); ++i) {
    if (current_run_.checkpoint_type != CP_CONTINUOUS) {
      CHECK_EQ(workers_[i]->checkpointing, false);
    }
  }

  //if continuous, assume we're at least starting a new set of deltas,
  //and maybe a full snapshot too.
  if (current_run_.checkpoint_type == CP_CONTINUOUS) {
    LOG(INFO) << "Beginning new checkpoint epoch with deltaOnly=" << deltaOnly;
    start_deltacheckpoint_ = Now();
    //if this was a full snapshot, then remember that.
    if (!deltaOnly) {
      prev_ccp_full_ = Now();
    }
  }

  LOG(INFO) << "All workers finished checkpointing.  Flushing epoch " << checkpoint_epoch_ << ".";
  finish_checkpoint_writefile(checkpoint_epoch_);

  //if continuous, then finishing a checkpoint is the implicit
  //start of a new checkpoint, but we don't want to send out
  //separate start messages.
  if (current_run_.checkpoint_type == CP_CONTINUOUS) {
    checkpoint_epoch_ += 1;
    checkpointing_ = true;

    LOG(INFO) << "Starting new checkpoint: " << checkpoint_epoch_;

    File::Mkdirs(StringPrintf("%s/epoch_%05d/",
                              FLAGS_checkpoint_write_dir.c_str(), checkpoint_epoch_));
  } else {
    checkpointing_ = false;
    current_run_.checkpoint_type = CP_NONE;
  }
  last_checkpoint_ = Now();
}

void Master::checkpoint() {
  start_checkpoint();
  finish_checkpoint();
}

bool Master::restore() {
  if (!FLAGS_restore) {
    LOG(INFO) << "Restore disabled by flag.";
    return false;
  }

  if (!shards_assigned_) {
    assign_tables();
    send_table_assignments();
  }

  Timer t;
  vector<string> matches = File::MatchingFilenames(FLAGS_checkpoint_read_dir + "/*/checkpoint.finished");
  if (matches.empty()) {
    return false;
  }

  // Glob returns results in sorted order, so our last checkpoint will be the last.
  // CRM 2011-10-13: Cycle backwards looking for full checkpoints for continuous, since
  // zero or more last may be deltas.
  int epoch = -1;
  int lastepoch = -1; //the last finished epoch
  string topfinishfile = ""; //where to restore vars and params from
  
  const char* fname = matches.back().c_str();
  CHECK_EQ(sscanf(fname, (FLAGS_checkpoint_read_dir + "/epoch_%05d/checkpoint.finished").c_str(), &epoch),
         1) << "Unexpected filename: " << fname;

  //Remember the highest finished checkpoint
  lastepoch = (epoch>lastepoch)?epoch:lastepoch;
  if (topfinishfile == "") topfinishfile = matches.back();

  CHECK_EQ(matches.empty(),false) << "Ran out of non-delta-only checkpoints to start from!";

  LOG(INFO) << "Restoring master state from file: " << topfinishfile << ", workers from epoch " << epoch << ".";

  RecordFile rf(topfinishfile, "r");
  CheckpointInfo info;
  Args checkpoint_vars;
  Args params;
  CHECK(rf.read(&info));
  CHECK(rf.read(&params));
  CHECK(rf.read(&checkpoint_vars));

  // XXX - RJP need to figure out how to properly handle rolling checkpoints.
  current_run_.params.FromMessage(params);

  cp_vars_.FromMessage(checkpoint_vars);


  LOG(INFO) << "Restored master state from checkpoint " << MP(info.kernel_epoch(), info.checkpoint_epoch());
  LOG(INFO) << "Restoring worker state from epoch " << epoch << " forward.";

  // The next checkpoint should be committed to a separate epoch from the
  // restored checkpoint.
  kernel_epoch_ = info.kernel_epoch() + 1;
  checkpoint_epoch_ = info.checkpoint_epoch() + 1;

  StartRestore req;
  req.set_epoch(epoch);
  network_->SyncBroadcast(MTYPE_RESTORE, req);

  LOG(INFO) << "Checkpoint restored in " << t.elapsed() << " seconds.";
  return true;
}

void Master::run_all(RunDescriptor r) {
  CHECK_NE(r.table, (void*)NULL) << "Must specify a table for locality.";
  run_range(r, range(r.table->num_shards()));
}

void Master::run_one(RunDescriptor r) {
  run_range(r, range(1));
}

void Master::run_range(RunDescriptor r, vector<int> shards) {
  r.shards = shards;
  run(r);
}

WorkerState* Master::worker_for_shard(int table, int shard) {
  for (size_t i = 0; i < workers_.size(); ++i) {
    if (workers_[i]->serves(Taskid(table, shard))) { return workers_[i]; }
  }

  return NULL;
}

WorkerState* Master::assign_worker(int table, int shard) {
  WorkerState* ws = worker_for_shard(table, shard);
  int64_t work_size = tables_[table]->shard_size(shard);

  if (ws) {
//    LOG(INFO) << "Worker for shard: " << MP(table, shard, ws->id);
    ws->assign_task(new TaskState(Taskid(table, shard), work_size));
    return ws;
  }

  WorkerState* best = NULL;
  for (size_t i = 0; i < workers_.size(); ++i) {
    WorkerState& w = *workers_[i];
    if (w.alive() && (best == NULL || w.shards.size() < best->shards.size())) {
      best = workers_[i];
    }
  }

  CHECK(best != NULL) << "Ran out of workers!  Increase the number of partitions per worker!";

//  LOG(INFO) << "Assigned " << MP(table, shard, best->id);
  CHECK(best->alive());

  VLOG(1) << "Assigning " << MP(table, shard) << " to " << best->id;
  best->assign_shard(shard, true);
  best->assign_task(new TaskState(Taskid(table, shard), work_size));
  return best;
}

void Master::send_table_assignments() {
  ShardAssignmentRequest req;

  for (size_t i = 0; i < workers_.size(); ++i) {
    WorkerState& w = *workers_[i];
    for (ShardSet::iterator j = w.shards.begin(); j != w.shards.end(); ++j) {
      ShardAssignment* s  = req.add_assign();
      s->set_new_worker(i);
      s->set_table(j->table);
      s->set_shard(j->shard);
//      s->set_old_worker(-1);
    }
  }

  network_->SyncBroadcast(MTYPE_SHARD_ASSIGNMENT, req);
}

bool Master::steal_work(const RunDescriptor& r, int idle_worker,
                        double avg_completion_time) {
  if (!FLAGS_work_stealing) {
    return false;
  }

  WorkerState &dst = *workers_[idle_worker];

  if (!dst.alive()) {
    return false;
  }

  // Find the worker with the largest number of pending tasks.
  WorkerState& src = **max_element(workers_.begin(), workers_.end(), &WorkerState::PendingCompare);
  if (src.num_pending() == 0) {
    return false;
  }

  vector<TaskState*> pending = src.pending();

  TaskState *task = *max_element(pending.begin(), pending.end(), TaskState::WeightCompare);
  if (task->stolen) {
    return false;
  }

  double average_size = 0;

  for (int i = 0; i < r.table->num_shards(); ++i) {
    average_size += r.table->shard_size(i);
  }
  average_size /= r.table->num_shards();

  // Weight the cost of moving the table versus the time savings.
  double move_cost = std::max(1.0,
                         2 * task->size * avg_completion_time / average_size);
  double eta = 0;
  for (size_t i = 0; i < pending.size(); ++i) {
    TaskState *p = pending[i];
    eta += std::max(1.0, p->size * avg_completion_time / average_size);
  }

//  LOG(INFO) << "ETA: " << eta << " move cost: " << move_cost;

  if (eta <= move_cost) {
    return false;
  }

  const Taskid& tid = task->id;
  task->stolen = true;

  LOG(INFO) << "Worker " << idle_worker << " is stealing task "
            << MP(tid.shard, task->size) << " from worker " << src.id;
  dst.assign_shard(tid.shard, true);
  src.assign_shard(tid.shard, false);

  src.remove_task(task);
  dst.assign_task(task);
  return true;
}

void Master::assign_tables() {
  shards_assigned_ = true;

  // Assign workers for all table shards, to ensure every shard has an owner.
  TableRegistry::Map &tables = TableRegistry::Get()->tables();
  for (TableRegistry::Map::iterator i = tables.begin(); i != tables.end(); ++i) {
    if (!i->second->num_shards()) {
	  VLOG(2) << "Note: assigning tables; table " << i->first << " has no shards.";
    }
    for (int j = 0; j < i->second->num_shards(); ++j) {
      assign_worker(i->first, j);
    }
  }
}

void Master::assign_tasks(const RunDescriptor& r, vector<int> shards) {
  for (size_t i = 0; i < workers_.size(); ++i) {
    WorkerState& w = *workers_[i];
    w.clear_tasks(); //XXX: did not delete task state, memory leak
  }

  for (size_t i = 0; i < shards.size(); ++i) {
    assign_worker(r.table->id(), shards[i]);
  }
}

int Master::dispatch_work(const RunDescriptor& r) {
  int num_dispatched = 0;
  KernelRequest w_req;
  for (size_t i = 0; i < workers_.size(); ++i) {
    WorkerState& w = *workers_[i];
    if (w.num_pending() > 0 && w.num_active() == 0) {
      w.get_next(r, &w_req);
      Args* p = r.params.ToMessage();
      w_req.mutable_args()->CopyFrom(*p);
      delete p;
      num_dispatched++;
      network_->Send(w.id + 1, MTYPE_RUN_KERNEL, w_req);
    }
  }
  return num_dispatched;
}

void Master::dump_stats() {
  string status;
  for (int k = 0; k < config_.num_workers(); ++k) {
    status += StringPrintf("%d/%d ",
        workers_[k]->num_finished(),
        workers_[k]->num_assigned());
  }
  LOG(INFO) << StringPrintf("Running %s (%d); %s; assigned: %d done: %d",
                            current_run_.method.c_str(), current_run_.shards.size(),
                            status.c_str(), dispatched_, finished_);

}

int Master::reap_one_task() {
  MethodStats &mstats = method_stats_[current_run_.kernel + ":" + current_run_.method];
  KernelDone done_msg;
  int w_id = 0;

  if (network_->TryRead(rpc::ANY_SOURCE, MTYPE_KERNEL_DONE, &done_msg, &w_id)) {

    w_id -= 1;

    WorkerState& w = *workers_[w_id];

    Taskid task_id(done_msg.kernel().table(), done_msg.kernel().shard());
    //      TaskState* task = w.work[task_id];
    //
    //      LOG(INFO) << "TASK_FINISHED "
    //                << r.method << " "
    //                << task_id.table << " " << task_id.shard << " on "
    //                << w_id << " in "
    //                << Now() - w.last_task_start << " size "
    //                << task->size <<
    //                " worker " << w.total_size();

    for (int i = 0; i < done_msg.shards_size(); ++i) {
      const ShardInfo &si = done_msg.shards(i);
      tables_[si.table()]->UpdatePartitions(si);
    }

    w.set_finished(task_id);

    w.total_runtime += Now() - w.last_task_start;
    mstats.set_shard_time(mstats.shard_time() + Now() - w.last_task_start);
    mstats.set_shard_calls(mstats.shard_calls() + 1);
    w.ping();
    return w_id;
  } else {
    Sleep(FLAGS_sleep_time);
    return -1;
  }

}

void Master::run(RunDescriptor r) {
  if (!FLAGS_checkpoint && r.checkpoint_type != CP_NONE) {
    LOG(INFO) << "Checkpoint is disabled by flag.";
    r.checkpoint_type = CP_NONE;
  }

  // HACKHACKHACK - register ourselves with any existing tables
  for (TableRegistry::Map::iterator i = tables_.begin(); i != tables_.end(); ++i) {
    i->second->set_helper(this);
  }

  CHECK_EQ(current_run_.shards.size(), finished_) << " Cannot start kernel before previous one is finished ";
  finished_ = dispatched_ = 0;

  KernelInfo *k = KernelRegistry::Get()->kernel(r.kernel);
  CHECK_NE(r.table, (void*)NULL) << "Table locality must be specified!";
  CHECK_NE(k, (void*)NULL) << "Invalid kernel class " << r.kernel;
  CHECK_EQ(k->has_method(r.method), true) << "Invalid method: " << MP(r.kernel, r.method);

  VLOG(1) << "Running: " << r.kernel << " : " << r.method << " : " << *r.params.ToMessage();

  vector<int> shards = r.shards;

  MethodStats &mstats = method_stats_[r.kernel + ":" + r.method];
  mstats.set_calls(mstats.calls() + 1);

  // Fill in the list of tables to checkpoint, if it was left empty.
  if (r.checkpoint_tables.empty()) {
    for (TableRegistry::Map::iterator i = tables_.begin(); i != tables_.end(); ++i) {
      r.checkpoint_tables.push_back(i->first);
    }
  }

  current_run_ = r;
  current_run_start_ = Now();

  if (!shards_assigned_) {
    //only perform table assignment before the first kernel run
    assign_tables();
    send_table_assignments();
  }

  kernel_epoch_++;

  VLOG(1) << "Current run: " << shards.size() << " shards";
  assign_tasks(current_run_, shards);

  dispatched_ = dispatch_work(current_run_);

  //XXX:in its current state, does not make sense not to call barrier at the end
//  if (r.barrier) {
  barrier();
//  }
}

void Master::cp_barrier() {
  current_run_.checkpoint_type = CP_TASK_COMMIT;
  barrier();
}

/*
void Master::enable_trigger(const TriggerID triggerid, int table, bool enable) {
  EnableTrigger trigreq;
  for (int i = 0; i < workers_.size(); ++i) {
    WorkerState& w = *workers_[i];
    trigreq.set_trigger_id(triggerid);
    trigreq.set_table(table);
	trigreq.set_enable(enable);
    network_->Send(w.id + 1, MTYPE_ENABLE_TRIGGER, trigreq);
  }

}
*/

void Master::barrier() {
  MethodStats &mstats = method_stats_[current_run_.kernel + ":" + current_run_.method];

  VLOG(3) << "Starting barrier() with finished_=" << finished_;

  if (current_run_.checkpoint_type == CP_CONTINUOUS && !checkpointing_) {
    start_checkpoint();
  } else if (current_run_.checkpoint_type == CP_CONTINUOUS) {
    VLOG(1) << "Already performing continuous checkpoint";
  }
  while (finished_ < current_run_.shards.size()) {
    PERIODIC(10, {
          DumpProfile();
          dump_stats();
        });

    if (current_run_.checkpoint_type == CP_INTERVAL &&
        Now() - last_checkpoint_ > current_run_.checkpoint_interval) {
      checkpoint();
    }

    if (reap_one_task() >= 0) {
      finished_++;

      PERIODIC(0.1, {
            double avg_completion_time =
            mstats.shard_time() / mstats.shard_calls();

            bool need_update = false;
            for (int i = 0; i < workers_.size(); ++i) {
              WorkerState& w = *workers_[i];

              // Don't try to steal tasks if the payoff is too small.
              if (mstats.shard_calls() > 10 &&
                  avg_completion_time > 0.2 &&
                  !checkpointing_ &&
                  w.idle_time() > 0.5) {
                if (steal_work(current_run_, w.id, avg_completion_time)) {
                  need_update = true;
                }
              }

              if (current_run_.checkpoint_type == CP_TASK_COMMIT &&
                  0.7 * current_run_.shards.size() < finished_ &&
                  w.idle_time() > 0 &&
                  !w.checkpointing) {
                start_worker_checkpoint(w.id, current_run_);
              }
            }

            if (need_update) {
              // Update the table assignments.
              send_table_assignments();
            }

          });

      if (dispatched_ < current_run_.shards.size()) {
        dispatched_ += dispatch_work(current_run_);
      }
    }

    if ((current_run_.checkpoint_type == CP_CONTINUOUS) && (Now() > start_deltacheckpoint_ + FLAGS_mintime_ccp_delta)) {
      finish_checkpoint();
    }

  }

  VLOG(3) << "All kernels finished in barrier() with finished_=" << finished_;
  VLOG(1) << "Kernels finished, in flush/apply phase";

  bool quiescent;
  EmptyMessage empty;
  int worker_id = 0;
  do {
    quiescent = true;


    //1st round-trip to make sure all workers have flushed everything
    network_->Broadcast(MTYPE_WORKER_FLUSH, empty);
    VLOG(2) << "Sent flush broadcast to workers" << endl;

    size_t flushed = 0;
    size_t applied = 0;
    FlushResponse done_msg;
    while (flushed < workers_.size()) {
	  VLOG(3) << "Waiting for flush responses (" << flushed << " received)" << endl;
      if (network_->TryRead(rpc::ANY_SOURCE,
                            MTYPE_FLUSH_RESPONSE,
                            &done_msg,
                            &worker_id)) {
        flushed++;
        if (done_msg.updatesdone() > 0) {
          quiescent = false;
        }

        VLOG(1) << "Received flush response " << flushed
                << " of " << workers_.size()
                << " with " << done_msg.updatesdone()
                << " updates done.";
      } else {
        Sleep(FLAGS_sleep_time);
      }
      barriertasks();

    }

    //2nd round-trip to make sure all workers have applied all updates
    //XXX: incorrect if MPI does not guarantee remote delivery
    network_->Broadcast(MTYPE_WORKER_APPLY, empty);
    VLOG(2) << "Sent apply broadcast to workers" << endl;

    //If we don't wait for Apply responses, then we will send more FLUSH messages
    //potentially even out of other with APPLY, which can easily cause deadlocks
    //on the sl mutex because of multiple simultaneous attempted HandlePutRequest()s

    EmptyMessage apply_msg;
    while (applied < workers_.size()) {
	  VLOG(3) << "Waiting for apply responses (" << applied << " received)" << endl;
      if (network_->TryRead(rpc::ANY_SOURCE, MTYPE_WORKER_APPLY_DONE, &apply_msg, &worker_id)) {
        applied++;
        VLOG(2) << "Received apply done " << applied << " of " << workers_.size();
      } else {
        Sleep(FLAGS_sleep_time);
      }
      barriertasks();
    }

    VLOG(1) << "Flush/apply loop: " << flushed << "/" << applied << "f/a, quiescent = " << quiescent;
  } while (!quiescent);

  // Finally, we can do some cleanup/finalization tasks.  For now, this only
  // includes turning off any pending long triggers.
  // Now it also turns all continuous checkpointing to interval so that it can
  // be stopped be the finish_checkpoint() below.
  network_->Broadcast(MTYPE_WORKER_FINALIZE, empty);
  VLOG(3) << "Sent finalize broadcast to workers";
  int finalized = 0;
  EmptyMessage finalize_msg;
  while (finalized < workers_.size()) {
    if (network_->TryRead(rpc::ANY_SOURCE, MTYPE_WORKER_FINALIZE_DONE, &finalize_msg, &worker_id)) {
      finalized++;
      VLOG(3) << "Received finalize done " << finalized << " of " << workers_.size();
    } else {
      Sleep(FLAGS_sleep_time);
    }
  }

  if (current_run_.checkpoint_type == CP_CONTINUOUS && checkpointing_) {
    VLOG(2) << "Forcing continuous checkpointing to stop";
    current_run_.checkpoint_type = CP_TASK_COMMIT; //don't let it trigger more cycles
    finish_checkpoint();
  } else if (current_run_.checkpoint_type == CP_TASK_COMMIT) {
    if (!checkpointing_) {
      start_checkpoint();
    }
    finish_checkpoint();
  }

  mstats.set_total_time(mstats.total_time() + Now() - current_run_start_);
  LOG(INFO) << "Kernel '" << current_run_.method << "' finished in " << Now() - current_run_start_;
}

void Master::barriertasks() {
  PERIODIC(10, {
    DumpProfile();
    dump_stats();
  });

  //need to keep checkpointing alive
  if (current_run_.checkpoint_type == CP_CONTINUOUS && Now() > start_deltacheckpoint_ + FLAGS_mintime_ccp_delta) {
	finish_checkpoint();
  }

}

static void TestTaskSort() {
  vector<TaskState*> t;
  for (int i = 0; i < 100; ++i) {
    t.push_back(new TaskState(Taskid(0, i), rand()));
  }

  sort(t.begin(), t.end(), &TaskState::WeightCompare);
  for (int i = 1; i < 100; ++i) {
    CHECK_LE(t[i-1]->size, t[i]->size);
  }
}

REGISTER_TEST(TaskSort, TestTaskSort());
}
