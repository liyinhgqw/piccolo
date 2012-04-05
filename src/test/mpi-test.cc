#include <mpi.h>
#include <boost/thread.hpp>
#include <vector>
#include <string>
#include <stdio.h>

#include "util/common.h"
#include <gflags/gflags.h>

DEFINE_int32(requests, 1000, "");
DEFINE_int32(request_size, 10000, "");

using namespace std;
struct SendReq {
  MPI::Request req;
  string data;
};

list<SendReq*> reqs;

void send() {
  MPI::Intracomm world = MPI::COMM_WORLD;
  for (int i = 0; i < FLAGS_requests; ++i) {
    SendReq *r = new SendReq;
    r->data.assign(FLAGS_request_size, 'a');
    r->req = world.Isend(&r->data[0], r->data.size(), MPI::BYTE, i % world.Get_size(), 1);
    reqs.push_back(r);
  }
}

void test_send_done() {
  list<SendReq*>::iterator i = reqs.begin();
  while (i != reqs.end()) {
    if ((*i)->req.Test()) {
      delete (*i);
      i = reqs.erase(i);
    } else {
      ++i;
    }
  }
}

int receive() {
  MPI::Intracomm world = MPI::COMM_WORLD;
  MPI::Status status;

  int c = 0;
  while (world.Iprobe(rpc::ANY_SOURCE, 1, status)) {
    string s;
    s.resize(status.Get_count(MPI::BYTE));
    world.Recv(&s[0], status.Get_count(MPI::BYTE), MPI::BYTE,
               status.Get_source(), 1);
    ++c;
  }

  return c;
}

int main(int argc, char **argv) {
  piccolo::Init(argc, argv);

  for (int i = 0; i < 100; ++i) {
    send();
    int got = 0;
    while (got < FLAGS_requests && !reqs.empty()) {
      test_send_done();
      got += receive();

      PERIODIC(1, LOG(INFO) << "received... " << got);
    }

    fprintf(stderr, "Working... %d\n", i);
  }

  return 0;
}
