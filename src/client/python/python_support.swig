%module piccolo

#include "client/python/python_support.h"

using namespace std;

struct PyObject;

namespace google {
  namespace protobuf {
    typedef int int32;
    typedef long int64;
  }
}

%typemap(in) long& { $1 = PyInt_AsLong($input); }
%typemap(out) long& { $result = PyInt_FromLong(*$1); }

%typemap(in) long { $1 = PyInt_AsLong($input); }
%typemap(out) long { $result = PyInt_FromLong($1); }

%typemap(in) PyObject* { $1 = $input; Py_IncRef($1); }
%typemap(out) PyObject* { $result = $1; }

%typemap(in) PyObject*& { $1 = $input; Py_IncRef($1); }
%typemap(out) PyObject*& { $result = *$1; }

%typemap(in) const PyObject*& { $1 = $input; Py_IncRef($1); }
%typemap(out) const PyObject*& { $result = *$1; }


%typemap(in) (int argc, char **argv) {
  if (PyList_Check($input)) {
    int i;
    $1 = PyList_Size($input);
    $2 = (char **) malloc(($1+1)*sizeof(char *));
    for (i = 0; i < $1; i++) {
      PyObject *o = PyList_GetItem($input,i);
      if (PyString_Check(o))
        $2[i] = PyString_AsString(PyList_GetItem($input,i));
      else {
        PyErr_SetString(PyExc_TypeError,"list must contain strings");
        free($2);
        return NULL;
      }
    }
    $2[i] = 0;
  } else {
    PyErr_SetString(PyExc_TypeError,"not a list");
    return NULL;
  }
}

%typemap(freearg) (int argc, char **argv) {
  free((char *) $2);
}

namespace std {
  %typemap(in) string& (string temp) {
    if (PyString_Check($input)) {
      temp = PyString_AsString($input);
      $1 = &temp;
     } else{
       PyErr_SetString(PyExc_TypeError,"Expected a string type!");
       return NULL; 
     }
  }
  
  %typemap(in) string (string temp) {
    if (PyString_Check($input)) {
      $1 = PyString_AsString($input);
     } else{
       PyErr_SetString(PyExc_TypeError,"Expected a string type!");
       return NULL; 
     }
  }

  %typemap(out) string { 
    $result = PyString_FromString(($1).c_str());
  }
  
  %typemap(out) string& { 
    $result = PyString_FromString((*$1).c_str());
  }
}

%{
#include "client/python/python_support.h"
#include "kernel/table.h"

namespace dsm {
typedef TypedGlobalTable<string, int64_t> IntTable;
typedef TypedGlobalTable<string, string> StringTable;

typedef TypedTableIterator<string, int64_t> IntIterator;
typedef TypedTableIterator<string, string> StringIterator;

//typedef Trigger<string, int64_t> IntTrigger;
//typedef Trigger<string, string> StringTrigger;

typedef PythonTrigger<string, int64_t> IntPythonTrigger;
typedef PythonTrigger<string, string> StringPythonTrigger;

}%}

#define GOOGLE_PROTOBUF_VERSION 2003000
#define LIBPROTOBUF_EXPORT

typedef long int64_t;

%include "util/rpc.h"

%include "kernel/kernel.h"
%include "kernel/table-registry.h"

%include "master/master.h"
%include "client/python/python_support.h"

typedef int TriggerID;

void Init(int argc, char** argv);

%inline %{
  namespace dsm {
  struct IntAccum { 
    static Accumulator<int64_t>* Min() { return new dsm::Accumulators<int64_t>::Min; }
    static Accumulator<int64_t>* Max() { return new dsm::Accumulators<int64_t>::Max; }
    static Accumulator<int64_t>* Sum() { return new dsm::Accumulators<int64_t>::Sum; }
    static Accumulator<int64_t>* Replace() { return new dsm::Accumulators<int64_t>::Replace; }
  };
  
  struct StringAccum { 
    static Accumulator<string>* Min() { return new dsm::Accumulators<string>::Min; }
    static Accumulator<string>* Max() { return new dsm::Accumulators<string>::Max; }
    static Accumulator<string>* Sum() { return new dsm::Accumulators<string>::Sum; }
    static Accumulator<string>* Replace() { return new dsm::Accumulators<string>::Replace; }
  };

  struct IntTrigger {
    static Trigger<string,int64_t>* Null() { return new dsm::Triggers<string,int64_t>::NullTrigger; }
    static Trigger<string,int64_t>* ReadOnly() { return new dsm::Triggers<string,int64_t>::ReadOnlyTrigger; }
    static Trigger<string,int64_t>* PythonCode(const string& codeshort, const string& codelong) { return new IntPythonTrigger(0,codeshort,codelong); }
  };

  struct StringTrigger {
    static Trigger<string,string>* Null() { return new dsm::Triggers<string,string>::NullTrigger; }
    static Trigger<string,string>* ReadOnly() { return new dsm::Triggers<string,string>::ReadOnlyTrigger; }
    static Trigger<string,string>* PythonCode(const string& codeshort, const string& codelong) { return new StringPythonTrigger(0,codeshort,codelong); }
  };
  
  static Sharding::String* StringSharding() { return new Sharding::String; }
  static Sharding::Mod* ModSharding() { return new Sharding::Mod; }
  static Sharding::UintMod* UintModSharding() { return new Sharding::UintMod; }
  
  struct SwigDomainSharding : public Sharder<string> {
    int operator()(const string& in, int num_shards) {
      int d_end = in.find(" ");
      return SuperFastHash(in.data(), d_end) % num_shards;
    }
  };
  
  static Sharder<string>* DomainSharding() { return new SwigDomainSharding; }
}
%};

namespace dsm {
class ConfigData {
public:
  ConfigData();
  virtual ~ConfigData();
  ConfigData(const ConfigData& from);
  ConfigData& operator=(const ConfigData& from);
  
  int num_workers() const;
  void set_num_workers(int value);

  int worker_id() const;
  void set_worker_id(int value);

  int master_id() const;
  void set_master_id(int value);
};

class RecordFile {
public:
  enum CompressionType {
    NONE = 0,
    LZO = 1
  };

  RecordFile(const string& path, const string& mode, int compression=NONE);
  ~RecordFile();

  void write(const google::protobuf::Message &m);
  bool read(google::protobuf::Message *m);
  const char* name();

  bool eof();
  void sync();
};

bool StartWorker(const ConfigData& conf);

template <class K, class V>
struct TypedTableIterator {
private:
  TypedIterator();
public:
  void key_str(string *out);
  void value_str(string *out);
  bool done();
  void Next();

  const K& key();
  V& value();
}; 

template <class K, class V>
struct Trigger<K, V> {
private:
  Trigger();
public:
  Trigger(GlobalTable* thistable);
  Init(GlobalTable* thistable);
  bool Fire(const K& k, const V& current, V& update, bool isNew);
};

template <class K, class V>
struct PythonTrigger<K, V> {
public:
  PythonTrigger(GlobalTable* thistable, const string& code);
  void Init(GlobalTable* thistable);
  bool Fire(const K& k, const V& current, V& update, bool isNew);
  bool CallPythonTrigger(PyObjectPtr callable, PyObjectPtr key, const V& current, V& update);
};

template <class K, class V> 
struct TypedGlobalTable<K, V>  {
private:
  TypedGlobalTable();
public:
  
  V get(const K& k);
  bool contains(const K& k);
  void update (const K& k, const V&);
  void enqueue_update(const K& k, const V&);
  void put(const K& k, const V&);
  
  void enable_retrigger(K k);

  TypedTableIterator<K, V>* get_iterator(int shard);
  
  void SendUpdates();
  void HandlePutRequests();

  %extend {
    // Register a trigger
    void py_swap_accumulator(const string& codeshort, const string& codelong) {
//      PythonTrigger<K, V>* trigger = new PythonTrigger<K, V>($self, codeshort, codelong);
		LOG(FATAL) << "Python may not yet swap accumulators" << endl;
        return;
    }
  }
};

%template(IntTable) TypedGlobalTable<string, int64_t>;
%template(StringTable) TypedGlobalTable<string, string>;

/*
%template(IntTrigger) Trigger<string, int64_t>;
%template(StringTrigger) Trigger<string, string>;
%template(IntPythonTrigger) PythonTrigger<string, int64_t>;
%template(StringPythonTrigger) PythonTrigger<string, string>;
*/

%template(IntIterator) TypedTableIterator<string, int64_t>;
%template(StringIterator) TypedTableIterator<string, string>;

%extend DSMKernel {
  TypedGlobalTable<string, string>* GetStringTable(int idx) { return $self->get_table<string, string>(idx); }
  TypedGlobalTable<string, int64_t>* GetIntTable(int idx) { return $self->get_table<string, int64_t>(idx); }

//  PythonTrigger<string, string>* GetStringPythonTrigger(int tidx, int idx) {
//    return (PythonTrigger<string, string>*)($self->get_table<string, string>(tidx)->trigger(idx));
//  }
//  PythonTrigger<string, int64_t>* GetIntPythonTrigger(int tidx, int idx) {
//    return (PythonTrigger<string, int64_t>*)($self->get_table<string, int64_t>(tidx)->trigger(idx));
//  }
}

%template(CreateStringTable) CreateTable<string, string>;
%template(CreateIntTable) CreateTable<string, int64_t>;

%extend Master {
  void py_run_all(const string& code, void *locality) {
    RunDescriptor r("PythonKernel", "run_python_code", (GlobalTable*)locality);
    r.params.put<string>("python_code", code);
    r.shards = range(((GlobalTable*)locality)->num_shards());
    $self->run(r);
  }

  // Run the given kernel function on one (arbitrary) worker node.
  void py_run_one(const string& code, void*locality) {
    RunDescriptor r("PythonKernel", "run_python_code", (GlobalTable*)locality);
    r.params.put<string>("python_code", code);
    r.shards = range(1);
    $self->run(r);
  }

}

}
