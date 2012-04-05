#include "glog/logging.h"
#include "python_support.h"
#include <boost/python.hpp>

using namespace boost::python;
using namespace google::protobuf;
using namespace std;

DEFINE_double(crawler_runtime, -1, "Amount of time to run, in seconds.");
DEFINE_bool(crawler_triggers, false, "Use trigger-based crawler (t/f).");
static DSMKernel *the_kernel;

namespace piccolo {

DSMKernel* kernel() {
  return the_kernel;
}

double crawler_runtime() {
  return FLAGS_crawler_runtime;
}

bool crawler_triggers() {
  return FLAGS_crawler_triggers;
}

int PythonSharder::operator()(const string& k, int shards) {
  PyObjectPtr result = PyEval_CallFunction(c_, "(si)", k.c_str(), shards);
  if (PyErr_Occurred()) {
    PyErr_Print();
    exit(1);
  }

  long r = PyInt_AsLong(result);
//  Py_DecRef(result);
  return r;
}

void PythonAccumulate::Accumulate(PyObjectPtr* a, const PyObjectPtr& b) {
  PyObjectPtr result = PyEval_CallFunction(c_, "OO", *a, b);
  if (PyErr_Occurred()) {
    PyErr_Print();
    exit(1);
  }

//  Py_DecRef(*a);
  *a = result;
//  Py_DecRef(const_cast<PyObjectPtr>(b));
}

template<class K, class V>
PythonTrigger<K, V>::PythonTrigger(piccolo::GlobalTable* thistable, const string& codeshort, const string& codelong) {
  Init(thistable);
  params_.put("python_code_short", codeshort);
  params_.put("python_code_long", codelong);
  //trigid = thistable->register_trigger(this);
}

template<class K, class V>
void PythonTrigger<K, V>::Init(piccolo::GlobalTable* thistable) {
  try {
    object sys_module = import("sys");
    object sys_ns = sys_module.attr("__dict__");
    crawl_module_ = import("crawler");
    crawl_ns_ = crawl_module_.attr("__dict__");
  } catch (error_already_set& e) {
    PyErr_Print();
    exit(1);
  }
}

template<class K, class V>
bool PythonTrigger<K, V>::LongFire(const K k, bool lastrun) {
  string python_code = params_.get<string> ("python_code_long");
  PyObject *key, *callable;
  callable = PyObject_GetAttrString(crawl_module_.ptr(), python_code.c_str());
  key = PyString_FromString(k.c_str());

  // Make sure all the callfunctionobjarg arguments are fine
  if (key == NULL || callable == NULL) {
    LOG(ERROR) << "Failed to launch trigger " << python_code << "!";
    if (key == NULL) LOG(ERROR) << "[FAIL] key was null";
    if (callable == NULL) LOG(ERROR) << "[FAIL] callable was null";
    return true;
  }

  V dummyv;
  bool rv = PythonTrigger<K, V>::CallPythonTrigger(callable, key, &dummyv, dummyv, true, lastrun); //hijack "isNew" for "lastrun"
  LOG(INFO) << "returning " << (rv?"TRUE":"FALSE") << " from long trigger";
  return rv;
}

template<class K, class V>
void PythonTrigger<K, V>::Fire(const K* k, V* value, const V& newvalue, bool* doUpdate, bool isNew) { //const V& current, V& update) {
  string python_code = params_.get<string> ("python_code_short");
  PyObject *key, *callable;
  callable = PyObject_GetAttrString(crawl_module_.ptr(), python_code.c_str());
  key = PyString_FromString(k->c_str());

  // Make sure all the callfunctionobjarg arguments are fine
  if (key == NULL || callable == NULL) {
    LOG(ERROR) << "Failed to launch trigger " << python_code << "!";
    if (key == NULL) LOG(ERROR) << "[FAIL] key was null";
    if (callable == NULL) LOG(ERROR) << "[FAIL] callable was null";
    *doUpdate = true;
    return;
  }

  bool rv = PythonTrigger<K, V>::CallPythonTrigger(callable, key, value, newvalue, false, isNew);
  LOG(INFO) << "returning " << (rv?"TRUE":"FALSE") << " from normal trigger";
  *doUpdate = rv;
  return;
}

template<class K, class V>
bool PythonTrigger<K, V>::CallPythonTrigger(PyObjectPtr callable, PyObjectPtr key, V* value, const V& newvalue, bool isLongTrigger, bool isNewOrLast) {
  LOG(FATAL) << "No such CallPythonTrigger for this key/value pair type!";
  exit(1);
}

template<>
bool PythonTrigger<string, int64_t>::CallPythonTrigger(PyObjectPtr callable, PyObjectPtr key, int64_t* value, const int64_t& newvalue, bool isLongTrigger, bool isNewOrLast) {
  PyObjectPtr retval;

  //Handle LongTriggers
  if (isLongTrigger) {
    PyObject* lastrun_obj = PyBool_FromLong((long)isNewOrLast);
    if (lastrun_obj == NULL) {
      LOG(ERROR) << "Failed to bootstrap <int,int> long trigger launch";
      return true;
    }
    try {
      retval = PyObject_CallFunctionObjArgs(callable, key, lastrun_obj, NULL);
      Py_DECREF(callable);
    } catch (error_already_set& e) {
      PyErr_Print();
      exit(1);
    }
    return (retval == Py_True)?true:((retval == Py_False)?false:(bool)PyInt_AsLong(retval));
  }
 
  PyObject* cur_obj = PyLong_FromLongLong(*value);
  PyObject* upd_obj = PyLong_FromLongLong(newvalue);
  PyObject* isnew_obj = PyBool_FromLong((long)isNewOrLast);

  if (cur_obj == NULL || upd_obj == NULL || isnew_obj == NULL) {
    LOG(ERROR) << "Failed to bootstrap <int,int> trigger launch";
    return true;
  }
  try {
    retval = PyObject_CallFunctionObjArgs(
		callable, key, cur_obj, upd_obj, isnew_obj, NULL);
    Py_DECREF(callable);
  } catch (error_already_set& e) {
    PyErr_Print();
    exit(1);
  }

  *value = PyLong_AsLongLong(cur_obj);

  return (retval == Py_True)?true:((retval == Py_False)?false:(bool)PyInt_AsLong(retval));
}

template<>
bool PythonTrigger<string, string>::CallPythonTrigger(PyObjectPtr callable, PyObjectPtr key, string* value, const string& newvalue, bool isLongTrigger, bool isNewOrLast) {
  PyObjectPtr retval;

  //Handle LongTriggers
  if (isLongTrigger) {
    PyObject* lastrun_obj = PyBool_FromLong((long)isNewOrLast);
    if (lastrun_obj == NULL) {
      LOG(ERROR) << "Failed to bootstrap <int,int> long trigger launch";
      return true;
    }
    try {
      retval = PyObject_CallFunctionObjArgs(callable, key, lastrun_obj, NULL);
      Py_DECREF(callable);
    } catch (error_already_set& e) {
      PyErr_Print();
      exit(1);
    }
    return (retval == Py_True)?true:((retval == Py_False)?false:(bool)PyInt_AsLong(retval));
  }
 
  PyObject* cur_obj = PyString_FromString(value->c_str());
  PyObject* upd_obj = PyString_FromString(newvalue.c_str());
  PyObject* isnew_obj = PyBool_FromLong((long)isNewOrLast);

  if (cur_obj == NULL || upd_obj == NULL || isnew_obj == NULL) {
    LOG(ERROR) << "Failed to bootstrap <string,string> trigger launch";
    return true;
  }
  try {
    retval = PyObject_CallFunctionObjArgs(
		callable, key, cur_obj, upd_obj, isnew_obj, NULL);
    Py_DECREF(callable);
  } catch (error_already_set& e) {
    PyErr_Print();
    exit(1);
  }

  *value = PyString_AsString(cur_obj);

  return (retval == Py_True)?true:((retval == Py_False)?false:(bool)PyInt_AsLong(retval));
}

class PythonKernel: public DSMKernel {
public:
  virtual ~PythonKernel() {}
  PythonKernel() {
    try {
      object sys_module = import("sys");
      object sys_ns = sys_module.attr("__dict__");
      exec("path += ['src/examples/crawler']", sys_ns, sys_ns);
      exec("path += ['bin/release/examples/']", sys_ns, sys_ns);
      exec("path += ['bin/debug/examples/']", sys_ns, sys_ns);

      crawl_module_ = import("crawler");
      crawl_ns_ = crawl_module_.attr("__dict__");
    } catch (error_already_set e) {
      PyErr_Print();
      exit(1);
    }
  }

  void run_python_code() {
    the_kernel = this;
    string python_code = get_arg<string> ("python_code");
    LOG(INFO) << "Executing python code: " << python_code;
    try {
      exec(StringPrintf("%s\n", python_code.c_str()).c_str(), crawl_ns_, crawl_ns_);
    } catch (error_already_set e) {
      PyErr_Print();
      exit(1);
    }
  }

  void swap_python_accumulator() {
    LOG(FATAL) << "Swapping accumulators in Python not yet implemented.";
//    the_kernel = this;
//    string python_code = get_arg<string> ("python_accumulator");
//    LOG(INFO) << "Swapping python accumulator: " << python_code;
//    try {
//    } catch (error_already_set e) {
//      PyErr_Print();
//      exit(1);
//    }
  }

private:
  object crawl_module_;
  object crawl_ns_;
};
REGISTER_KERNEL(PythonKernel)
;
REGISTER_METHOD(PythonKernel, run_python_code)
;
REGISTER_METHOD(PythonKernel, swap_python_accumulator)
;

template class PythonTrigger<string, string>;
template class PythonTrigger<string, int64_t>;

}
