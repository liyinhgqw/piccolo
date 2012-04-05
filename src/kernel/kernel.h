#ifndef KERNELREGISTRY_H_
#define KERNELREGISTRY_H_

#include "kernel/table.h"
#include "kernel/global-table.h"
#include "kernel/local-table.h"

#include "util/common.h"
#include <boost/function.hpp>
#include <boost/lexical_cast.hpp>

#include <map>

namespace piccolo {

template <class K, class V>
class TypedGlobalTable;

class TableBase;
class Worker;

#ifndef SWIG
class MarshalledMap {
public:
  struct MarshalledValue {
    virtual string ToString() const = 0;
    virtual void FromString(const string& s) = 0;
    virtual void set(const void* nv) = 0;
    virtual void* get() const = 0;
  };

  template <class T>
  struct MarshalledValueT  : public MarshalledValue {
    MarshalledValueT() : v(new T) {}
    ~MarshalledValueT() { delete v; }

    string ToString() const {
      string tmp;
      m_.marshal(*v, &tmp);
      return tmp;
    }

    void FromString(const string& s) {
      m_.unmarshal(s, v);
    }

    void* get() const { return v; }
    void set(const void *nv) {
      *v = *(T*)nv;
    }

    mutable Marshal<T> m_;
    T *v;
  };

  template <class T>
  void put(const string& k, const T& v) {
    if (serialized_.find(k) != serialized_.end()) {
      serialized_.erase(serialized_.find(k));
    }

    if (p_.find(k) == p_.end()) {
      p_[k] = new MarshalledValueT<T>;
    }

    p_[k]->set(&v);
  }

  template <class T>
  T& get(const string& k) const {
    if (serialized_.find(k) != serialized_.end()) {
      p_[k] = new MarshalledValueT<T>;
      p_[k]->FromString(serialized_[k]);
      serialized_.erase(serialized_.find(k));
    }

    return *(T*)p_.find(k)->second->get();
  }

  bool contains(const string& key) const {
    return p_.find(key) != p_.end() ||
           serialized_.find(key) != serialized_.end();
  }

  Args* ToMessage() const {
    Args* out = new Args;
    for (std::map<string, MarshalledValue*>::const_iterator i = p_.begin(); i != p_.end(); ++i) {
      Arg *p = out->add_param();
      p->set_key(i->first);
      p->set_value(i->second->ToString());
    }
    return out;
  }

  // We can't immediately deserialize the parameters passed in, since sadly we don't
  // know the type yet.  Instead, save the string values on the side, and de-serialize
  // on request.
  void FromMessage(const Args& p) {
    for (int i = 0; i < p.param_size(); ++i) {
      serialized_[p.param(i).key()] = p.param(i).value();
    }
  }

private:
  mutable std::map<string, MarshalledValue*> p_;
  mutable std::map<string, string> serialized_;
};
#endif


class DSMKernel {
public:
  // Called upon creation of this kernel by a worker.
  virtual void InitKernel() {}

  // The table and shard being processed.
  int current_shard() const { return shard_; }
  int current_table() const { return table_id_; }

  template <class T>
  T& get_arg(const string& key) const {
    return args_.get<T>(key);
  }

  template <class T>
  T& get_cp_var(const string& key, T defval=T()) {
    if (!cp_.contains(key)) {
      cp_.put(key, defval);
    }
    return cp_.get<T>(key);
  }

  GlobalTable* get_table(int id);

  template <class K, class V>
  TypedGlobalTable<K, V>* get_table(int id) {
    return dynamic_cast<TypedGlobalTable<K, V>*>(get_table(id));
  }
private:
  friend class Worker;
  friend class Master;

  void initialize_internal(Worker* w,
                           int table_id, int shard);

  void set_args(const MarshalledMap& args);
  void set_checkpoint(const MarshalledMap& args);

  Worker *w_;
  int shard_;
  int table_id_;
  MarshalledMap args_;
  MarshalledMap cp_;
};

struct KernelInfo {
  virtual ~KernelInfo() {}
  KernelInfo(const char* name) : name_(name) {}

  virtual DSMKernel* create() = 0;
  virtual void Run(DSMKernel* obj, const string& method_name) = 0;
  virtual bool has_method(const string& method_name) = 0;

  string name_;
};

template <class C>
struct KernelInfoT : public KernelInfo {
  typedef void (C::*Method)();
  std::map<string, Method> methods_;

  KernelInfoT(const char* name) : KernelInfo(name) {}

  DSMKernel* create() { return new C; }

  void Run(DSMKernel* obj, const string& method_id) {
    boost::function<void (C*)> m(methods_[method_id]);
    m((C*)obj);
  }

  bool has_method(const string& name) {
    return methods_.find(name) != methods_.end();
  }

  void register_method(const char* mname, Method m) { methods_[mname] = m; }
};

class ConfigData;
class KernelRegistry {
public:
  typedef std::map<string, KernelInfo*> Map;
  Map& kernels() { return m_; }
  KernelInfo* kernel(const string& name) { return m_[name]; }

  static KernelRegistry* Get();
private:
  KernelRegistry() {}
  Map m_;
};

template <class C>
struct KernelRegistrationHelper {
  KernelRegistrationHelper(const char* name) {
    KernelRegistry::Map& kreg = KernelRegistry::Get()->kernels();

    CHECK(kreg.find(name) == kreg.end());
    kreg.insert(std::make_pair(name, new KernelInfoT<C>(name)));
  }
};

template <class C>
struct MethodRegistrationHelper {
  MethodRegistrationHelper(const char* klass, const char* mname, void (C::*m)()) {
    ((KernelInfoT<C>*)KernelRegistry::Get()->kernel(klass))->register_method(mname, m);
  }
};

#define REGISTER_KERNEL(klass)\
  static KernelRegistrationHelper<klass> k_helper_ ## klass(#klass);

#define REGISTER_METHOD(klass, method)\
  static MethodRegistrationHelper<klass> m_helper_ ## klass ## _ ## method(#klass, #method, &klass::method);

#define REGISTER_RUNNER(r)\
  int KernelRunner(const ConfigData& c) {\
    return r(c);\
  }\

}
#endif /* KERNELREGISTRY_H_ */
