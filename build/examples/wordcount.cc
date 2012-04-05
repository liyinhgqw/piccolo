
#include "client/client.h"

using namespace piccolo;

#line 1 "../src/examples/wordcount.pp"
#include "client/client.h"

using namespace std;
using namespace piccolo;

DEFINE_string(book_source, "/home/yavcular/books/520.txt", "");

static TextTable* books;
static TypedGlobalTable<string, int>* counts;

static int WordCount(const ConfigData& conf) {
  counts = CreateTable(0, 1, new Sharding::String, new Accumulators<int>::Sum);
  books = CreateTextTable(1, FLAGS_book_source, false);

  StartWorker(conf);

  Master m(conf);
  m.run_all("P_P_P_srcP_examplesP_wordcountP_ppMapKernel1", "map", books);

  m.run_all("P_P_P_srcP_examplesP_wordcountP_ppMapKernel2", "map", counts);

  return 0;
}
REGISTER_RUNNER(WordCount);

class P_P_P_srcP_examplesP_wordcountP_ppMapKernel1 : public DSMKernel {
public:
  virtual ~P_P_P_srcP_examplesP_wordcountP_ppMapKernel1() {}
  template <class K, class Value0>
  void run_iter(const K& k, Value0 &line) {
#line 18 "../src/examples/wordcount.pp"
    
    vector<StringPiece> words = StringPiece::split(line, " ");
    for (int j = 0; j < words.size(); ++j) {
      words[j].strip();
      counts->update(words[j].AsString(), 1);
    };
  }
  
  template <class TableA>
  void run_loop(TableA* a) {
    typename TableA::Iterator *it =  a->get_typed_iterator(current_shard());
    for (; !it->done(); it->Next()) {
      run_iter(it->key(), it->value());
    }
    delete it;
  }
  
  void map() {
      run_loop(books);
  }
};

REGISTER_KERNEL(P_P_P_srcP_examplesP_wordcountP_ppMapKernel1);
REGISTER_METHOD(P_P_P_srcP_examplesP_wordcountP_ppMapKernel1, map);


class P_P_P_srcP_examplesP_wordcountP_ppMapKernel2 : public DSMKernel {
public:
  virtual ~P_P_P_srcP_examplesP_wordcountP_ppMapKernel2() {}
  template <class K, class Value0>
  void run_iter(const K& k, Value0 &c) {
#line 26 "../src/examples/wordcount.pp"
    
     if (c > 50) {
       printf("%20s : %d\n", k.c_str(), c);
      };
  }
  
  template <class TableA>
  void run_loop(TableA* a) {
    typename TableA::Iterator *it =  a->get_typed_iterator(current_shard());
    for (; !it->done(); it->Next()) {
      run_iter(it->key(), it->value());
    }
    delete it;
  }
  
  void map() {
      run_loop(counts);
  }
};

REGISTER_KERNEL(P_P_P_srcP_examplesP_wordcountP_ppMapKernel2);
REGISTER_METHOD(P_P_P_srcP_examplesP_wordcountP_ppMapKernel2, map);


