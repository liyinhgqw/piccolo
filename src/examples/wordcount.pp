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
  PMap({ line : books }, {
    vector<StringPiece> words = StringPiece::split(line, " ");
    for (int j = 0; j < words.size(); ++j) {
      words[j].strip();
      counts->update(words[j].AsString(), 1);
    }
  });

  PMap({ c : counts}, {
     if (c > 50) {
       printf("%20s : %d\n", k.c_str(), c);
      }
  });
  return 0;
}
REGISTER_RUNNER(WordCount);
