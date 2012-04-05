#include "util/hashmap.h"
#include "util/common.h"
#include "kernel/table.h"

DEFINE_int32(count, 10000, "");
DEFINE_int32(size, 10000, "");

using namespace piccolo;

int main(int argc, char **argv) {
  Init(argc, argv);

  HashMap<int, int> test_map(10);
  for (int i = 0; i < FLAGS_count; ++i) {
    test_map.accumulate(i % FLAGS_size, 1, Accumulators<int>::sum);
  }

  for (int i = 0; i < FLAGS_size; ++i) {
    CHECK_EQ(test_map.get(i), FLAGS_count / FLAGS_size);
  }

}
