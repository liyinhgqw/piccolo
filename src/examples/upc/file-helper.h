#ifndef FILEHELPER_H_
#define FILEHELPER_H_

#ifdef __cplusplus
extern "C" {
#endif

// Wrapper around recordfile calls to make them accessible from C.
struct RFile;

typedef struct {
  int id;
  int num_neighbors;
  int *neighbors;
} GraphEntry;

struct RFile *RecordFile_Open(const char* f, const char *mode);
GraphEntry *RecordFile_ReadGraphEntry(struct RFile *r);

void RecordFile_Close(struct RFile* f);

#ifdef __cplusplus
}
#endif

#endif /* FILEHELPER_H_ */
