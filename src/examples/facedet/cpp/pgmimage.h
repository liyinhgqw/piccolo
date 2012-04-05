/*
 ******************************************************************
 * HISTORY
 * 15-Oct-94  Jeff Shufelt (js), Carnegie Mellon University
 *      Prepared for 15-681, Fall 1994.
 *
 * 29-June-10 Christopher Mitchell, Courant Institute
 *      Modified for use with Piccolo project
 *      For use with train.info format from Scouter project
 *
 ******************************************************************
 */

#ifndef _PGMIMAGE_H_

#define _PGMIMAGE_H_

typedef struct {
  char *name;
  int rows, cols;
  int *data;
} IMAGE;

typedef struct {
  int n;
  IMAGE **list;
} IMAGELIST;

/*** User accessible macros ***/

#define ROWS(img)  ((img)->rows)
#define COLS(img)  ((img)->cols)
#define NAME(img)   ((img)->name)

/*** User accessible functions ***/

char *img_basename(char *filename);
IMAGE *img_alloc();
IMAGE *img_creat(char *name, int nr, int nc);
void img_free(IMAGE *img);
void img_setpixel(IMAGE *img, int r, int c, int val);
int img_getpixel(IMAGE *img, int r, int c);
IMAGE *img_open(char *filename);
int img_write(IMAGE *img, char *filename);
IMAGELIST *imgl_alloc();
void imgl_add(IMAGELIST *il, IMAGE *img);
void imgl_free(IMAGELIST *il);
void imgl_load_images_from_infofile(IMAGELIST *il, const char *path, const char *filename);
void imgl_munge_name(char *buf);

#endif
