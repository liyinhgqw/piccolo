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

IMAGE *img_open();
IMAGE *img_creat();
void img_setpixel();
int img_getpixel();
int img_write();
void img_free();

void imgl_load_images_from_infofile(IMAGELIST* il, char *path, char* filename);
void imgl_munge_name(char *buf);

IMAGELIST *imgl_alloc();
void imgl_add();
void imgl_free();

#endif
