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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pgmimage.h"

//extern char *malloc();
//extern char *realloc();
//extern char *strcpy();

char *img_basename(char *filename)
{
  char *new, *part;
  int len, dex;

  len = strlen(filename);  dex = len - 1;
  while (dex > -1) {
    if (filename[dex] == '/') {
      break;
    } else {
      dex--;
    }
  }
  dex++;
  part = &(filename[dex]);
  len = strlen(part);
  new = (char *) malloc ((unsigned) ((len + 1) * sizeof (char)));
  strcpy(new, part);
  return(new);
}


IMAGE *img_alloc()
{
  IMAGE *new;

  new = (IMAGE *) malloc (sizeof (IMAGE));
  if (new == NULL) {
    printf("IMGALLOC: Couldn't allocate image structure\n");
    return (NULL);
  }
  new->rows = 0;
  new->cols = 0;
  new->name = NULL;
  new->data = NULL;
  return (new);
}


IMAGE *img_creat(char *name, int nr, int nc)
{
  int i, j;
  IMAGE *new;

  new = img_alloc();
  new->data = (int *) malloc ((unsigned) (nr * nc * sizeof(int)));
  new->name = img_basename(name);
  new->rows = nr;
  new->cols = nc;
  for (i = 0; i < nr; i++) {
    for (j = 0; j < nc; j++) {
      img_setpixel(new, i, j, 0);
    }
  }
  return (new);
}


void img_free(IMAGE *img)
{
  if (img->data) free ((char *) img->data);
  if (img->name) free ((char *) img->name);
  free ((char *) img);
}


void img_setpixel(IMAGE *img, int r, int c, int val)
{
  int nc;

  nc = img->cols;
  img->data[(r * nc) + c] = val;
}


int img_getpixel(IMAGE *img, int r, int c)
{
  int nc;

  nc = img->cols;
  return (img->data[(r * nc) + c]);
}


IMAGE *img_open(char *filename)
{
  IMAGE *new;
  FILE *pgm;
  char line[512], intbuf[100], ch;
  int type, nc, nr, maxval, i, j, k, found;

  new = img_alloc();
  if ((pgm = fopen(filename, "r")) == NULL) {
    printf("IMGOPEN: Couldn't open '%s'\n", filename);
    return(NULL);
  }

  new->name = img_basename(filename);

  /*** Scan pnm type information, expecting P5 ***/
  if (NULL == fgets(line, 511, pgm)) {
    printf("IMGOPEN: Failed to read data from PGM file\n");
    fclose(pgm);
    return(NULL);
  }
  sscanf(line, "P%d %d %d %d", &type, &nc, &nr, &maxval);
  if (type != 5 && type != 2) {
    printf("IMGOPEN: Only handles pgm files (type P5 or P2)\n");
    fclose(pgm);
    return(NULL);
  }

  /*** Get dimensions of pgm ***/
  //fgets(line, 511, pgm);
  //sscanf(line, "%d %d", &nc, &nr);
  new->rows = nr;
  new->cols = nc;

//  printf("[%d bytes for a (%dx%d) image]",(unsigned)(nr * nc * sizeof(int)),nc,nr);

  /*** Get maxval ***/
  //fgets(line, 511, pgm);
  //sscanf(line, "%d", &maxval);
  if (maxval > 255) {
    printf("IMGOPEN: Only handles pgm files of 8 bits or less\n");
    fclose(pgm);
    return(NULL);
  }


  new->data = (int *) malloc ((unsigned) (nr * nc * sizeof(int)));
  if (new->data == NULL) {
    printf("IMGOPEN: Couldn't allocate space for image data\n");
    fclose(pgm);
    return(NULL);
  }

  if (type == 5) {

    for (i = 0; i < nr; i++) {
      for (j = 0; j < nc; j++) {
        img_setpixel(new, i, j, fgetc(pgm));
      }
    }

  } else if (type == 2) {

    for (i = 0; i < nr; i++) {
      for (j = 0; j < nc; j++) {

        k = 0;  found = 0;
        while (!found) {
          ch = (char) fgetc(pgm);
          if (ch >= '0' && ch <= '9') {
            intbuf[k] = ch;  k++;
  	  } else {
            if (k != 0) {
              intbuf[k] = '\0';
              found = 1;
	    }
	  }
	}
        img_setpixel(new, i, j, atoi(intbuf));

      }
    }

  } else {
    printf("IMGOPEN: Fatal impossible error\n");
    fclose(pgm);
    return (NULL);
  }

  fclose(pgm);
  return (new);
}


int img_write(IMAGE *img, char *filename)
{
  int i, j, nr, nc, k, val;
  FILE *iop;

  nr = img->rows;  nc = img->cols;
  iop = fopen(filename, "w");
  fprintf(iop, "P2\n");
  fprintf(iop, "%d %d\n", nc, nr);
  fprintf(iop, "255\n");

  k = 1;
  for (i = 0; i < nr; i++) {
    for (j = 0; j < nc; j++) {
      val = img_getpixel(img, i, j);
      if ((val < 0) || (val > 255)) {
        printf("IMG_WRITE: Found value %d at row %d col %d\n", val, i, j);
        printf("           Setting it to zero\n");
        val = 0;
      }
      if (k % 10) {
        fprintf(iop, "%d ", val);
      } else {
        fprintf(iop, "%d\n", val);
      }
      k++;
    }
  }
  fprintf(iop, "\n");
  fclose(iop);
  return (1);
}


IMAGELIST *imgl_alloc()
{
  IMAGELIST *new;

  new = (IMAGELIST *) malloc (sizeof (IMAGELIST));
  if (new == NULL) {
    printf("IMGL_ALLOC: Couldn't allocate image list\n");
    return(NULL);
  }

  new->n = 0;
  new->list = NULL;

  return (new);
}


void imgl_add(IMAGELIST *il, IMAGE *img)
{
  int n;

  n = il->n;

  if (n == 0) {
    il->list = (IMAGE **) malloc ((unsigned) (sizeof (IMAGE *)));
  } else {
    il->list = (IMAGE **) realloc ((char *) il->list,
      (unsigned) ((n+1) * sizeof (IMAGE *)));
  }

  if (il->list == NULL) {
    printf("IMGL_ADD: Couldn't reallocate image list\n");
  }

  il->list[n] = img;
  il->n = n+1;
}


void imgl_free(IMAGELIST *il)
{
  free((char *) il->list);
  free((char *) il);
}


void imgl_load_images_from_infofile(IMAGELIST *il, const char *path, const char *filename)
{
  IMAGE *iimg;
  FILE *fp;
  char buf[2000];
  char concatpath[2048];

  if (filename[0] == '\0') {
    printf("IMGL_LOAD_IMAGES_FROM_INFOFILE: Invalid file '%s'\n", filename);
    return;
  }
  if (path[0] == '\0') {
    printf("IMGL_LOAD_IMAGES_FROM_INFOFILE: Invalid path '%s'\n", path);
    return;
  }

  strcpy(concatpath,path);
  if (path[strlen(path)-1] != '/')
    strcat(concatpath,"/");
  strcat(concatpath,filename);

  if ((fp = fopen(concatpath, "r")) == NULL) {
    printf("IMGL_LOAD_IMAGES_FROM_INFOFILE: Couldn't open '%s'\n", concatpath);
    return;
  }

  while (fgets(buf, 1999, fp) != NULL) {

    imgl_munge_name(buf);
//    printf("Loading '%s'...", buf);  fflush(stdout);
    concatpath[strlen(path)+(int)(path[strlen(path-1)]!= '/')] = '\0';		//force the path to end after
																	//the slash.
    strcat(concatpath,buf);
    if ((iimg = img_open(concatpath)) == 0) {
      printf("Couldn't open '%s'\n", concatpath);
    } else {
      imgl_add(il, iimg);
//      printf("done\n");
    }
    fflush(stdout);
  }

  fclose(fp);
}


// CM: Function modified to split on tab char as well
//     as newline, as per train.info format.
void imgl_munge_name(char *buf)
{
  int j;

  j = 0;
  while (buf[j] != '\n' && buf[j] != '\t' ) j++;
  buf[j] = '\0';
}
