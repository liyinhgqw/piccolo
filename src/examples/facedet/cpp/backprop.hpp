/*
 ******************************************************************
 * HISTORY
 * 15-Oct-94  Jeff Shufelt (js), Carnegie Mellon University
 *      Prepared for 15-681, Fall 1994.
 *
 * 30-Jun-10  Christopher Mitchell, Courant Institute
 *      Modified into BackProp class for Piccolo Project
 ******************************************************************
 */

#ifndef _BACKPROP_H_

#define _BACKPROP_H_

#include <iostream>
#include <stdlib.h>
#include <math.h>

using namespace std;

#define ABS(x)          (((x) > 0.0) ? (x) : (-(x)))
#define BIGRND 0x7fffffff

#define fastcopy(to,from,len)\
{\
  register char *_to,*_from;\
  register int _i,_l;\
  _to = (char *)(to);\
  _from = (char *)(from);\
  _l = (len);\
  for (_i = 0; _i < _l; _i++) *_to++ = *_from++;\
}

/*** The neural network data structure.  The network is assumed to
     be a fully-connected feedforward three-layer network.
     Unit 0 in each layer of units is the threshold unit; this means
     that the remaining units are indexed from 1 to n, inclusive.
 ***/

typedef struct {
  int input_n;                  /* number of input units */
  int hidden_n;                 /* number of hidden units */
  int output_n;                 /* number of output units */

  double *input_units;          /* the input units */
  double *hidden_units;         /* the hidden units */
  double *output_units;         /* the output units */

  double *hidden_delta;         /* storage for hidden unit error */
  double *output_delta;         /* storage for output unit error */

  double *target;               /* storage for target vector */

  double **input_weights;       /* weights from input to hidden layer */
  double **hidden_weights;      /* weights from hidden to output layer */

                                /*** The next two are for momentum ***/
  double **input_prev_weights;  /* previous change on input to hidden wgt */
  double **hidden_prev_weights; /* previous change on hidden to output wgt */
} BPNN;

class BackProp {
	public:
		void bpnn_initialize(int seed);
		void bpnn_free(BPNN *net, bool delstructure = true);
		BPNN *bpnn_create(int n_in, int n_hidden, int n_out);
		void bpnn_recreate(BPNN *net, int n_in, int n_hidden, int n_out, bool no_free = false);
		void bpnn_adjust_weights(double *delta, int ndelta, double *ly, int nly, double **w, double **oldw, double eta, double momentum);
		void bpnn_feedforward(BPNN *net);
		void bpnn_train(BPNN *net, double eta, double momentum, double *eo, double *eh);
		void bpnn_save(BPNN *net, const char *filename);
		BPNN *bpnn_read(const char* filename);
		void bpnn_layerforward(double *l1, double *l2, double **conn, int n1, int n2);
		void bpnn_output_error(double *delta,double *target,double *output,int nj,double *err);
		void bpnn_hidden_error(double *delta_h,int nh,double *delta_o,int no,double **who,double *hidden,double *err);

	private:
		double drnd();
		double dpn1();
		double squash(double x);
		double *alloc_1d_dbl(int n);
		double **alloc_2d_dbl(int m, int n);
		void bpnn_randomize_weights(double **w, int m, int n);
		void bpnn_zero_weights(double **w, int m, int n);
		BPNN *bpnn_internal_create(int n_in, int n_hidden, int n_out);
		void bpnn_internal_populate(BPNN* newvalnet, int n_in, int n_hidden, int n_out);
		
};

#endif
