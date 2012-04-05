/*
 ******************************************************************
 * HISTORY
 * 15-Oct-94  Jeff Shufelt (js), Carnegie Mellon University
 *	Prepared for 15-681, Fall 1994.
 *
 ******************************************************************
 */

#include <stdio.h>
#include <fcntl.h>
#include "backprop.hpp"

/*** Return random number between 0.0 and 1.0 ***/
double BackProp::drnd()
{
  return ((double) random() / (double) BIGRND);
}

/*** Return random number between -1.0 and 1.0 ***/
double BackProp::dpn1()
{
  return ((drnd() * 2.0) - 1.0);
}

/*** The squashing function.  Currently, it's a sigmoid. ***/

double BackProp::squash(double x)
{
  return (1.0 / (1.0 + exp(-x)));
}


/*** Allocate 1d array of doubles ***/

double *BackProp::alloc_1d_dbl(int n)
{
  double *newval;

  newval = (double *) malloc ((unsigned) (n * sizeof (double)));
  if (newval == NULL) {
    printf("ALLOC_1D_DBL: Couldn't allocate array of doubles\n");
    return (NULL);
  }
  return (newval);
}


/*** Allocate 2d array of doubles ***/

double **BackProp::alloc_2d_dbl(int m, int n)
{
  int i;
  double **newval;

  newval = (double **) malloc ((unsigned) (m * sizeof (double *)));
  if (newval == NULL) {
    printf("ALLOC_2D_DBL: Couldn't allocate array of dbl ptrs\n");
    return (NULL);
  }

  for (i = 0; i < m; i++) {
    newval[i] = alloc_1d_dbl(n);
  }

  return (newval);
}


void BackProp::bpnn_randomize_weights(double **w, int m, int n)
{
  int i, j;

  for (i = 0; i <= m; i++) {
    for (j = 0; j <= n; j++) {
      w[i][j] = dpn1();
    }
  }
}


void BackProp::bpnn_zero_weights(double **w, int m, int n)
{
  int i, j;

  for (i = 0; i <= m; i++) {
    for (j = 0; j <= n; j++) {
      w[i][j] = 0.0;
    }
  }
}


void BackProp::bpnn_initialize(int seed)
{
  printf("Random number generator seed: %d\n", seed);
  srandom(seed);
}


void BackProp::bpnn_internal_populate(BPNN* newvalnet, int n_in, int n_hidden, int n_out) {
  newvalnet->input_n = n_in;
  newvalnet->hidden_n = n_hidden;
  newvalnet->output_n = n_out;
  newvalnet->input_units = alloc_1d_dbl(n_in + 1);
  newvalnet->hidden_units = alloc_1d_dbl(n_hidden + 1);
  newvalnet->output_units = alloc_1d_dbl(n_out + 1);

  newvalnet->hidden_delta = alloc_1d_dbl(n_hidden + 1);
  newvalnet->output_delta = alloc_1d_dbl(n_out + 1);
  newvalnet->target = alloc_1d_dbl(n_out + 1);

  newvalnet->input_weights = alloc_2d_dbl(n_in + 1, n_hidden + 1);
  newvalnet->hidden_weights = alloc_2d_dbl(n_hidden + 1, n_out + 1);

  newvalnet->input_prev_weights = alloc_2d_dbl(n_in + 1, n_hidden + 1);
  newvalnet->hidden_prev_weights = alloc_2d_dbl(n_hidden + 1, n_out + 1);
}

BPNN *BackProp::bpnn_internal_create(int n_in, int n_hidden, int n_out)
{
  BPNN *newvalnet;

  newvalnet = (BPNN *) malloc (sizeof (BPNN));
  if (newvalnet == NULL) {
    printf("BPNN_CREATE: Couldn't allocate neural network\n");
    return (NULL);
  }

  bpnn_internal_populate(newvalnet, n_in, n_hidden, n_out);

  return (newvalnet);
}


void BackProp::bpnn_free(BPNN *net, bool delstructure)
{
  int n1, n2, i;

  if (!net)
    return;

  n1 = net->input_n;
  n2 = net->hidden_n;

  printf("%p %p %p\n",net->input_units,net->hidden_units,net->output_units);
  if (net->input_units)
    free((char *) net->input_units);
  if (net->hidden_units)
    free((char *) net->hidden_units);
  if (net->output_units)
    free((char *) net->output_units);

  printf("%p %p %p\n",net->hidden_delta,net->output_delta,net->target);
  if (net->hidden_delta)
    free((char *) net->hidden_delta);
  if (net->output_delta)
    free((char *) net->output_delta);
  if (net->target)
    free((char *) net->target);

  printf("%p %p\n",net->input_weights,net->input_prev_weights);
  for (i = 0; i <= n1; i++) {
    if (net->input_weights && net->input_weights[i])
      free((char *) net->input_weights[i]);
    if (net->input_prev_weights && net->input_prev_weights[i])
      free((char *) net->input_prev_weights[i]);
  }
  if (net->input_weights)
    free((char *) net->input_weights);
  if (net->input_prev_weights)
    free((char *) net->input_prev_weights);

  printf("%p %p\n",net->hidden_weights,net->hidden_prev_weights);
  for (i = 0; i <= n2; i++) {
    if (net->hidden_weights && net->hidden_weights[i])
      free((char *) net->hidden_weights[i]);
    if (net->hidden_prev_weights && net->hidden_prev_weights[i])
      free((char *) net->hidden_prev_weights[i]);
  }
  if (net->hidden_weights)
    free((char *) net->hidden_weights);
  if (net->hidden_prev_weights)
    free((char *) net->hidden_prev_weights);

  if (delstructure && net)
    free((char *) net);
}


/*** Creates a newval fully-connected network from scratch,
     with the given numbers of input, hidden, and output units.
     Threshold units are automatically included.  All weights are
     randomly initialized.

     Space is also allocated for temporary storage (momentum weights,
     error computations, etc).
***/

void BackProp::bpnn_recreate(BPNN* net, int n_in, int n_hidden, int n_out, bool no_free) {
  if (!no_free)
    bpnn_free(net,false);
  bpnn_internal_populate(net,n_in,n_hidden,n_out);
}

BPNN *BackProp::bpnn_create(int n_in, int n_hidden, int n_out)
{

  BPNN *newvalnet;

  newvalnet = bpnn_internal_create(n_in, n_hidden, n_out);

#ifdef INITZERO
  bpnn_zero_weights(newvalnet->input_weights, n_in, n_hidden);
#else
  bpnn_randomize_weights(newvalnet->input_weights, n_in, n_hidden);
#endif
  bpnn_randomize_weights(newvalnet->hidden_weights, n_hidden, n_out);
  bpnn_zero_weights(newvalnet->input_prev_weights, n_in, n_hidden);
  bpnn_zero_weights(newvalnet->hidden_prev_weights, n_hidden, n_out);

  return (newvalnet);
}



void BackProp::bpnn_layerforward(double *l1, double *l2, double **conn, int n1, int n2)
{
  double sum;
  int j, k;

  /*** Set up thresholding unit ***/
  l1[0] = 1.0;

  /*** For each unit in second layer ***/
  for (j = 1; j <= n2; j++) {

    /*** Compute weighted sum of its inputs ***/
    sum = 0.0;
    for (k = 0; k <= n1; k++) {
      sum += conn[k][j] * l1[k];
    }
    l2[j] = squash(sum);
  }

}


void BackProp::bpnn_output_error(double *delta,double *target,double *output,int nj,double *err)
{
  int j;
  double o, t, errsum;

  errsum = 0.0;
  for (j = 1; j <= nj; j++) {
    o = output[j];
    t = target[j];
    delta[j] = o * (1.0 - o) * (t - o);
    errsum += ABS(delta[j]);
  }
  *err = errsum;
}


void BackProp::bpnn_hidden_error(double *delta_h,int nh,double *delta_o,int no,double **who,double *hidden,double *err)
{
  int j, k;
  double h, sum, errsum;

  errsum = 0.0;
  for (j = 1; j <= nh; j++) {
    h = hidden[j];
    sum = 0.0;
    for (k = 1; k <= no; k++) {
      sum += delta_o[k] * who[j][k];
    }
    delta_h[j] = h * (1.0 - h) * sum;
    errsum += ABS(delta_h[j]);
  }
  *err = errsum;
}


void BackProp::bpnn_adjust_weights(double *delta, int ndelta, double *ly, int nly, double **w, double **oldw, double eta, double momentum)
{
  double newval_dw;
  int k, j;

  ly[0] = 1.0;
  for (j = 1; j <= ndelta; j++) {
    for (k = 0; k <= nly; k++) {
      newval_dw = ((eta * delta[j] * ly[k]) + (momentum * oldw[k][j]));
      w[k][j] += newval_dw;
      oldw[k][j] = newval_dw;
    }
  }
}


void BackProp::bpnn_feedforward(BPNN *net)
{
  int in, hid, out;

  in = net->input_n;
  hid = net->hidden_n;
  out = net->output_n;

  /*** Feed forward input activations. ***/
  bpnn_layerforward(net->input_units, net->hidden_units,
      net->input_weights, in, hid);
  bpnn_layerforward(net->hidden_units, net->output_units,
      net->hidden_weights, hid, out);

}


void BackProp::bpnn_train(BPNN *net, double eta, double momentum, double *eo, double *eh)
{
  int in, hid, out;
  double out_err, hid_err;

  in = net->input_n;
  hid = net->hidden_n;
  out = net->output_n;

  /*** Feed forward input activations. ***/
  bpnn_layerforward(net->input_units, net->hidden_units,
      net->input_weights, in, hid);
  bpnn_layerforward(net->hidden_units, net->output_units,
      net->hidden_weights, hid, out);

  /*** Compute error on output and hidden units. ***/
  bpnn_output_error(net->output_delta, net->target, net->output_units,
      out, &out_err);
  bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out,
      net->hidden_weights, net->hidden_units, &hid_err);
  *eo = out_err;
  *eh = hid_err;

  /*** Adjust input and hidden weights. ***/
  bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid,
      net->hidden_weights, net->hidden_prev_weights, eta, momentum);
  bpnn_adjust_weights(net->hidden_delta, hid, net->input_units, in,
      net->input_weights, net->input_prev_weights, eta, momentum);

}

void BackProp::bpnn_save(BPNN *net, const char *filename)
{
  int fd, n1, n2, n3, i, j, memcnt;
  double dvalue, **w;
  char *mem;

  if ((fd = creat(filename, 0644)) == -1) {
    printf("BPNN_SAVE: Cannot create '%s'\n", filename);
    return;
  }

  n1 = net->input_n;  n2 = net->hidden_n;  n3 = net->output_n;
  printf("Saving %dx%dx%d network to '%s'\n", n1, n2, n3, filename);
  fflush(stdout);

  if (0 > write(fd, (char *) &n1, sizeof(int)) || 
      0 > write(fd, (char *) &n2, sizeof(int)) ||
      0 > write(fd, (char *) &n3, sizeof(int))) {
    fprintf(stderr,"Could not write network to file!\n");
    exit(-1);
  }

  memcnt = 0;
  w = net->input_weights;
  mem = (char *) malloc ((unsigned) ((n1+1) * (n2+1) * sizeof(double)));
  for (i = 0; i <= n1; i++) {
    for (j = 0; j <= n2; j++) {
      dvalue = w[i][j];
      fastcopy(&mem[memcnt], &dvalue, sizeof(double));
      memcnt += sizeof(double);
    }
  }
  write(fd, mem, (n1+1) * (n2+1) * sizeof(double));
  free(mem);

  memcnt = 0;
  w = net->hidden_weights;
  mem = (char *) malloc ((unsigned) ((n2+1) * (n3+1) * sizeof(double)));
  for (i = 0; i <= n2; i++) {
    for (j = 0; j <= n3; j++) {
      dvalue = w[i][j];
      fastcopy(&mem[memcnt], &dvalue, sizeof(double));
      memcnt += sizeof(double);
    }
  }
  if (0 > write(fd, mem, (n2+1) * (n3+1) * sizeof(double))) {
    fprintf(stderr,"Could not write network to file!\n");
    exit(-1);
  }
  free(mem);

  close(fd);
  return;
}


BPNN *BackProp::bpnn_read(const char* filename)
{
  char *mem;
  BPNN *newval;
  int fd, n1, n2, n3, i, j, memcnt;

  if ((fd = open(filename, 0, 0644)) == -1) {
    return (NULL);
  }

  printf("Reading '%s'\n", filename);  fflush(stdout);

  read(fd, (char *) &n1, sizeof(int));
  read(fd, (char *) &n2, sizeof(int));
  read(fd, (char *) &n3, sizeof(int));
  newval = bpnn_internal_create(n1, n2, n3);

  printf("'%s' contains a %dx%dx%d network\n", filename, n1, n2, n3);
  printf("Reading input weights...");  fflush(stdout);

  memcnt = 0;
  mem = (char *) malloc ((unsigned) ((n1+1) * (n2+1) * sizeof(double)));
  read(fd, mem, (n1+1) * (n2+1) * sizeof(double));
  for (i = 0; i <= n1; i++) {
    for (j = 0; j <= n2; j++) {
      fastcopy(&(newval->input_weights[i][j]), &mem[memcnt], sizeof(double));
      memcnt += sizeof(double);
    }
  }
  free(mem);

  printf("Done\nReading hidden weights...");  fflush(stdout);

  memcnt = 0;
  mem = (char *) malloc ((unsigned) ((n2+1) * (n3+1) * sizeof(double)));
  read(fd, mem, (n2+1) * (n3+1) * sizeof(double));
  for (i = 0; i <= n2; i++) {
    for (j = 0; j <= n3; j++) {
      fastcopy(&(newval->hidden_weights[i][j]), &mem[memcnt], sizeof(double));
      memcnt += sizeof(double);
    }
  }
  free(mem);
  close(fd);

  printf("Done\n");  fflush(stdout);

  bpnn_zero_weights(newval->input_prev_weights, n1, n2);
  bpnn_zero_weights(newval->hidden_prev_weights, n2, n3);

  return (newval);
}
