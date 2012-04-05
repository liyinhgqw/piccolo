#include "client/client.h"
#include "examples/examples.pb.h"

extern "C" {
#include "facedet/cpp/pgmimage.h"
}
#include "facedet/cpp/backprop.hpp"
#include "facedet/cpp/imagenet.hpp"

#include <sys/time.h>
#include <sys/resource.h>
#include <algorithm>
#include <libgen.h>

using namespace piccolo;
using namespace std;

static int NUM_WORKERS = 2;
#define PREFETCH 1024

DEFINE_string(infopn, "trainpn_random.info",
    "File containing list of training images");
DEFINE_string(netname, "/home/kerm/piccolo.hg/src/examples/facedet/netsave.net",
    "Filename of neural network save file");
DEFINE_string(pathpn, "/home/kerm/piccolo.hg/src/examples/facedet/trainset/",
    "Path to training data");
DEFINE_int32(epochs, 100, "Number of training epochs");
DEFINE_int32(hidden_neurons, 16, "Number of hidden heurons");
DEFINE_int32(savedelta, 100, "Save net every (savedelta) epochs");
DEFINE_int32(sharding, 1000, "Images per kernel execution");
DEFINE_bool(list_errors, false, "If true, will ennumerate misclassed images");
DEFINE_int32(total_ims, 43115, "Total number of images in DB");
DEFINE_int32(im_x, 32, "Image width in pixels");
DEFINE_int32(im_y, 32, "Image height in pixels");
DEFINE_bool(verify, true,
    "If true, will check initial data put into tables for veracity");
DEFINE_double(eta, 0.3, "Learning rate of BPNN");
DEFINE_double(momentum, 0.3, "Momentum of BPNN");

static TypedGlobalTable<int, BPNN>* nn_models = NULL;
static TypedGlobalTable<int, BPNN>* nn_prev_models = NULL;
static TypedGlobalTable<int, IMAGE>* train_ims = NULL;
static TypedGlobalTable<int, double>* performance = NULL;

//-----------------------------------------------
// Marshalling for BPNN type
//-----------------------------------------------
namespace piccolo {
template<> struct Marshal<BPNN> : MarshalBase {
  static void marshal(const BPNN& t, string *out) {
    int sizes[3];
    int i;
    VLOG(1)
        << "Marshalling BPNN from " << t.input_n << ", " << t.hidden_n << ", "
            << t.output_n;
    sizes[0] = t.input_n;
    sizes[1] = t.hidden_n;
    sizes[2] = t.output_n;
    out->append((char*) sizes, 3 * sizeof(int));
    for (i = 0; i < t.input_n + 1; i++) {
      out->append((char*) t.input_weights[i],
          (1 + t.hidden_n) * sizeof(double));
    }
    for (i = 0; i < t.hidden_n + 1; i++) {
      out->append((char*) t.hidden_weights[i],
          (1 + t.output_n) * sizeof(double));
    }
    VLOG(1) << "Marshalled BPNN to string of size " << out->length();
  }
  static void unmarshal(const StringPiece &s, BPNN* t) {
    BackProp bpnn;
    int sizes[3];
    memcpy(sizes, s.data, 3 * sizeof(int));
    int offset = 3 * sizeof(int), i;
    bpnn.bpnn_recreate(t, sizes[0], sizes[1], sizes[2], true);
    for (i = 0; i < t->input_n + 1; i++) {
      memcpy(t->input_weights[i], s.data + offset,
          (1 + t->hidden_n) * sizeof(double));
      offset += (1 + t->hidden_n) * sizeof(double);
    }
    for (i = 0; i < t->hidden_n + 1; i++) {
      memcpy(t->hidden_weights[i], s.data + offset,
          (1 + t->output_n) * sizeof(double));
      offset += (1 + t->output_n) * sizeof(double);
    }
    VLOG(1) << "Unmarshalled BPNN from string";
  }
};
}

//-----------------------------------------------
// Marshalling for IMAGE* type
//-----------------------------------------------
namespace piccolo {
template<> struct Marshal<IMAGE> : MarshalBase {
  static void marshal(const IMAGE& t, string *out) {
    char sizes[4];
    out->clear();
    sizes[0] = (char) ((t.rows) / 256);
    sizes[1] = (char) ((t.rows) % 256);
    sizes[2] = (char) ((t.cols) / 256);
    sizes[3] = (char) ((t.cols) % 256);
    out->append(sizes, 4);
    out->append((char*) (t.data), sizeof(int) * (t.rows) * (t.cols));
    sizes[0] = (char) (((strlen(t.name)) > 256) ? 256 : (strlen(t.name)));
    out->append(sizes, 1);
    out->append((char*) (t.name), (int) sizes[0]);
    VLOG(3)
        << "Marshalled image " << t.name << " to string of size "
            << out->length();
  }
  static void unmarshal(const StringPiece &s, IMAGE* t) {
    int r, c, sl;
    r = (256 * (unsigned int) (s.data[0])) + (unsigned int) (s.data[1]);
    c = (256 * (unsigned int) (s.data[2])) + (unsigned int) (s.data[3]);
    t->rows = r;
    t->cols = c;
    if (NULL == (t->data = (int*) malloc(sizeof(int) * r * c))) {
      fprintf(stderr, "Failed to marshal an image: out of memory.\n");
      exit(-1);
    }
    memcpy(t->data, s.data + 4, sizeof(int) * r * c);
    sl = (unsigned int) (s.data[4 + sizeof(int) * r * c]);
    if (NULL == (t->name = (char*) malloc(sl + 1))) {
      fprintf(stderr, "Failed to marshal an image: out of memory.\n");
      exit(-1);
    }
    strncpy(t->name, s.data + 5 + sizeof(int) * r * c, sl);
    t->name[sl] = '\0';
    VLOG(3) << "Unmarshalled image " << t->name << " from string";
  }
};
}

//----------------------------------------------------
// Face Classifier Kernel
// Takes a set of face samples, trains against it,
// then writes back a set of delta weights and biases
// from the oiginal model.
//----------------------------------------------------

class FCKernel: public DSMKernel {
public:
  int iter;
  BPNN *net;
  BackProp bpnn;
  ImageNet imnet;
  int hiddenn, imgsize;
  double out_err, hid_err, sumerr;

  void InitKernel() {
    imgsize = FLAGS_im_x * FLAGS_im_y;
    hiddenn = FLAGS_hidden_neurons;
    net = bpnn.bpnn_create(imgsize, hiddenn, 1);
  }

  void Initialize() {
    string netname, pathpn, infopn;
    IMAGELIST *trainlist;
    IMAGE *iimg;
    int seed;
    int train_n, i, j;

    //defaults
    seed = 20100630;
    netname = FLAGS_netname;
    pathpn = FLAGS_pathpn;
    infopn = FLAGS_infopn;
    fprintf(stdout, "Initialize kernel invoked.\n");

    /*** Create imagelists ***/
    trainlist = imgl_alloc();

    /*** Don't try to train if there's no training data ***/
    if (infopn.length() == 0 || pathpn.length() == 0) {
      fprintf(stderr,
          "FaceClass: Must specify path and filename of training data\n");
      exit(-1);
    }

    /*** Loading training images ***/
    imgl_load_images_from_infofile(trainlist, pathpn.c_str(), infopn.c_str());

    /*** If we haven't specified a network save file, we should... ***/
    if (netname.length() == 0) {
      fprintf(stderr, "Faceclass: Must specify an output file\n");
      exit(-1);
    }

    /*** Initialize the neural net package ***/
    bpnn.bpnn_initialize(seed);

    /*** Show number of images in train, test1, test2 ***/
    fprintf(stdout, "%d images in training set\n", trainlist->n);

    /*** If we've got at least one image to train on, go train the net ***/
    train_n = trainlist->n;
    if (train_n <= 0) {
      fprintf(stderr,
          "FaceClass: Must have at least one image to train from\n");
      exit(-1);
    }

    /*** Turn the IMAGELIST into a database of images, ie,  TypedGlobalTable<int, IMAGE>* train_ims ***/
    train_ims->resize(trainlist->n);
    for (i = 0; i < trainlist->n; i++) {
      train_ims->update(i, *(trainlist->list[i]));
    }
    train_ims->SendUpdates();

    /*** If requested, grab all the images out again and make sure they're correct ***/
    if (FLAGS_verify == true) {
      int goodims = 0;
      for (i = 0; i < trainlist->n; i++) {
        IMAGE testim = train_ims->get(i);
        if (testim.rows == (trainlist->list[i])->rows
            && testim.cols == (trainlist->list[i])->cols) {
          if (!strcmp(testim.name, (trainlist->list[i])->name)) {
            int immatch = 0;
            for (j = 0; j < (testim.rows * testim.cols); j++) {
              if (testim.data[j] != (trainlist->list[i])->data[j]) {
                immatch++;
                break;
              }
            }
            if (immatch) {
              fprintf(
                  stderr,
                  "[Verify] Image %d did not match image data in DB (%d/%d pixels mismatched)\n",
                  i, immatch, (testim.rows * testim.cols));
            } else {
              goodims++;
            }
          } else {
            fprintf(stderr,
                "[Verify] Image %d did not match image name in DB\n", i);
          }
        } else {
          if (train_ims->contains(i)) {
            fprintf(
                stderr,
                "[Verify] Image %d has incorrect dimensions in DB (got %dx%d, expected %dx%d)\n",
                i, testim.cols, testim.rows, (trainlist->list[i])->cols,
                (trainlist->list[i])->rows);
          } else {
            fprintf(stderr, "[Verify] Image key %d not found in DB!\n", i);
          }
        }
      }
      printf("[Verify] %d of %d images matched correctly in the DB\n", goodims,
          trainlist->n);
    }

    /*** Read network in if it exists, otherwise make one from scratch ***/
    if ((net = bpnn.bpnn_read(netname.c_str())) == NULL) {
      printf("Creating new network '%s'\n", netname.c_str());
      iimg = trainlist->list[0];
      imgsize = ROWS(iimg) * COLS(iimg);
      /* bthom ===========================
       make a net with:
       imgsize inputs, N hidden units, and 1 o      utput unit
       */
      net = bpnn.bpnn_create(imgsize, hiddenn, 1);
    }

    nn_prev_models->update(0, *net);
    nn_prev_models->SendUpdates();

    performance->update(0, 0.0f);
    performance->SendUpdates();
    if (FLAGS_verify == true) {
      if (performance->get(0) != 0.0)
        fprintf(stderr, "[Verify] Performance image count was not zero.\n");
    }
  }

  void TrainIteration() {
    int whichtable = 0;

    double cum_perf = 0.0;
    vector<double> perfs;
    vector<int> indices;
    double perfmin = 999.99, perfmax = -999.99;

    for (int i = 0; i < nn_prev_models->num_shards(); i++) {
      if (nn_prev_models->contains(i) && performance->contains(i)) {
        if (performance->get(i) > perfmax)
          perfmax = performance->get(i);
        if (performance->get(i) < perfmin)
          perfmin = performance->get(i);
      }
    }
    for (int i = 0; i < nn_prev_models->num_shards(); i++) {
      if (nn_prev_models->contains(i) && performance->contains(i)) {
        cum_perf += (performance->get(i)) - perfmin;
        perfs.push_back(cum_perf);
        indices.push_back(i);
      }
    }
    float whichmodelperf = (cum_perf) * ((float) rand() / (float) RAND_MAX);
//			printf("\nPicked 0 <= %f <= %f\n",whichmodelperf,cum_perf);
    vector<int>::iterator it1 = indices.begin();
    vector<double>::iterator it2 = perfs.begin();
    for (; it1 < indices.end() && it2 < perfs.end(); it1++, it2++) {
      if (whichmodelperf <= *it2) {
//					printf("%d->%f; ",*it1,*it2);
        whichtable = *it1;
        if (0 > (net = TableToBPNN(whichtable))) {
          fprintf(stderr, "Fatal error: could not load bpnn from table\n");
          exit(-1);
        }
        break;
      }
    }

//			printf("Shard %d picked model %d\n",current_shard(),whichtable);

    TypedTableIterator<int, IMAGE> *it = train_ims->get_typed_iterator(
        current_shard(), PREFETCH);
    for (; !it->done(); it->Next()) {
      IMAGE thisimg = it->value(); //grab this image

      imnet.load_input_with_image(&thisimg, net); //load the input layer
      imnet.load_target(&thisimg, net); //load target output layer

      /*** Feed forward input activations. ***/
      bpnn.bpnn_layerforward(net->input_units, net->hidden_units,
          net->input_weights, imgsize, hiddenn);
      bpnn.bpnn_layerforward(net->hidden_units, net->output_units,
          net->hidden_weights, hiddenn, 1);

      /*** Compute error on output and hidden units. ***/
      bpnn.bpnn_output_error(net->output_delta, net->target, net->output_units,
          1, &out_err);
      bpnn.bpnn_hidden_error(net->hidden_delta, hiddenn, net->output_delta, 1,
          net->hidden_weights, net->hidden_units, &hid_err);

      /*** Adjust input and hidden weights. ***/
//				double thiseta = FLAGS_eta/(double)train_ims->num_shards();
      double thiseta = FLAGS_eta;
      bpnn.bpnn_adjust_weights(net->output_delta, 1, net->hidden_units, hiddenn,
          net->hidden_weights, net->hidden_prev_weights, thiseta,
          FLAGS_momentum);
      bpnn.bpnn_adjust_weights(net->hidden_delta, hiddenn, net->input_units,
          imgsize, net->input_weights, net->input_prev_weights, thiseta,
          FLAGS_momentum);
    }

//			printf("Current shard is %d\n",current_shard());
    nn_models->update(current_shard(), *net);
    nn_models->SendUpdates();
    //bpnn.bpnn_free(net,false);
  }

  void PerformanceCheck() {
    bool classed; //true if this image was correct classed

    for (int j = 0; j < nn_prev_models->num_shards(); j++) {
      if (nn_prev_models->contains(j)) {
        if (0 > (net = TableToBPNN(j))) {
          fprintf(stderr, "Fatal error: could not load bpnn from table\n");
          exit(-1);
        }
        TypedTableIterator<int, IMAGE> *it = train_ims->get_typed_iterator(j,
            PREFETCH);

        for (; !it->done(); it->Next()) {
          IMAGE thisimg = it->value(); //grab this image

          imnet.load_input_with_image(&thisimg, net); //load the input layer
          bpnn.bpnn_feedforward(net);
          imnet.load_target(&thisimg, net); //load target output layer
          //					performance->update(1,err);							//accumulate more error

          classed = (net->output_units[1] > 0.5);
          classed = (net->target[1] > 0.5) ? classed : !classed;
          performance->update(j, (classed ? 1 : 0)); //accumulate correct classifications
        }
        //bpnn.bpnn_free(net,false);
      }
    }
//			printf("Got %d of %d images\n",correctims,ims);
    performance->SendUpdates();

    //This part is like swap()
    nn_prev_models->update(current_shard(), *net);
    nn_prev_models->SendUpdates();

  }

  void DisplayPerformance() {
    double sumperf = 0.0, thisperf;
    int perfshards = 0;

    for (int j = 0; j < performance->num_shards(); j++) {
      for (TypedTableIterator<int, double> *it =
          performance->get_typed_iterator(j, PREFETCH); !it->done();
          it->Next()) {
        thisperf = it->value();
        performance->update(it->key(), -it->value());
        thisperf /= (double) FLAGS_total_ims;
        sumperf += thisperf;
        perfshards++;
      }
    }

    printf("Performance: globally, %2.2f%% images correctly classified\n",
        100.0 * (sumperf / (double) perfshards));
    /*
     performance->update(0,-ims_correct);
     performance->update(1,-total_err);
     performance->SendUpdates();
     if (performance->get(0) != 0)
     LOG(FATAL) << "!!BUG!! Correct image count not reset to zero";
     if (performance->get(1) != 0)
     LOG(FATAL) << "!!BUG!! Accumulated error not reset to zero";
     */
  }

private:
  BPNN* TableToBPNN(int whichmodel) {
    static BPNN thisnet_ = nn_prev_models->get(whichmodel);
    return &thisnet_;
  }

};

REGISTER_KERNEL(FCKernel);
REGISTER_METHOD(FCKernel, InitKernel);
REGISTER_METHOD(FCKernel, Initialize);
REGISTER_METHOD(FCKernel, TrainIteration);
REGISTER_METHOD(FCKernel, PerformanceCheck);
REGISTER_METHOD(FCKernel, DisplayPerformance);

int Faceclass(const ConfigData& conf) {
  int i;

  int imageshards = ceil(FLAGS_total_ims / FLAGS_sharding);
  nn_models = CreateTable(0, imageshards, new Sharding::Mod,
      new Accumulators<BPNN>::Replace);
  nn_prev_models = CreateTable(1, imageshards, new Sharding::Mod,
      new Accumulators<BPNN>::Replace);
  train_ims = CreateTable(2, imageshards, new Sharding::Mod,
      new Accumulators<IMAGE>::Replace);
  performance = CreateTable(3, imageshards, new Sharding::Mod,
      new Accumulators<double>::Sum);

  StartWorker(conf);
  Master m(conf);

  NUM_WORKERS = conf.num_workers();
  printf("---- Initializing FaceClass on %d workers ----\n", NUM_WORKERS);

  //m.run_all("FCKernel","InitKernel",train_ims);
  m.run_one("FCKernel", "Initialize", train_ims);

  if (FLAGS_epochs > 0) {
    printf("Training underway (going to %d epochs)\n", FLAGS_epochs);
    printf("Will save network every %d epochs\n", FLAGS_savedelta);
    fflush(stdout);
  }

  for (i = 0; i < FLAGS_epochs; i++) {
    printf("--- Running epoch %03d of %03d ---\n", i, FLAGS_epochs);
    m.run_all("FCKernel", "TrainIteration", train_ims);
    m.run_all("FCKernel", "PerformanceCheck", train_ims);
    m.run_one("FCKernel", "DisplayPerformance", performance);
  }

  return 0;
}
REGISTER_RUNNER(Faceclass);
