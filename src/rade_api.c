/*---------------------------------------------------------------------------*\

  rade_api.c

  Library of API functions that implement the Radio Autoencoder API.

\*---------------------------------------------------------------------------*/

/*
  Copyright (C) 2024 David Rowe

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions
  are met:

  - Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  - Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
  LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#define VERSION 1  // bump me every time API changes

#include <assert.h>
#include "rade_api.h"

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "rade_core.h" 
#include "rade_enc.h"
#include "rade_enc_data.h"
#include "rade_dec.h"
#include "rade_dec_data.h"

static PyThreadState* main_thread_state; // needed to unlock the GIL after initialization

struct rade {
  int flags;
  int auxdata;
  int nb_total_features;
  int num_used_features;
  int num_features;     
 
  npy_intp Nmf, Neoo;     
  npy_intp nin, nin_max;   
  npy_intp n_features_in, n_features_out;  
      
  RADEEnc      enc_model;
  RADEEncState enc_state;
  PyObject *pModule_radae_tx, *pInst_radae_tx;
  PyObject *pMeth_radae_tx, *pArgs_radae_tx;
  npy_intp n_floats_in;
  float *floats_in;  // could be features or latents z
  RADE_COMP *tx_out;
  PyObject *pMeth_radae_tx_eoo, *pArgs_radae_tx_eoo;
  RADE_COMP *tx_eoo_out;

  RADEDec      dec_model;
  RADEDecState dec_state;
  PyObject *pModule_radae_rx, *pInst_radae_rx;
  PyObject *pMeth_radae_rx, *pArgs_radae_rx;
  PyObject *pMeth_sum_uw_errors;
  npy_intp n_floats_out;
  float *floats_out;
  RADE_COMP *rx_in;
};


void check_error(PyObject *p, char s1[], char s2[]) {
    if (p == NULL) {
        if (PyErr_Occurred()) {
            PyErr_Print();
        }
        fprintf(stderr, "Error: %s %s\n", s1, s2);
        // TODO: for library make this fail gracefully with return code or exception
        exit(1);
    }
}

void check_callable(PyObject *p, char s1[], char s2[]) {
    if (!PyCallable_Check(p))
        check_error(NULL, s1, s2);
}

/* helper function to call a Python "getter" method that return a long */
long call_getter(PyObject *pInst, char meth_name[]) {
    PyObject *pMeth, *pArgs, *pValue;
    long ret;

    pMeth = PyObject_GetAttrString(pInst, meth_name);
    check_error(pMeth, "can't find", meth_name);
    check_callable(pMeth, meth_name, "is not callable");

    pArgs = Py_BuildValue("()");
    pValue = PyObject_CallObject(pMeth, pArgs);
    check_error(pValue, "call to", meth_name);
    ret = PyLong_AsLong(pValue);

    Py_DECREF(pArgs);
    Py_DECREF(pValue);
    Py_XDECREF(pMeth);

    return ret;
}

// returns 0 for success
int rade_tx_open(struct rade *r) {
    PyObject *pName, *pClass;
    PyObject *pValue, *pArgs, *pkwArgs;
    char *python_module_name = "radae_txe";
    char *do_radae_tx_meth_name = "do_radae_tx";
    char *do_eoo_meth_name = "do_eoo";

    // Load module of Python code
    pName = PyUnicode_DecodeFSDefault(python_module_name);
    r->pModule_radae_tx = PyImport_Import(pName);
    check_error(r->pModule_radae_tx, "importing", python_module_name);
    Py_DECREF(pName);

    // Find class and create an instance
    pClass = PyObject_GetAttrString(r->pModule_radae_tx, "radae_tx");
    check_error(pClass, "finding class", "radae_tx");
    pArgs = Py_BuildValue("(s)", "model19_check3/checkpoints/checkpoint_epoch_100.pth");
    pkwArgs = Py_BuildValue("{s:i}", "bypass_enc", r->flags & RADE_USE_C_ENCODER);
    r->pInst_radae_tx = PyObject_Call(pClass, pArgs, pkwArgs);
    check_error(r->pInst_radae_tx, "Creating instance of class", "radae_tx");
    Py_DECREF(pClass);
    Py_DECREF(pArgs);
    Py_DECREF(pkwArgs);

    r->n_features_in = (int)call_getter(r->pInst_radae_tx, "get_n_features_in");
    r->n_floats_in = (int)call_getter(r->pInst_radae_tx, "get_n_floats_in");
    r->Nmf = (int)call_getter(r->pInst_radae_tx, "get_Nmf");
    r->Neoo = (int)call_getter(r->pInst_radae_tx, "get_Neoo");
    fprintf(stderr, "n_features_in: %d n_floats_in: %d Nmf: %d Neoo: %d\n", (int)r->n_features_in, (int)r->n_floats_in, (int)r->Nmf, (int)r->Neoo);
        
    // RADAE Tx ---------------------------------------------------------

    r->pArgs_radae_tx = PyTuple_New(2);

    r->pMeth_radae_tx = PyObject_GetAttrString(r->pInst_radae_tx, do_radae_tx_meth_name);
    check_error(r->pMeth_radae_tx, "finding",  do_radae_tx_meth_name);
    check_callable(r->pMeth_radae_tx, do_radae_tx_meth_name, "not callable");

    // 1st Python function arg - numpy array of float features (Python core encoder) or latents (C core encoder)
    r->floats_in = (float*)malloc(sizeof(float)*r->n_floats_in);
    assert(r->floats_in != NULL);
    pValue = PyArray_SimpleNewFromData(1, &r->n_floats_in, NPY_FLOAT, r->floats_in);
    check_error(pValue, "setting up numpy array", "floats_in");
    PyTuple_SetItem(r->pArgs_radae_tx, 0, pValue);

    // 2nd Python arg is a numpy array used for output to C
    r->tx_out = (RADE_COMP*)malloc(sizeof(RADE_COMP)*r->Nmf);
    assert(r->tx_out != NULL);
    pValue = PyArray_SimpleNewFromData(1, &r->Nmf, NPY_CFLOAT, r->tx_out);
    check_error(pValue, "setting up numpy array", "tx_out");
    PyTuple_SetItem(r->pArgs_radae_tx, 1, pValue);

    // End of Over --------------------------------------------------------

    r->pMeth_radae_tx_eoo = PyObject_GetAttrString(r->pInst_radae_tx, do_eoo_meth_name);
    check_error(r->pMeth_radae_tx_eoo, "finding",  do_eoo_meth_name);
    check_callable(r->pMeth_radae_tx_eoo, do_eoo_meth_name, "not callable");
    r->pArgs_radae_tx_eoo = PyTuple_New(1);

    // Python arg is a numpy array used for output to C
    r->tx_eoo_out = (RADE_COMP*)malloc(sizeof(RADE_COMP)*r->Neoo);
    assert(r->tx_eoo_out != NULL);
    pValue = PyArray_SimpleNewFromData(1, &r->Neoo, NPY_CFLOAT, r->tx_eoo_out);
    check_error(pValue, "setting up numpy array", "tx_eoo_out");
    PyTuple_SetItem(r->pArgs_radae_tx_eoo, 0, pValue);

    if (r->flags & RADE_USE_C_ENCODER) {
      if (init_radeenc(&r->enc_model, radeenc_arrays, r->num_features*RADE_FRAMES_PER_STEP) != 0) {
        fprintf(stderr, "Error initialising built-in C encoder model\n");
        exit(1);        
      }
      rade_init_encoder(&r->enc_state);
    }

    return 0;
}

void rade_tx_close(struct rade *r) {
  // TODO we may need more of these, see if there are any memory leaks
  Py_DECREF(r->pArgs_radae_tx);
  Py_DECREF(r->pMeth_radae_tx);
  Py_DECREF(r->pMeth_radae_tx_eoo);
  Py_DECREF(r->pArgs_radae_tx_eoo);
  Py_DECREF(r->pInst_radae_tx);
  Py_DECREF(r->pModule_radae_tx);

  free(r->floats_in);
  free(r->tx_out);
  free(r->tx_eoo_out);
}

// returns 0 for success
int rade_rx_open(struct rade *r) {
    PyObject *pName, *pClass;
    PyObject *pValue;
    PyObject *pArgs, *pkwArgs;
    char *python_module_name = "radae_rxe";
    char *do_radae_rx_meth_name = "do_radae_rx";
    char *sum_uw_errors_meth_name = "sum_uw_errors";

    // Load module of Python code
    fprintf(stderr, "loading: %s\n", python_module_name);
    pName = PyUnicode_DecodeFSDefault(python_module_name);
    r->pModule_radae_rx = PyImport_Import(pName);
    check_error(r->pModule_radae_rx, "importing", python_module_name);
    Py_DECREF(pName);

    // Find class and create an instance
    pClass = PyObject_GetAttrString(r->pModule_radae_rx, "radae_rx");
    check_error(pClass, "finding class", "radae_rx");
    pArgs = Py_BuildValue("(s)", "model19_check3/checkpoints/checkpoint_epoch_100.pth");
    float foff_err = 0.0;
    if (r->flags & RADE_FOFF_TEST) foff_err = 10.0;
    int verbose = 2;
    if (r->flags & RADE_VERBOSE_0) verbose = 0;
    pkwArgs = Py_BuildValue("{s:i,s:f,s:i}", "bypass_dec", r->flags & RADE_USE_C_DECODER, "foff_err", foff_err, "v", verbose);
    check_error(pkwArgs, "setting up pkwArgs", "");
    r->pInst_radae_rx = PyObject_Call(pClass, pArgs, pkwArgs);
    check_error(r->pInst_radae_rx, "Creating instance of class", "radae_rx");
    Py_DECREF(pClass);
    Py_DECREF(pArgs);
    Py_DECREF(pkwArgs);

    r->n_features_out = (int)call_getter(r->pInst_radae_rx, "get_n_features_out");
    r->n_floats_out = (int)call_getter(r->pInst_radae_rx, "get_n_floats_out");
    r->nin_max = (int)call_getter(r->pInst_radae_rx, "get_nin_max");
    r->nin = (int)call_getter(r->pInst_radae_rx, "get_nin");
    fprintf(stderr, "n_features_out: %d n_floats_out: %d nin_max: %d nin: %d\n", (int)r->n_features_out, (int)r->n_floats_out, (int)r->nin_max, (int)r->nin);
        
    r->pMeth_radae_rx = PyObject_GetAttrString(r->pInst_radae_rx, do_radae_rx_meth_name);
    check_error(r->pMeth_radae_rx, "finding",  do_radae_rx_meth_name);
    check_callable(r->pMeth_radae_rx, do_radae_rx_meth_name, "not callable");

    r->pArgs_radae_rx = PyTuple_New(2);

    // 1st Python function arg - input numpy array of csingle rx samples
    r->rx_in = (RADE_COMP*)malloc(sizeof(RADE_COMP)*r->nin_max);
    assert(r->rx_in != NULL);
    pValue = PyArray_SimpleNewFromData(1, &r->nin_max, NPY_CFLOAT, r->rx_in);
    check_error(pValue, "setting up numpy array", "buffer_complex");
    PyTuple_SetItem(r->pArgs_radae_rx, 0, pValue);

    // 2nd Python arg - output numpy array of float features (Python core dec) or latents (C core decoder)
    r->floats_out = (float*)malloc(sizeof(float)*r->n_floats_out);
    assert(r->floats_out != NULL);
    pValue = PyArray_SimpleNewFromData(1, &r->n_floats_out, NPY_FLOAT, r->floats_out);
    check_error(pValue, "setting up numpy array", "floats_out");
    PyTuple_SetItem(r->pArgs_radae_rx, 1, pValue);
 
    r->pMeth_sum_uw_errors = PyObject_GetAttrString(r->pInst_radae_rx, sum_uw_errors_meth_name);
    check_error(r->pMeth_sum_uw_errors, "finding", sum_uw_errors_meth_name);
    check_callable(r->pMeth_sum_uw_errors, sum_uw_errors_meth_name, "not callable");

    if (r->flags & RADE_USE_C_DECODER) {
      if (init_radedec(&r->dec_model, radedec_arrays, r->num_features*RADE_FRAMES_PER_STEP) != 0) {
        fprintf(stderr, "Error initialising built-in C decoder model\n");
        exit(1);        
      }
      rade_init_decoder(&r->dec_state);
    }
    return 0;
}

void rade_rx_close(struct rade *r) {
  Py_DECREF(r->pArgs_radae_rx);
  Py_DECREF(r->pMeth_radae_rx);
  Py_DECREF(r->pMeth_sum_uw_errors);
  Py_DECREF(r->pInst_radae_rx);
  Py_DECREF(r->pModule_radae_rx);

  free(r->floats_out);
  free(r->rx_in);
}

void rade_initialize(void) {
  Py_Initialize();
  main_thread_state = PyEval_SaveThread();
}

void rade_finalize(void) {
  PyEval_RestoreThread(main_thread_state);
  int ret = Py_FinalizeEx();
  if (ret < 0) {
    fprintf(stderr, "Error with Py_FinalizeEx()\n");
  }
}

struct rade *rade_open(char model_file[], int flags) {
  int ret;
  struct rade *r = (struct rade*)malloc(sizeof(struct rade));
  assert(r != NULL);
  r->flags = flags;

  // Acquire the Python GIL (needed for multithreaded use)
  PyGILState_STATE gstate = PyGILState_Ensure();

  // TODO: implement me
  fprintf(stderr, "model file: %s\n", model_file);

  // need import array for numpy
  ret = _import_array();
  fprintf(stderr, "import_array returned: %d\n", ret);
  
  // TODO a better way of handling these constants, e.g. read from model
  r->auxdata = 1;
  r->nb_total_features = 36;
  r->num_used_features = 20;
  r->num_features = r->num_used_features + r->auxdata;     

  rade_tx_open(r);
  rade_rx_open(r);
  assert(r->n_features_in == r->n_features_out);

  // Release Python GIL
  PyGILState_Release(gstate);
 
  return r;
}

void rade_close(struct rade *r) {
  // Acquire the Python GIL (needed for multithreaded use)
  PyGILState_STATE gstate = PyGILState_Ensure();

  rade_tx_close(r);
  rade_rx_close(r);

  // Release Python GIL
  PyGILState_Release(gstate);
}

int rade_version(void) { return VERSION; }
int rade_n_tx_out(struct rade *r) { assert(r != NULL); return (int)r->Nmf; }
int rade_n_tx_eoo_out(struct rade *r) { assert(r != NULL); return (int)r->Neoo; }
int rade_nin_max(struct rade *r) { assert(r != NULL); return r->nin_max; }
int rade_nin(struct rade *r) { assert(r != NULL); return r->nin; }

int rade_n_features_in_out(struct rade *r) {
  assert(r != NULL); 
  return r->n_features_in; 
}

int rade_tx(struct rade *r, RADE_COMP tx_out[], float floats_in[]) {
  assert(r != NULL);
  assert(floats_in != NULL);
  assert(tx_out != NULL);

  // Acquire the Python GIL (needed for multithreaded use)
  PyGILState_STATE gstate = PyGILState_Ensure();

  if (r->flags & RADE_USE_C_ENCODER) {
    // sanity check: need integer number of feature vecs
    assert(r->n_features_in % r->nb_total_features == 0);
    int n_feature_vecs = r->n_features_in/r->nb_total_features;
    // sanity check: integer number of core_encoder calls
    assert(n_feature_vecs % RADE_FRAMES_PER_STEP == 0);
    int n_core_encoder = n_feature_vecs/RADE_FRAMES_PER_STEP;
    int input_dim = r->num_features*RADE_FRAMES_PER_STEP;
    float features[input_dim];
    // TODO: need a way to configure arch for testing (and/or automagically detect using opus logic)
    int arch = 0;
    int bottleneck = 3;

    //fprintf(stderr,"n_feature_vecs: %d n_core_encoder: %d input_dim: %d\n", n_feature_vecs, n_core_encoder, input_dim);

    for(int c=0; c<n_core_encoder; c++) {
        for (int i=0; i<RADE_FRAMES_PER_STEP; i++) {
            for(int j=0; j<r->num_used_features; j++)
                features[i*r->num_features+j] = floats_in[(c*RADE_FRAMES_PER_STEP+i)*r->nb_total_features+j];
            if (r->auxdata)
                features[i*r->num_features+r->num_used_features] = -1.0;
        }
        rade_core_encoder(&r->enc_state, &r->enc_model, &r->floats_in[RADE_LATENT_DIM*c], features, arch, bottleneck);
    }
  } else {
    memcpy(r->floats_in, floats_in, sizeof(float)*(r->n_floats_in));
  }
  PyObject_CallObject(r->pMeth_radae_tx, r->pArgs_radae_tx);
  memcpy(tx_out, r->tx_out, sizeof(RADE_COMP)*(r->Nmf));

  // Release Python GIL
  PyGILState_Release(gstate);

  return r->Nmf;
}

int rade_tx_eoo(struct rade *r, RADE_COMP tx_eoo_out[]) {
  assert(r != NULL);
  assert(tx_eoo_out != NULL);

  // Acquire the Python GIL (needed for multithreaded use)
  PyGILState_STATE gstate = PyGILState_Ensure();

  PyObject_CallObject(r->pMeth_radae_tx_eoo, r->pArgs_radae_tx_eoo);
  memcpy(tx_eoo_out, r->tx_eoo_out, sizeof(RADE_COMP)*(r->Neoo));

  // Release Python GIL
  PyGILState_Release(gstate);

  return r->Neoo;
}

int rade_rx(struct rade *r, float features_out[], RADE_COMP rx_in[]) {
  PyObject *pValue;
  assert(r != NULL);
  assert(features_out != NULL);
  assert(rx_in != NULL);

  // Acquire the Python GIL (needed for multithreaded use)
  PyGILState_STATE gstate = PyGILState_Ensure();

  memcpy(r->rx_in, rx_in, sizeof(RADE_COMP)*(r->nin));
  pValue = PyObject_CallObject(r->pMeth_radae_rx, r->pArgs_radae_rx);
  check_error(pValue, "return value", "from do_rx_radae");
  long valid_out = PyLong_AsLong(pValue);

  if (valid_out) {
    if (r->flags & RADE_USE_C_DECODER) {
      // sanity check: need integer number of latent vecs
      assert(r->n_floats_out % RADE_LATENT_DIM == 0);
      int n_latent_vecs = r->n_floats_out/RADE_LATENT_DIM;
      int output_dim = r->num_features*RADE_FRAMES_PER_STEP;
      float features[output_dim];

      // zero out unused ends of feature vecs
      for(int i=0; i<r->n_features_out; i++) features_out[i] = 0.0;

      // TODO: need a way to configure arch for testing (and/or automagically detect using opus logic)
      int arch = 0;
      //fprintf(stderr,"n_latent_vecs: %d output_dim: %d\n", n_latent_vecs, output_dim);

      int uw_errors = 0;
      for(int c=0; c<n_latent_vecs; c++) {
        rade_core_decoder(&r->dec_state, &r->dec_model, features, &r->floats_out[RADE_LATENT_DIM*c], arch);
        for (int i=0; i<RADE_FRAMES_PER_STEP; i++) {
          for(int j=0; j<r->num_used_features; j++)
            features_out[(c*RADE_FRAMES_PER_STEP+i)*r->nb_total_features+j] = features[i*r->num_features + j];
        }
        // just use first aux data symbol for each set of RADE_FRAMES_PER_STEP, as they are all the same decoded value
        if (r->auxdata) {
          if (features[r->num_used_features] > 0) uw_errors++;
          //fprintf(stderr, "aux_symb: %f uw_errors: %d\n", features[r->num_used_features],uw_errors);
        }
      }

      // write number of errors in aux bits back to radae_rxe instance for use by state machine
      if (r->auxdata) {
        PyObject *pArgs_sum_uw_errors = Py_BuildValue("(i)", uw_errors);
        PyObject_CallObject(r->pMeth_sum_uw_errors, pArgs_sum_uw_errors);
        Py_DECREF(pArgs_sum_uw_errors);
      }
    }
    else {
      assert(r->n_floats_out == r->n_features_out);
      memcpy(features_out, r->floats_out, sizeof(float)*(r->n_floats_out));
    }
  }

  // sample nin so we have an updated copy
  r->nin = (int)call_getter(r->pInst_radae_rx, "get_nin");

  // Release Python GIL
  PyGILState_Release(gstate);

  if (valid_out)
    return r->n_features_out;
  else
    return 0;
}

int rade_sync(struct rade *r) {
  assert(r != NULL);

  // Acquire the Python GIL (needed for multithreaded use)
  PyGILState_STATE gstate = PyGILState_Ensure();

  int result = (int)call_getter(r->pInst_radae_rx, "get_sync");

  // Release Python GIL
  PyGILState_Release(gstate);

  return result;
}

// TODO: we need a float getter
float rade_freq_offset(struct rade *r) {
  assert(r != NULL);
  return 0;
}

RADE_EXPORT int rade_snrdB_3k_est(struct rade *r) {
  assert(r != NULL);

  // Acquire the Python GIL (needed for multithreaded use)
  PyGILState_STATE gstate = PyGILState_Ensure();

  int result = (int)call_getter(r->pInst_radae_rx, "get_snrdB_3k_est");

  // Release Python GIL
  PyGILState_Release(gstate);

  return result;
}
