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

struct rade {
  npy_intp Nmf, Neoo;     
  npy_intp nin, nin_max;   
  npy_intp n_features_in, n_features_out;

  PyObject *pModule_radae_tx;
  PyObject *pFunc_radae_tx, *pArgs_radae_tx;
  float *features_in;
  RADE_COMP *tx_out;
  PyObject *pFunc_radae_tx_eoo, *pArgs_radae_tx_eoo;
  RADE_COMP *tx_eoo_out;

  PyObject *pModule_radae_rx, *pInst_radae_rx;
  PyObject *pMeth_radae_rx, *pArgs_radae_rx;
  float *features_out;
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
    PyObject *pName;
    PyObject *pValue;
    char *python_module_name = "radae_txe";
    char *do_radae_tx_func_name = "do_radae_tx";
    char *do_eoo_func_name = "do_eoo";

    pName = PyUnicode_DecodeFSDefault(python_module_name);
    r->pModule_radae_tx = PyImport_Import(pName);
    check_error(r->pModule_radae_tx, "importing", python_module_name);
    Py_DECREF(pName);

    r->n_features_in = (int)call_getter(r->pModule_radae_tx, "get_nb_floats");
    r->Nmf = (int)call_getter(r->pModule_radae_tx, "get_Nmf");
    r->Neoo = (int)call_getter(r->pModule_radae_tx, "get_Neoo");
    fprintf(stderr, "n_features_in: %d Nmf: %d Neoo: %d\n", (int)r->n_features_in, (int)r->Nmf, (int)r->Neoo);
        
    // RADAE Tx ---------------------------------------------------------

    r->pArgs_radae_tx = PyTuple_New(2);

    r->pFunc_radae_tx = PyObject_GetAttrString(r->pModule_radae_tx, do_radae_tx_func_name);
    check_error(r->pFunc_radae_tx, "finding",  do_radae_tx_func_name);
    check_callable(r->pFunc_radae_tx, do_radae_tx_func_name, "not callable");

    // 1st Python function arg - numpy array of float features
    r->features_in = (float*)malloc(sizeof(float)*r->n_features_in);
    assert(r->features_in != NULL);
    pValue = PyArray_SimpleNewFromData(1, &r->n_features_in, NPY_FLOAT, r->features_in);
    check_error(pValue, "setting up numpy array", "features_in");
    PyTuple_SetItem(r->pArgs_radae_tx, 0, pValue);

    // 2nd Python arg is a numpy array used for output to C
    r->tx_out = (RADE_COMP*)malloc(sizeof(RADE_COMP)*r->Nmf);
    assert(r->tx_out != NULL);
    pValue = PyArray_SimpleNewFromData(1, &r->Nmf, NPY_CFLOAT, r->tx_out);
    check_error(pValue, "setting up numpy array", "tx_out");
    PyTuple_SetItem(r->pArgs_radae_tx, 1, pValue);

    // End of Over --------------------------------------------------------

    r->pFunc_radae_tx_eoo = PyObject_GetAttrString(r->pModule_radae_tx, do_eoo_func_name);
    check_error(r->pFunc_radae_tx_eoo, "finding",  do_eoo_func_name);
    check_callable(r->pFunc_radae_tx_eoo, do_eoo_func_name, "not callable");
    r->pArgs_radae_tx_eoo = PyTuple_New(1);

    // Python arg is a numpy array used for output to C
    r->tx_eoo_out = (RADE_COMP*)malloc(sizeof(RADE_COMP)*r->Neoo);
    assert(r->tx_eoo_out != NULL);
    pValue = PyArray_SimpleNewFromData(1, &r->Neoo, NPY_CFLOAT, r->tx_eoo_out);
    check_error(pValue, "setting up numpy array", "tx_eoo_out");
    PyTuple_SetItem(r->pArgs_radae_tx_eoo, 0, pValue);

    return 0;
}

void rade_tx_close(struct rade *r) {
  // TODO we may need more of these, see if there are any memory leaks
  Py_DECREF(r->pArgs_radae_tx);
  Py_DECREF(r->pFunc_radae_tx);
  Py_DECREF(r->pFunc_radae_tx_eoo);
  Py_DECREF(r->pArgs_radae_tx_eoo);
  Py_DECREF(r->pModule_radae_tx);

  free(r->features_in);
  free(r->tx_out);
  free(r->tx_eoo_out);
}

// returns 0 for success
int rade_rx_open(struct rade *r) {
    PyObject *pName, *pClass;
    PyObject *pValue;
    PyObject *pArgs;
    char *python_module_name = "radae_rxe";
    char *do_radae_rx_meth_name = "do_radae_rx";

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
    r->pInst_radae_rx = PyObject_CallObject(pClass, pArgs);
    check_error(r->pInst_radae_rx, "Creating instance of class", "radae_rx");
    Py_DECREF(pClass);
    Py_DECREF(pArgs);

    r->n_features_out = (int)call_getter(r->pInst_radae_rx, "get_n_features_out");
    r->nin_max = (int)call_getter(r->pInst_radae_rx, "get_nin_max");
    r->nin = (int)call_getter(r->pInst_radae_rx, "get_nin");
    fprintf(stderr, "n_features_out: %d nin_max: %d nin: %d\n", (int)r->n_features_out, (int)r->nin_max, (int)r->nin);
        
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

    // 2nd Python arg - output numpy array of float features
    r->features_out = (float*)malloc(sizeof(float)*r->n_features_out);
    assert(r->features_out != NULL);
    pValue = PyArray_SimpleNewFromData(1, &r->n_features_out, NPY_FLOAT, r->features_out);
    check_error(pValue, "setting up numpy array", "features_out");
    PyTuple_SetItem(r->pArgs_radae_rx, 1, pValue);
 
    return 0;
}

void rade_rx_close(struct rade *r) {
  Py_DECREF(r->pArgs_radae_rx);
  Py_DECREF(r->pMeth_radae_rx);
  Py_DECREF(r->pInst_radae_rx);
  Py_DECREF(r->pModule_radae_rx);

  free(r->features_out);
  free(r->rx_in);
}

struct rade *rade_open(char model_file[]) {
  int ret;
  struct rade *r = (struct rade*)malloc(sizeof(struct rade));
  assert(r != NULL);

  // TODO: implement me
  fprintf(stderr, "model file: %s\n", model_file);
  Py_Initialize();

  // need import array for numpy
  ret = _import_array();
  fprintf(stderr, "import_array returned: %d\n", ret);
  
  fprintf(stderr, "before tx_open()\n");
  rade_tx_open(r);
  fprintf(stderr, "after tx_open()\n");
  rade_rx_open(r);
  assert(r->n_features_in == r->n_features_out);

  return r;
}

void rade_close(struct rade *r) {
  rade_tx_close(r);
  rade_rx_close(r);

  int ret = Py_FinalizeEx();
  if (ret < 0) {
    fprintf(stderr, "Error with Py_FinalizeEx()\n");
  }
}

int rade_version(void) { return VERSION; }
int rade_n_tx_out(struct rade *r) { assert(r != NULL); return (int)r->Nmf; }
int rade_n_tx_eoo_out(struct rade *r) { assert(r != NULL); return (int)r->Neoo; }
int rade_nin_max(struct rade *r) { assert(r != NULL); return r->nin_max; }
int rade_nin(struct rade *r) { assert(r != NULL); return r->nin; }

int rade_n_features_in_out(struct rade *r) {
  assert(r != NULL); 
  return (int)r->n_features_in; 
}

int rade_tx(struct rade *r, RADE_COMP tx_out[], float features_in[]) {
  assert(r != NULL);
  assert(features_in != NULL);
  assert(tx_out != NULL);

  memcpy(r->features_in, features_in, sizeof(float)*(r->n_features_in));
  PyObject_CallObject(r->pFunc_radae_tx, r->pArgs_radae_tx);
  memcpy(tx_out, r->tx_out, sizeof(RADE_COMP)*(r->Nmf));
  return r->Nmf;
}

int rade_tx_eoo(struct rade *r, RADE_COMP tx_eoo_out[]) {
  assert(r != NULL);
  assert(tx_eoo_out != NULL);
  PyObject_CallObject(r->pFunc_radae_tx_eoo, r->pArgs_radae_tx_eoo);
  memcpy(tx_eoo_out, r->tx_eoo_out, sizeof(RADE_COMP)*(r->Neoo));
  return r->Neoo;
}

int rade_rx(struct rade *r, float features_out[], RADE_COMP rx_in[]) {
  PyObject *pValue;
  assert(r != NULL);
  assert(features_out != NULL);
  assert(rx_in != NULL);

  memcpy(r->rx_in, rx_in, sizeof(RADE_COMP)*(r->nin));
  pValue = PyObject_CallObject(r->pMeth_radae_rx, r->pArgs_radae_rx);
  check_error(pValue, "return value", "from do_rx_radae");
  long valid_out = PyLong_AsLong(pValue);
  memcpy(features_out, r->features_out, sizeof(float)*(r->n_features_out));
  // sample nin so we have an updated copy
  r->nin = (int)call_getter(r->pInst_radae_rx, "get_nin");
  if (valid_out)
    return r->n_features_out;
  else
    return 0;
}

int rade_sync(struct rade *r) {
  assert(r != NULL);
  return (int)call_getter(r->pInst_radae_rx, "get_sync");
}

// TODO: we need a float getter
float rade_freq_offset(struct rade *r) {
  assert(r != NULL);
  return 0;
}

