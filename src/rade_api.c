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


int radae_tx_open(struct rade *r) {
    PyObject *pName;
    PyObject *pValue;
    char *python_module_name = "radae_tx";
    char *do_radae_tx_func_name = "do_radae_tx";
    char *do_eoo_func_name = "do_eoo";

    pName = PyUnicode_DecodeFSDefault(python_module_name);
    r->pModule_radae_tx = PyImport_Import(pName);
    check_error(r->pModule_radae_tx, "importing", python_module_name);
    Py_DECREF(pName);

    r->n_features_in = (int)call_getter(r->pModule_radae_tx, "get_nb_floats");
    r->Nmf = (int)call_getter(r->pModule_radae_tx, "get_Nmf");
    r->Neoo = (int)call_getter(r->pModule_radae_tx, "get_Neoo");
    fprintf(stderr, "nb_features_in: %d Nmf: %d Neoo: %d\n", (int)r->n_features_in, (int)r->Nmf, (int)r->Neoo);
        
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

  int ret = Py_FinalizeEx();
  if (ret < 0) {
    fprintf(stderr, "Error with Py_FinalizeEx()\n");
  }
}

struct rade *rade_open(char model_file[]) {
  int ret;
  struct rade *r = (struct rade*)malloc(sizeof(struct rade));
  assert(r != NULL);

  Py_Initialize();

  // need import array for numpy
  ret = _import_array();
  fprintf(stderr, "import_array returned: %d\n", ret);
  
  if (radae_tx_open(r) == 0)
    return r;
  else
    return NULL;
}

void rade_close(struct rade *r) {
   rade_tx_close(r);
}

int rade_version(void) { return VERSION; }

int rade_n_tx_out(struct rade *r) { assert(r != NULL); return (int)r->Nmf; }

int rade_max_nin(struct rade *r) { return 0; }

int rade_n_features_in_out(struct rade *r) {
  assert(r != NULL); 
  return (int)r->n_features_in; 
}

void rade_tx(struct rade *r, RADE_COMP tx_out[], float features_in[]) {
  assert(r != NULL);
  assert(features_in != NULL);
  assert(tx_out != NULL);

  memcpy(r->features_in, features_in, sizeof(float)*(r->n_features_in));
  PyObject_CallObject(r->pFunc_radae_tx, r->pArgs_radae_tx);
  memcpy(tx_out, r->tx_out, sizeof(RADE_COMP)*(r->Nmf));
}

void rade_tx_eoo(struct rade *r, RADE_COMP tx_eoo_out[]) {
  assert(r != NULL);
  assert(tx_out_eoo != NULL);
  PyObject_CallObject(r->pFunc_radae_tx_eoo, r->pArgs_radae_tx_eoo);
  memcpy(tx_eoo_out, r->tx_eoo_out, sizeof(RADE_COMP)*(r->Neoo));
}

int rade_nin(struct rade *r) {
  return 0;
}

int rade_rx(struct rade *r, float features_out[], RADE_COMP rx_in[]) {
  return 0;
}

int rade_sync(struct rade *r) {
  return 0;
}

float rade_freq_offset(struct rade *r) {
  return 0;
}

