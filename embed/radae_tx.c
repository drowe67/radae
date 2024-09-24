/* Top level C program for embedded version of radae_tx.py */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

#ifdef _WIN32
// For _setmode().
#include <io.h>
#include <fcntl.h>
#endif // _WIN32

/* help function to call a Python "getter" function with no arguments that returns a long */
long call_getter(PyObject *pModule, char func_name[]) {
    PyObject *pFunc, *pValue;
    pFunc = PyObject_GetAttrString(pModule, func_name);
    long ret;

    if (pFunc && PyCallable_Check(pFunc)) {
        pValue = PyObject_CallObject(pFunc, NULL);
        if (pValue != NULL) {
            ret = PyLong_AsLong(pValue);
            Py_DECREF(pValue);
        }
        else {
            Py_DECREF(pFunc);
            PyErr_Print();
            fprintf(stderr,"Call to %s failed\n", func_name);
            // TODO when porting to library modify function to return error code, caller shuts down gracefully
            exit(1);
        }
    }
    else {
        if (PyErr_Occurred())
            PyErr_Print();
        fprintf(stderr, "Cannot find function \"%s\"\n", func_name);
        // TODO: fix when ported to library
        exit(1);
    }
    Py_XDECREF(pFunc);

    return ret;
}

int main(void)
{
    PyObject *pName, *pModule, *pFunc;
    PyObject *pValue;
    PyObject *pArgs;
    char *python_name = "radae_tx";
    char *do_radae_tx_func_name = "do_radae_tx";
    char *do_eoo_func_name = "do_eoo";
    npy_intp nb_floats, Nmf, Neoo;

    Py_Initialize();

    // need import array for numpy
    int ret = _import_array();
    fprintf(stderr, "import_array returned: %d\n", ret);

    // name of Python script
    pName = PyUnicode_DecodeFSDefault(python_name);
    /* Error checking of pName left out */
    pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    if (pModule != NULL) {
        nb_floats = (int)call_getter(pModule, "get_nb_floats");
        Nmf = (int)call_getter(pModule, "get_Nmf");
        Neoo = (int)call_getter(pModule, "get_Neoo");
        fprintf(stderr, "nb_floats: %d Nmf: %d Neoo: %d\n", (int)nb_floats, (int)Nmf, (int)Neoo);
        
        pFunc = PyObject_GetAttrString(pModule, do_radae_tx_func_name);
        if (pFunc && PyCallable_Check(pFunc)) {

            pArgs = PyTuple_New(2);

            // 1st Python function arg - numpy array of float features
            float buffer_f32[nb_floats];
            pValue = PyArray_SimpleNewFromData(1, &nb_floats, NPY_FLOAT, buffer_f32);
            if (pValue == NULL) {
                PyErr_Print();
                fprintf(stderr,"Error setting up numpy array for buffer_f32\n");
            }
            PyTuple_SetItem(pArgs, 0, pValue);

            // 2nd Python arg is a numpy array used for output to C
            float tx_out[2*Nmf];
            pValue = PyArray_SimpleNewFromData(1, &Nmf, NPY_CFLOAT, tx_out);
            if (pValue == NULL) {
                PyErr_Print();
                fprintf(stderr,"Error setting up numpy array for tx_out\n");
            }
            PyTuple_SetItem(pArgs, 1, pValue);

#ifdef _WIN32
            // Note: freopen() returns NULL if filename is NULL, so
            // we have to use setmode() to make it a binary stream instead.
            _setmode(_fileno(stdin), O_BINARY);
            _setmode(_fileno(stdout), O_BINARY);
#endif // _WIN32

            // We are assuming once args are set up we can make repeat call with the same args, even though
            // data in arrays changes
            while((unsigned)nb_floats == fread(buffer_f32, sizeof(float), nb_floats, stdin)) {
                // do the function call
                PyObject_CallObject(pFunc, pArgs);
                fwrite(tx_out, 2*sizeof(float), Nmf, stdout);
                fflush(stdout);
            }

            Py_DECREF(pArgs);
        }
        else {
            if (PyErr_Occurred())
                PyErr_Print();
            fprintf(stderr, "Cannot find function \"%s\"\n", do_radae_tx_func_name);
        }
        Py_XDECREF(pFunc);

        // End of Over 
        pFunc = PyObject_GetAttrString(pModule, do_eoo_func_name);
        if (pFunc && PyCallable_Check(pFunc)) {

            pArgs = PyTuple_New(1);

            // Python arg is a numpy array used for output to C
            float eoo_out[2*Neoo];
            pValue = PyArray_SimpleNewFromData(1, &Neoo, NPY_CFLOAT, eoo_out);
            if (pValue == NULL) {
                PyErr_Print();
                fprintf(stderr,"Error setting up numpy array for eoo_out\n");
            }
            PyTuple_SetItem(pArgs, 0, pValue);
            PyObject_CallObject(pFunc, pArgs);
            fwrite(eoo_out, 2*sizeof(float), Neoo, stdout);
            fflush(stdout);
            Py_DECREF(pArgs);
        }
        else {
            if (PyErr_Occurred())
                PyErr_Print();
            fprintf(stderr, "Cannot find function \"%s\"\n", do_eoo_func_name);
        }
        Py_XDECREF(pFunc);
        
        Py_DECREF(pModule);
    }
    else {
        PyErr_Print();
        fprintf(stderr, "Failed to load \"%s\"\n", python_name);
        return 1;
    }
    if (Py_FinalizeEx() < 0) {
        return 120;
    }
    return 0;
}
