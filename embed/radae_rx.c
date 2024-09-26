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

int main(void) {
    PyObject *pName, *pModule, *pClass, *pInst, *pMeth;
    PyObject *pValue;
    PyObject *pArgs;
    char *python_module_name = "radae_rx";
    char *do_radae_rx_meth_name = "do_radae_rx";
    npy_intp n_floats_out, nin_max, nin;

    Py_Initialize();

    // need import array for numpy
    int ret = _import_array();
    fprintf(stderr, "import_array returned: %d\n", ret);

    // Load module of Python code
    pName = PyUnicode_DecodeFSDefault(python_module_name);
    pModule = PyImport_Import(pName);
    check_error(pModule, "importing", python_module_name);
    Py_DECREF(pName);

    // Find class and creat an instance
    pClass = PyObject_GetAttrString(pModule, "radae_rx");
    check_error(pClass, "finding class", "radae_rx");
    pArgs = Py_BuildValue("(s)", "../model19_check3/checkpoints/checkpoint_epoch_100.pth");
    pInst = PyObject_CallObject(pClass, pArgs);
    check_error(pInst, "Creating instance of class", "radae_rx");
    Py_DECREF(pClass);
    Py_DECREF(pArgs);

    n_floats_out = (int)call_getter(pInst, "get_n_floats_out");
    nin_max = (int)call_getter(pInst, "get_nin_max");
    nin = (int)call_getter(pInst, "get_nin");
    fprintf(stderr, "n_floats_out: %d nin_max: %d nin: %d\n", (int)n_floats_out, (int)nin_max, (int)nin);
        
    pMeth = PyObject_GetAttrString(pInst, do_radae_rx_meth_name);
    check_error(pMeth, "finding",  do_radae_rx_meth_name);
    check_callable(pMeth, do_radae_rx_meth_name, "not callable");

    pArgs = PyTuple_New(2);

    // 1st Python function arg - input numpy array of csingle rx samples
    float buffer_complex[2*nin_max];
    pValue = PyArray_SimpleNewFromData(1, &nin_max, NPY_CFLOAT, buffer_complex);
    check_error(pValue, "setting up numpy array", "buffer_complex");
    PyTuple_SetItem(pArgs, 0, pValue);

    // 2nd Python arg - output numpy array of float features
    float features_out[n_floats_out];
    pValue = PyArray_SimpleNewFromData(1, &n_floats_out, NPY_FLOAT, features_out);
    check_error(pValue, "setting up numpy array", "features_out");
    PyTuple_SetItem(pArgs, 1, pValue);

#ifdef _WIN32
    // Note: freopen() returns NULL if filename is NULL, so
    // we have to use setmode() to make it a binary stream instead.
    _setmode(_fileno(stdin), O_BINARY);
    _setmode(_fileno(stdout), O_BINARY);
#endif // _WIN32

    // We are assuming once args are set up we can make repeat call with the same args, even though
    // data in arrays change
    while(fread(buffer_complex, sizeof(float), 2*nin, stdin) == (size_t)(2*nin)) {
        // do the function call
        pValue = PyObject_CallObject(pMeth, pArgs);
        check_error(pValue, "calling", do_radae_rx_meth_name);
        long valid_out = PyLong_AsLong(pValue);
        if (valid_out) {
            fwrite(features_out, sizeof(float), n_floats_out, stdout);
            fflush(stdout);
        }
        // note time varying, nin must be read before next call to do_radae_rx
        nin = (int)call_getter(pInst, "get_nin");
    }

    Py_DECREF(pArgs);
    Py_DECREF(pMeth);
    Py_DECREF(pInst);
 
    if (Py_FinalizeEx() < 0) {
        return 120;
    }
    return 0;
}
