// C top level loader for emebd_dec.py

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

#define NARGS 4

int main(int argc, char *argv[])
{
    PyObject *pName, *pModule, *pFunc;

    if (argc < 3) {
    fprintf(stderr,"Usage: %s pythonfile funcname\n", argv[0]);
        return 1;
    }

    Py_Initialize();
    // need import_array for numpy
    int ret = _import_array();
    fprintf(stderr, "import_array returned: %d\n", ret);

    // name of Python script
    pName = PyUnicode_DecodeFSDefault(argv[1]);
    /* Error checking of pName left out */
    pModule = PyImport_Import(pName);

    Py_DECREF(pName);

    if (pModule != NULL) {
        pFunc = PyObject_GetAttrString(pModule, argv[2]);
        /* pFunc is a new reference */

        if (pFunc && PyCallable_Check(pFunc)) {

            // do the function call
            PyObject_CallObject(pFunc, NULL);
        }
        else {
            if (PyErr_Occurred())
                PyErr_Print();
            fprintf(stderr, "Cannot find function \"%s\"\n", argv[2]);
        }
        Py_XDECREF(pFunc);
        Py_DECREF(pModule);
    }
    else {
        PyErr_Print();
        fprintf(stderr, "Failed to load \"%s\"\n", argv[1]);
        return 1;
    }
    if (Py_FinalizeEx() < 0) {
        return 120;
    }
    return 0;
}
