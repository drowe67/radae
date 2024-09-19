#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

#define NARGS 3

int main(int argc, char *argv[])
{
    PyObject *pName, *pModule, *pFunc;
    PyObject *pArgs, *pValue;

    if (argc < 3) {
    fprintf(stderr,"Usage: %s pythonfile funcname mult_arg1 mult_arg2\n", argv[0]);
        return 1;
    }

    Py_Initialize();

    // need import array for numpy
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
            pArgs = PyTuple_New(NARGS);

            // first two args from command line
            pValue = PyLong_FromLong(atol(argv[3]));
            PyTuple_SetItem(pArgs, 0, pValue);
            pValue = PyLong_FromLong(atol(argv[4]));
            PyTuple_SetItem(pArgs, 1, pValue);

            // 3rd Python function arg - set up numpy array
            long dims = 3;
            float array[] = {1.0,2.0,3.0};
            pValue = PyArray_SimpleNewFromData(1, &dims, NPY_FLOAT, array);
            if (pValue == NULL) {
                PyErr_Print();
                fprintf(stderr,"Error setting up numpy array\n");
            }
            PyTuple_SetItem(pArgs, 2, pValue);

            // do the function call
            pValue = PyObject_CallObject(pFunc, pArgs);
            Py_DECREF(pArgs);
            if (pValue != NULL) {
                printf("Result of call: %ld\n", PyLong_AsLong(pValue));
                Py_DECREF(pValue);
            }
            else {
                Py_DECREF(pFunc);
                Py_DECREF(pModule);
                PyErr_Print();
                fprintf(stderr,"Call failed\n");
                return 1;
            }
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
