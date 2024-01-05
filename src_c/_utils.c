//# Copyright (c) 2022-2024 Blecic and Cubillos
// chemcat is open-source software under the GPL-2.0 license (see LICENSE)

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <float.h>

#include "ind.h"


PyDoc_STRVAR(
    heat__doc__,
    "Compute the (unitless) molar heat capacity cp/R\n\
     (i.e., divided by the universal gas constant R)\n\
     for the current species at the requested temperatures.\n\
     This routine follows the CEA parameterization from:\n\
     https://ntrs.nasa.gov/citations/20020085330.");


static PyObject *heat(PyObject *self, PyObject *args){
    PyArrayObject *temperature, *a_coeffs, *t_coeffs, *heat_capacity;
    double temp;
    int n_tcoeff, ntemp, i, j;
    npy_intp dims[1];

    /* Load inputs: */
    if (!PyArg_ParseTuple(
            args,
            "OOO",
            &temperature, &a_coeffs, &t_coeffs))
        return NULL;

    dims[0] = ntemp = (int)PyArray_DIM(temperature, 0);
    heat_capacity = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_DOUBLE);

    for (i=0; i<ntemp; i++){
        temp = INDd(temperature,i);
        n_tcoeff = (int)PyArray_DIM(t_coeffs, 0);
        for (j=0; j<n_tcoeff; j++){
            if (temp < INDd(t_coeffs,(j+1)))
                break;
        }
        // Eq. (1) from https://ntrs.nasa.gov/citations/20020085330
        INDd(heat_capacity,i) =
            + IND2d(a_coeffs,j,0) * pow(temp,-2.0)
            + IND2d(a_coeffs,j,1) * pow(temp,-1.0)
            + IND2d(a_coeffs,j,2)
            + IND2d(a_coeffs,j,3) * temp
            + IND2d(a_coeffs,j,4) * pow(temp,2.0)
            + IND2d(a_coeffs,j,5) * pow(temp,3.0)
            + IND2d(a_coeffs,j,6) * pow(temp,4.0);
    }

    return Py_BuildValue("N", heat_capacity);
}


PyDoc_STRVAR(
    gibbs__doc__,
    "Compute the (unitless) Gibbs free energy G/RT = (H-TS) / RT,\n\
     for this species at the requested temperatures.\n\
     This routine follows the CEA parameterization from:\n\
     https://ntrs.nasa.gov/citations/20020085330.");


static PyObject *gibbs(PyObject *self, PyObject *args){
    PyArrayObject *temperature, *a_coeffs, *b_coeffs, *t_coeffs, *free_energy;
    double temp;
    int n_tcoeff, ntemp, i, j;
    npy_intp dims[1];

    /* Load inputs: */
    if (!PyArg_ParseTuple(
            args,
            "OOOO",
            &temperature, &a_coeffs, &b_coeffs, &t_coeffs))
        return NULL;

    dims[0] = ntemp = (int)PyArray_DIM(temperature, 0);
    free_energy = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_DOUBLE);

    for (i=0; i<ntemp; i++){
        temp = INDd(temperature,i);
        n_tcoeff = (int)PyArray_DIM(t_coeffs, 0);
        for (j=0; j<n_tcoeff; j++){
            if (temp < INDd(t_coeffs,(j+1)))
                break;
        }
        // Eq. (2) - Eq. (3) from https://ntrs.nasa.gov/citations/20020085330
        INDd(free_energy,i) =
            - IND2d(a_coeffs,j,0) * pow(temp,-2.0) * 0.5
            + IND2d(a_coeffs,j,1) / temp * (1.0 + log(temp))
            + IND2d(a_coeffs,j,2) * (1.0 - log(temp))
            - IND2d(a_coeffs,j,3) * temp* 0.5
            - IND2d(a_coeffs,j,4) * pow(temp,2.0) / 6.0
            - IND2d(a_coeffs,j,5) * pow(temp,3.0) / 12.0
            - IND2d(a_coeffs,j,6) * pow(temp,4.0) / 20.0
            + IND2d(b_coeffs,j,0) / temp
            - IND2d(b_coeffs,j,1);
    }

    return Py_BuildValue("N", free_energy);
}


/* The module doc string */
PyDoc_STRVAR(
    utils__doc__,
    "Wrapper for TEA/CEA C utilities."
);


/* A list of all the methods defined by this module. */
static PyMethodDef utils_methods[] = {
    {"heat", heat, METH_VARARGS, heat__doc__},
    {"gibbs", gibbs, METH_VARARGS, gibbs__doc__},
    {NULL, NULL, 0, NULL}    /* sentinel */
};


/* Module definition for Python 3. */
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_utils",
    utils__doc__,
    -1,
    utils_methods
};

/* When Python 3 imports a C module named 'X' it loads the module */
/* then looks for a method named "PyInit_"+X and calls it.        */
PyObject *PyInit__utils (void) {
    PyObject *module = PyModule_Create(&moduledef);
    import_array();
    return module;
}
