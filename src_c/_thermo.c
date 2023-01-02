// Copyright (c) 2022-2023 Blecic and Cubillos
// chemcat is open-source software under the GPL-2.0 license (see LICENSE)

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <float.h>
#include <math.h>

#include "ind.h"
#include "newton_raphson.h"


PyDoc_STRVAR(
    gibbs_energy_minimizer__doc__,
    "Gibbs free energy minimizer using Newton-Raphson algorithm."
);


static PyObject *gibbs_energy_minimizer(PyObject *self, PyObject *args){
    PyArrayObject
        *stoich_vals, *b0, *pilag, *abundances, *max_abundances,
        *h_ts, *mu, *x, *delta_ln_y;
    double
        temperature, abundance, total_abundance, tolx, tolf,
        lambda, tmp_val, corr, err, maxerr;
    int nspecies, nequations;

    // Load inputs:
    if (!PyArg_ParseTuple(args, "iiOOdOOOOdOOOdd",
            &nspecies, &nequations, &stoich_vals, &b0,
            &temperature, &h_ts,
            &pilag, &abundances, &max_abundances, &total_abundance,
            &mu, &x, &delta_ln_y, &tolx, &tolf))
        return NULL;

    double delta_ln_ybar = 0.0,  // Delta_ybar / ybar = xbar/ybar - 1
        relative_abundance_tol = 5.0e-6;
    int i, j, nr_status, iwk,
        k = 0,
        ntrial = 1,
        nmin_trials = 5,
        nmax_trials = 300;

    while (k < nmax_trials){
        k += 1;
        // Chemical potential (mu) at current {VMR,T,p}:
        for (j=0; j<nspecies; j++){
            if (INDd(abundances,j) <= 0)
                abundance = DBL_MIN;
            else
                abundance = INDd(abundances,j);
            INDd(mu,j) = INDd(h_ts,j) + log(abundance/total_abundance);
        }

        // Single Newton-Raphson iteration:
        nr_status = newton(
            nequations, nspecies, stoich_vals, pilag, delta_ln_ybar, abundances,
            mu, b0, total_abundance, ntrial, x, tolx, tolf);
        if (nr_status == 1){
            return Py_BuildValue("i", 1);
        }

        // Re-use variable, 1=OK, 0=There are NaNs in x
        nr_status = 1;
        for (i=0; i<nequations; i++){
            nr_status &= isnormal(INDd(x,i));
        }
        // The values of x seem to vary unbounded, which could lead to
        // take values above +/- DBL_MAX and break the code.
        // Unless there is a way to constrain their growth, my current
        // and very janky solution is to reset x:
        if (nr_status == 0)
            for (j=0; j<nequations; j++)
                INDd(x,j) = 0.0;

        // Update values of pi lagrange multipliers and Delta ln(ybar)
        for (j=0; j<nequations-1; j++)
            INDd(pilag,j) = INDd(x,j);
        delta_ln_ybar = INDd(x,(nequations-1));

        // Compute correction to abundances, Eq. (14) of White (1958):
        for (i=0; i<nspecies; i++){
            INDd(delta_ln_y,i) = delta_ln_ybar - INDd(mu,i);
            for (j=0; j<nequations-1; j++){
                INDd(delta_ln_y,i) += IND2i(stoich_vals,i,j) * INDd(pilag,j);
            }
        }

        // Compute lambda factor:
        lambda = 2.0 / fabs(5.0*delta_ln_ybar);
        for (i=0; i<nspecies; i++){
            if (INDd(abundances,i)/total_abundance > 1.0e-8){
                if (lambda > 2.0 / fabs(INDd(delta_ln_y,i)))
                    lambda = 2.0 / fabs(INDd(delta_ln_y,i));
            }
            else if (INDd(delta_ln_y,i) >= 0.0){
                if (INDd(abundances,i)/total_abundance > 10.0){
                    tmp_val = fabs(
                        log(INDd(abundances,i)/total_abundance - 10.0)
                        / (INDd(delta_ln_y,i) - delta_ln_ybar));
                    if (tmp_val < lambda)
                        lambda = tmp_val;
                }
            }
        }

        if (lambda > 1.0)
            lambda = 1.0;

        // Update total_abundance and abundances
        corr = lambda * delta_ln_ybar;
        if (corr < - 0.4)
            corr = -0.4;
        if (corr > 0.4)
            corr = 0.4;
        total_abundance *= exp(corr);

        for (i=0; i<nspecies; i++){
            corr = lambda * INDd(delta_ln_y,i);
            if (INDd(abundances,i)/total_abundance > 1.0e-08){
                if (corr < -2.0)
                    corr = -2.0;
                else if (corr > 2.0)
                    corr = 2.0;
            } else{  // Maximum correction factor < 1e50:
                if (corr < -115.0)
                    corr = -115.0;
                else if (corr > 115.0)
                    corr = 115.0;
            }

            // Go half-way to max if proposed step exceeds max:
            tmp_val = INDd(abundances,i) * exp(corr);
            if (tmp_val <= INDd(max_abundances,i))
                INDd(abundances,i) = tmp_val;
            else
                INDd(abundances,i) =
                    0.5 * (INDd(abundances,i) + INDd(max_abundances,i));
        }

        // Convergence:
        tmp_val = 0.0;
        for (i=0; i<nspecies; i++)
            tmp_val += INDd(abundances,i);
        iwk = 0;
        maxerr = 0.0;
        for (i=0; i<nspecies; i++){
            err = INDd(abundances,i) * fabs(INDd(delta_ln_y,i)) / tmp_val;
            if (err <= relative_abundance_tol)
                iwk += 1;
            if (err > maxerr)
                maxerr = err;
        }

        err = total_abundance * fabs(delta_ln_ybar) / tmp_val;
        if (err <= relative_abundance_tol)
            iwk += 1;
        if (err > maxerr)
            maxerr = err;

        if (iwk == nspecies+1 && k >= nmin_trials)
            break;
    }
    return Py_BuildValue("i", 0);
}


/* The module doc string */
PyDoc_STRVAR(
    thermo__doc__,
    "Wrapper for C extensions."
);


/* A list of all the methods defined by this module. */
static PyMethodDef thermo_methods[] = {
    {
        "gibbs_energy_minimizer",
        gibbs_energy_minimizer,
        METH_VARARGS,
        gibbs_energy_minimizer__doc__
    },
    {NULL, NULL, 0, NULL}    /* sentinel */
};


/* Module definition for Python 3. */
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_thermo",
    thermo__doc__,
    -1,
    thermo_methods
};

/* When Python 3 imports a C module named 'X' it loads the module */
/* then looks for a method named "PyInit_"+X and calls it.        */
PyObject *PyInit__thermo (void) {
    PyObject *module = PyModule_Create(&moduledef);
    import_array();
    return module;
}
