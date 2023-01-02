// Copyright (c) 2022-2023 Blecic and Cubillos
// chemcat is open-source software under the GPL-2.0 license (see LICENSE)

void usrfun(
    int nelements,
    PyArrayObject *pilag,
    double delta_ln_ybar,
    int nspec,
    PyArrayObject *stoich_vals,
    PyArrayObject *abundances,
    PyArrayObject *mu,
    PyArrayObject *b0,
    double total_abundance,
    PyArrayObject *x,
    PyArrayObject *fvec,
    PyArrayObject *fjac){

    int i, j, k;
    double s1, s2, s3;

    // Lagrange multipliers (pi) from previous iteration:
    for (i=0; i<nelements; i++)
        INDd(pilag,i) = INDd(x,i);

    // delta_ln_ybar is delta_ybar/ybar = xbar/ybar - 1 = u - 1:
    delta_ln_ybar = INDd(x,nelements);

    // Evaluate the set of Equations (18) of White (1958):
    // Evaluate Eq. (4) of White (1958), using x_i from Eq. (14):
    //     sum_i (a_ij * x_i) - b_j = 0
    for (j=0; j<nelements; j++){
        s1 = 0.0;  // sum_ik (r_kj * pi_k)
        s2 = 0.0;  // sum_i (a_ij * y_i)
        s3 = 0.0;  // sum_i (a_ij * mu_i * y_i) = sum_i (a_ij * f_i)
        for (i=0; i<nspec; i++){
            for (k=0; k<nelements; k++)
                s1 += IND2i(stoich_vals,i,j)
                    * IND2i(stoich_vals,i,k)
                    * INDd(abundances,i) * INDd(pilag,k);
            s2 += IND2i(stoich_vals,i,j) * INDd(abundances,i);
            s3 += IND2i(stoich_vals,i,j) * INDd(abundances,i) * INDd(mu,i);
        }
        INDd(fvec,j) = s1 + s2*(1+delta_ln_ybar) - INDd(b0,j) - s3;
    }
    // Evaluate Eq. (3) of White (1958), using x_i from Eq. (14):
    //     x_bar - sum_i (x_i) = 0
    s1 = 0.0;
    s2 = 0.0;
    s3 = 0.0;
    for (i=0; i<nspec; i++){
        for (k=0; k<nelements; k++)
            s1 += IND2i(stoich_vals,i,k) * INDd(abundances,i) * INDd(pilag,k);
        s2 += INDd(abundances,i);
        s3 += INDd(abundances,i) * INDd(mu,i);
    }
    INDd(fvec,nelements) = s1 + (s2-total_abundance)*(1.0+delta_ln_ybar) - s3;

    // Now, evaluate the Jacobian:
    for (j=0; j<nelements; j++){
        for (k=0; k<nelements; k++){
            IND2d(fjac,j,k) = 0.0;
            for (i=0; i<nspec; i++){
                IND2d(fjac,j,k) +=
                    IND2i(stoich_vals,i,j)
                    * IND2i(stoich_vals,i,k)
                    * INDd(abundances,i);
            }
        }
        IND2d(fjac,j,nelements) = 0.0;
        for (i=0; i<nspec; i++){
            IND2d(fjac,j,nelements) +=
                 IND2i(stoich_vals,i,j) * INDd(abundances,i);
        }
    }
    for (k=0; k<nelements; k++){
        IND2d(fjac,nelements,k) = 0.0;
        for (i=0; i<nspec; i++){
            IND2d(fjac,nelements,k) +=
                IND2i(stoich_vals,i,k) * INDd(abundances,i);
        }
    }
    IND2d(fjac,nelements,nelements) = 0.0;
    for (i=0; i<nspec; i++){
        IND2d(fjac,nelements,nelements) += INDd(abundances,i);
    }
    IND2d(fjac,nelements,nelements) -= total_abundance;

    return;
}


/* Following Numerical Recipes Section 2.3.1
   Performing the LU Decomposition

Replaces matrix a by the LU decomposition of a rowwise permutation of itself.
On output, it is arranged as in equation (2.3.14).

The indx output array records the row permutation effected by
the partial pivoting. */
int ludcmp(
    PyArrayObject *a,
    int n,
    PyArrayObject *indx){

    int imax=0;
    int i, j, k;
    double d = 1.0;  // parity
    double eps = 1e-40;
    double temp, big;
    PyArrayObject *vv;
    npy_intp size[1];

    size[0] = n;
    vv = (PyArrayObject *) PyArray_SimpleNew(1, size, NPY_DOUBLE);

    // Store implicit scaling of each row:
    for (i=0; i<n; i++){
        big = 0.0;  
        for (j=0; j<n; j++){
            if (fabs(IND2d(a,i,j)) > big)
                big = fabs(IND2d(a,i,j));
        }
        // Singular matrix in ludcmp:
        if (big == 0.0){
            Py_DECREF(vv);
            return 1;
        }
        INDd(vv,i) = 1.0 / big;
    }

    for (k=0; k<n; k++){
        // Search for largest pivot element:
        big = 0.0;
        for (i=k; i<n; i++){
            temp = INDd(vv,i) * fabs(IND2d(a,i,k));
            // Is the figure of merit for the pivot better than best so far?
            if (temp >= big){
                big = temp;
                imax = i;
            }
        }

        // Do we need to interchange rows?
        if (k != imax){
            for (j=0; j<n; j++){
                temp = IND2d(a,imax,j);
                IND2d(a,imax,j) = IND2d(a,k,j);
                IND2d(a,k,j) = temp;
            }
            d = -d;
            INDd(vv,imax) = INDd(vv,k);
        }

        INDi(indx,k) = imax;
        if (IND2d(a,k,k) == 0.0)
            IND2d(a,k,k) = eps;
        /* If the pivot element is zero, the matrix is singular.
           For some applications on singular matrices, it is desirable to
           substitute eps for zero. */
        // Divide by the pivot element:
        for (i=k+1; i<n; i++){
            temp = IND2d(a,i,k) /= IND2d(a,k,k);
            for (j=k+1; j<n; j++)
                IND2d(a,i,j) -= temp * IND2d(a,k,j);
        }
    }
    Py_DECREF(vv);
    return 0;
}


/* Following Numerical Recipes Section 2.3.1
   Performing the LU Decomposition

Solve the set of n linear equations A*x = b using the stored LU
decomposition of A.  The solution vector x overwrites the input b.
The input indx is the permutation vector returned by ludcmp. */
void solve(
    PyArrayObject *a,
    int n,
    PyArrayObject *indx,
    PyArrayObject *b){
    int i, ii=0, ip, j;
    double sum;

    // When ii is set to a positive value, it will become the index
    // of the first nonvanishing element of b. We now do
    // the forward substitution, equation (2.3.6).
    for (i=0; i<n; i++){
        ip = INDi(indx,i);
        sum = INDd(b,ip);
        INDd(b,ip) = INDd(b,i);
        if (ii != 0){
            for (j=ii-1; j<i; j++)
                sum -= IND2d(a,i,j) * INDd(b,j);
        }
        // A nonzero element was encountered, so from now on we will
        // have to do the sums in the loop above.
        else if (sum != 0.0){
            ii = i+1;
        }
        INDd(b,i) = sum;
    }

    // Now we do the back-substitution, equation (2.3.7):
    for (i=n-1; i>=0; i--){
        sum = INDd(b,i);
        for (j=i+1; j<n; j++)
            sum -= IND2d(a,i,j) * INDd(b,j);
        // Store a component of the solution vector X.
        INDd(b,i) = sum / IND2d(a,i,i);
    }
    return;
}


/* Following Numerical Recipes Section 2.5
   Iterative Improvement of a Solution to Linear Equations

Improves a solution vector x of the linear set of equations A*x = b.
The vectors b and x are input.  On output, x is modified, to an improved
set of values. alud and indx are outputs returned by ludcmp() of the LU
decomposition of a. */
void mprove(
    PyArrayObject *a,
    PyArrayObject *alud,
    int n,
    PyArrayObject *indx,
    PyArrayObject *b,
    PyArrayObject *x){
    int i, j;
    double sdp;
    PyArrayObject *r;
    npy_intp size[1];
    size[0] = n;

    r = (PyArrayObject *) PyArray_SimpleNew(1, size, NPY_DOUBLE);

    /* Calculate the right-hand side, accumulating the residual */
    for (i=0; i<n; i++){
        sdp = -INDd(b,i);
        for (j=0; j<n; j++)
            sdp += IND2d(a,i,j) * INDd(x,j);
        INDd(r,i) = sdp;
    }

    // Solve for the error term and subtract it from the old solution.
    solve(alud, n, indx, r);
    for (i=0; i<n; i++)
        INDd(x,i) -= INDd(r,i);

    Py_DECREF(r);
    return;
}


// Following Numerical Recipes Section 9.6
// Newton-Raphson Method for Nonlinear Systems of Equations
static int newton(
    int nequations,
    int nspecies,
    PyArrayObject *stoich_vals,
    PyArrayObject *pilag,
    double delta_ln_ybar,
    PyArrayObject *abundances,
    PyArrayObject *mu,
    PyArrayObject *b0,
    double total_abundance,
    int ntrial,
    PyArrayObject *x,
    double tolx,
    double tolf
    ){

    PyArrayObject *p, *indx, *fjac, *fvec, *fjac_save, *fvec_save;
    int i, j, k, code;
    double errf, errx;
    npy_intp dims[2], size[1];

    dims[0] = dims[1] = size[0] = nequations;
    fvec = (PyArrayObject *) PyArray_SimpleNew(1, size, NPY_DOUBLE);
    fjac = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    fvec_save = (PyArrayObject *) PyArray_SimpleNew(1, size, NPY_DOUBLE);
    fjac_save = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    p = (PyArrayObject *) PyArray_SimpleNew(1, size, NPY_DOUBLE);
    indx = (PyArrayObject *) PyArray_SimpleNew(1, size, NPY_DOUBLE);

    for (i=0; i<nequations; i++)
        INDi(indx,i) = 0;

    for (k=0; k<ntrial; k++){
        // User def supplies function values at x in fvec:
        usrfun(
            nequations-1, pilag, delta_ln_ybar, nspecies,
            stoich_vals, abundances, mu,
            b0, total_abundance, x, fvec, fjac);

        // Save fvec and fjac:
        for (i=0; i<nequations; i++){
            INDd(fvec_save,i) = -INDd(fvec,i);
            for (j=0; j<nequations; j++)
                IND2d(fjac_save,i,j) = IND2d(fjac,i,j);
        }

        // Check function convergence:
        errf = 0.0;
        for (i=0; i<nequations; i++){
            errf += fabs(INDd(fvec,i));
        }
        if (errf <= tolf){
            Py_DECREF(fvec);
            Py_DECREF(fjac);
            Py_DECREF(fvec_save);
            Py_DECREF(fjac_save);
            Py_DECREF(p);
            Py_DECREF(indx);
            return 0;
        }

        // Right-hand side of linear equations.
        for (i=0; i<nequations; i++)
            INDd(p,i) = -INDd(fvec,i);

        // Solve linear equations using LU decomposition.
        code = ludcmp(fjac, nequations, indx);
        if (code == 1){
            Py_DECREF(fvec);
            Py_DECREF(fjac);
            Py_DECREF(fvec_save);
            Py_DECREF(fjac_save);
            Py_DECREF(p);
            Py_DECREF(indx);
            return 1;
        }
        solve(fjac, nequations, indx, p);
        mprove(fjac_save, fjac, nequations, indx, fvec_save, p);

        /* Check root convergence */
        errx = 0.0;
        for (i=0; i<nequations; i++){  // Update solution.
            errx += fabs(INDd(p,i));
            INDd(x,i) += INDd(p,i);
        }
        if (errx <= tolx){
            Py_DECREF(fvec);
            Py_DECREF(fjac);
            Py_DECREF(fvec_save);
            Py_DECREF(fjac_save);
            Py_DECREF(p);
            Py_DECREF(indx);
            return 0;
        }
    }
    Py_DECREF(fvec);
    Py_DECREF(fjac);
    Py_DECREF(fvec_save);
    Py_DECREF(fjac_save);
    Py_DECREF(p);
    Py_DECREF(indx);
    return 0;
}
