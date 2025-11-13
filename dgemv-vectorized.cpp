const char* dgemv_desc = "Vectorized implementation of matrix-vector multiply.";

/*
 * This routine performs a dgemv operation
 * Y :=  A * X + Y
 * where A is n-by-n matrix stored in row-major format, and X and Y are n by 1 vectors.
 * On exit, A and X maintain their input values.
 */
void my_dgemv(int n, double* A, double* x, double* y) {
   // insert your code here: implementation of vectorized vector-matrix multiply
    const double* __restrict a  = A;
    const double* __restrict xv = x;
    double* __restrict yv       = y;

    for (int i = 0; i < n; ++i) {
        double s = 0.0;
        const double* __restrict Ai = a + (long)i * n;

        // Inner dot-product reduction across the row: vectorizes
        #pragma omp simd reduction(+:s)
        for (int j = 0; j < n; ++j) {
            s += Ai[j] * xv[j];
        }
        yv[i] += s;  // Y := A*X + Y
    }
}
