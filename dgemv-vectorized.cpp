#include <omp.h>
#include <cstddef>


const char* dgemv_desc = "Vectorized implementation of matrix-vector multiply.";

/*
 * This routine performs a dgemv operation
 * Y :=  A * X + Y
 * where A is n-by-n matrix stored in row-major format, and X and Y are n by 1 vectors.
 * On exit, A and X maintain their input values.
 */
void my_dgemv(int n, double* A, double* x, double* y) {

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        const double* Ai = A + static_cast<std::size_t>(i) * n;

        double sum = 0.0;

        // Vectorize the inner dot product for better CPU throughput
        #pragma omp simd reduction(+:sum)
        for (int j = 0; j < n; ++j) {
            sum += Ai[j] * x[j];
        }

        // Y := A*X + Y
        y[i] += sum;
    }
}
