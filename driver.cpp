#include <vector>
#include <cstdio>

// declare your function
void my_dgemv(int n, double* A, double* x, double* y);

int main() {
    int n = 4;
    std::vector<double> A(n*n), x(n), y(n, 0.0);

    // Fill A as identity, x = [1,2,3,4], y starts at 0
    for (int i=0;i<n;i++) for (int j=0;j<n;j++) A[i*n+j] = (i==j)?1.0:0.0;
    for (int i=0;i<n;i++) x[i] = i+1;

    my_dgemv(n, A.data(), x.data(), y.data()); // y := A*x + y -> y should become x

    for (int i=0;i<n;i++) std::printf("y[%d]=%.1f\n", i, y[i]);
    return 0;
}
