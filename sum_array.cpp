//
// Copyright (C) 2025, E. Wes Bethel, All Rights Reserved.
// For educational use only.
// 
#include <iostream>

int 
sum_array(int N, int A[])
{
   // input consists of an integer N and an array A[] of size N
   // your job: write code to compute the sum of all values in A[]
   // and return that sum to the caller
   int sum = 0;

   for(int i = 0; i < N; i++){
      sum += A[i];
   }
   // Put your code here, return the correct result
   return sum;
}


int main(int ac, char*av[])
{
   int N=(1<<10);
   int A[N];

   // initialize A
   for (int i=0;i<N;i++)
      A[i] = i;

   int total = sum_array(N, A);

   std::cout << "Sum of all " << N << " items in A[] is: " << total << std::endl;

   if (total == (N*(N-1)/2))
      std::cout << "Correct check succeeds, you computed the correct answer." << std::endl;
   else
      std::cout << "Correct check fails, you didn't compute the correct answer." << std::endl;

}
