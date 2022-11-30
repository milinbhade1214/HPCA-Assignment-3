#include <immintrin.h>



void singleThread(int N, int *matA, int *matB, int *output)
{  int x;
 //int B[N*N]; 
 // transpose matrix matB
 for(int j = 0; j < N; ++j)
    for(int i = j; i < N; ++i)
    {  
      int temp= matB[j*N+i];
      matB[j*N+i]=matB[i*N+j];
      matB[i*N+j]=temp;;
      }
 
  assert( N>=4 and N == ( N &~ (N-1)));
  // computing output matrix using avx instruction. 
  for(int rowA = 0; rowA < N; rowA +=2) {
    for(int colB = 0; colB < N; colB += 2){
   
       x = 0;
       __m256i r1=_mm256_setzero_si256();// all element in vector set to zero.
       // 1 iteration compute the 1 entry of output matrix. 
      for(int iter = 0; iter < N; iter=iter+8) 
      {
        __m256i a1= _mm256_loadu_si256((__m256i*)&matA[rowA*N+iter]);
        __m256i a2= _mm256_loadu_si256((__m256i*)&matA[(rowA+1)*N+iter]);
        __m256i b1= _mm256_loadu_si256((__m256i*)&matB[colB*N+iter]);
        __m256i b2= _mm256_loadu_si256((__m256i*)&matB[(colB+1)*N+iter]);
        
         r1+= _mm256_mullo_epi32(a1,b1);
         r1+= _mm256_mullo_epi32(a1,b2);
         r1+= _mm256_mullo_epi32(a2,b1);
         r1+= _mm256_mullo_epi32(a2,b2);
        
       
      }
      // typecast vector to integer array and and sum the element to find 1 entry
       int *res1= (int*)&r1;
        for(int k=0;k<8;k++)
            x+=res1[k];

      // compute output indices
      int rowC = rowA>>1;
        int colC = colB>>1;
      int indexC = rowC * (N>>1) + colC;
      output[indexC] = x;
    }
  }
}
