#include <pthread.h>
#include <immintrin.h> // for using avx instruction.

// Create other necessary functions here
// creating stuct to pass input to compute using single thread for each thread
struct single_thread_Input
{
int N_a;
int start_row;
int End_row;


int *A;
int *B;
int *C;
};


  void *singleThread1(void *Data)
  {
      struct single_thread_Input *In= (struct single_thread_Input*) Data; 
     int N=In->N_a;
      
    int *matA; int *matB; int *output;
     matA=In->A;
    matB=In->B;
     output=In->C;
      
      int start,end;
      start=In->start_row;
      end=In->End_row;
      int B=16;
    int x;

  assert( N>=4 and N == ( N &~ (N-1)));
  
  // for each thread only that row aacces is given for which they have responsibility to compute and second matrix bcz it require to compute.
  for(int rowA = start; rowA < end; rowA +=2) {
    for(int colB = 0; colB < N; colB += 2){
          x = 0;
       __m256i r1=_mm256_setzero_si256(); // all element in vector set to zero.
       // 1 iteration compute the 1 entry of output matrix. 
      for(int iter = 0; iter < N; iter=iter+8) 
      {
        __m256i a1= _mm256_loadu_si256((__m256i*)&matA[rowA*N+iter]);  // load 8 sequential element to vector a1 from address starting at given.
        __m256i a2= _mm256_loadu_si256((__m256i*)&matA[(rowA+1)*N+iter]);
        __m256i b1= _mm256_loadu_si256((__m256i*)&matB[colB*N+iter]);
        __m256i b2= _mm256_loadu_si256((__m256i*)&matB[(colB+1)*N+iter]);
        
        
         r1+= _mm256_mullo_epi32(a1,b1); // corresponding element get multiplied and it result added with r1.
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
  return NULL;
  }



// Fill in this function
void multiThread(int n, int *A, int *B, int *out)
{  
  int Nt;   //number of threads you want
    Nt=4;
   int add=n/Nt;
   pthread_t thr[Nt];  // Nt threads created 
   
    struct single_thread_Input Input[Nt];  //declare struct to pass the input with easch thread
    int start=0;
    
    for(int i=0;i<Nt;i++,start=start+add)
    { Input[i].N_a=n;
      Input[i].start_row=start;  // from this row i.e. start till start+add you have to compute entry that can be computed using this rows.
      Input[i].End_row=start+add;
      Input[i].A=A;  
      Input[i].B=B;
     Input[i].C=out;
    }
    
    for(int j=0;j<Nt;j++)
    {  
      pthread_create(&thr[j],NULL,singleThread1,(void*)&(Input[j]));  // calling single thread to compute and passing arguments in input[j].
    
    }
    for(int k=0;k<Nt;k++)
    { pthread_join(thr[k],NULL);
    }
    
}
