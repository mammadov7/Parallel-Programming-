/**
 * APPROXIMATE PATTERN MATCHING
 *
 * INF560
 */
 #include <string.h>
 #include <stdio.h>
 #include <stdlib.h>
 #include <fcntl.h>
 #include <unistd.h>
 #include <sys/time.h>
 
 #define APM_DEBUG 0
 
 #define CHECK(x) \
 do { \
   if (!(x)) { \
     fprintf(stderr, "%s:%d: ", __func__, __LINE__); \
     perror(#x); \
     exit(EXIT_FAILURE); \
   } \
 } while (0)

#define DIFFTEMPS(a,b) (((b).tv_sec - (a).tv_sec) + ((b).tv_usec - (a).tv_usec)/1000000.)
 
#define MIN3(a, b, c) ((a) < (b) ? ((a) < (c) ? (a) : (c)) : ((b) < (c) ? (b) : (c)))
 
 __global__ void cuda_levenshtein( 
      int buf_size, int local_buf_size,
      char *buf, char *pattern,
      int batch_size, int size_pattern,
      int *matches, int approx_factor
  ) {

    /* Traverse the input data up to the end of the file */
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int start = tid * batch_size;
    int end = start + batch_size;
    int *column;
    unsigned int j;
    if( end > n_bytes ) end = buf_size;
    if ( start < n_bytes ){
      column = (int *)malloc( (size_pattern+1) * sizeof( int ) ) ;

      for (j = start; j < end; j++ ){ 
        int distance = 0 ;
        int size = size_pattern;
        unsigned int x, y, lastdiag, olddiag;

        if ( local_buf_size < j + size_pattern )
          size = local_buf_size - j ;

/////////////////// Levenshtein ///////////////////////////        
        for (y = 1; y <= size; y++)
          column[y] = y;
        
        for (x = 1; x <= size; x++) {
            column[0] = x;
            lastdiag = x-1 ;
            for (y = 1; y <= size; y++) {
                olddiag = column[y];
                column[y] = MIN3(
                        column[y] + 1, 
                        column[y-1] + 1, 
                        lastdiag + (pattern[y-1] == buf[ j + x-1] ? 0 : 1)
                        );
                lastdiag = olddiag;
            }
        }
        distance = column[size];
/////////////// End of Levenshtein ///////////////////////        

        if ( distance <= approx_factor ) 
          matches[tid]++ ;

        } // End for j
    } // End if start
}

int sum( int *matches, int size ){
  int _sum = 0;
  for( int i = 0; i < size; i++ )
    _sum += matches[i];
  return _sum;
}

char* cuda_malloc(char *buf, int size){
    char* dBuf;
    
    CHECK( cudaSuccess == cudaMalloc( &dBuf, size) );
    CHECK( cudaSuccess == cudaMemcpy(dBuf, buf, n_bytes, cudaMemcpyHostToDevice) );

    return dBuf;
}


   /* Check each pattern one by one */

extern void cuda_call(int buf_size, int local_buf_size, char *dBuf, char *pattern, int approx_factor, int *ret){

    int size_pattern = strlen(pattern) ;
    int *matches, *dMatches;
    int nb_threads = 32, batch_size = 1000;
    int nb_blocks = (buf_size / batch_size + nb_threads) / nb_threads;
    char *dPatern;

    cudaMalloc(&dPatern, (size_pattern+1) * sizeof( char ) );
    cudaMemcpy(dPatern, pattern, size_pattern+1, cudaMemcpyHostToDevice);
    
    matches = (int *)malloc( (nb_blocks*nb_threads) * sizeof( int ) ) ;
    
    for( int j = 0; j < (nb_blocks*nb_threads); j++ )
        matches[j] = 0;
    
    cudaMalloc(&dMatches, (nb_blocks*nb_threads) * sizeof( int ));
    cudaMemcpy(dMatches, matches, (nb_blocks*nb_threads) * sizeof( int ), cudaMemcpyHostToDevice);
    
    cuda_levenshtein<<<nb_blocks, nb_threads>>>(  buf_size, local_buf_size , dBuf, dPatern, batch_size, size_pattern, dMatches, approx_factor );
    cudaDeviceSynchronize();
    cudaMemcpy(matches, dMatches, (nb_blocks*nb_threads) * sizeof( int ), cudaMemcpyDeviceToHost);
  
    *ret = sum(matches,nb_blocks*nb_threads);

    free(matches);
    cudaFree( dPatern );
    cudaFree( dMatches);
}
 
