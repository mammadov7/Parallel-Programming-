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


 char * 
 read_input_file( char * filename, int * size )
 {
     char * buf ;
     off_t fsize;
     int fd = 0 ;
     int n_bytes = 1 ;
 
     /* Open the text file */
     fd = open( filename, O_RDONLY ) ;
     if ( fd == -1 ) 
     {
         fprintf( stderr, "Unable to open the text file <%s>\n", filename ) ;
         return NULL ;
     }
 
 
     /* Get the number of characters in the textfile */
     fsize = lseek(fd, 0, SEEK_END);
     if ( fsize == -1 )
     {
         fprintf( stderr, "Unable to lseek to the end\n" ) ;
         return NULL ;
     }
 
 #if APM_DEBUG
     printf( "File length: %lld\n", fsize ) ;
 #endif
 
     /* Go back to the beginning of the input file */
     if ( lseek(fd, 0, SEEK_SET) == -1 ) 
     {
         fprintf( stderr, "Unable to lseek to start\n" ) ;
         return NULL ;
     }
 
     /* Allocate data to copy the target text */
     buf = (char *)malloc( fsize * sizeof ( char ) ) ;
     if ( buf == NULL ) 
     {
         fprintf( stderr, "Unable to allocate %lld byte(s) for main array\n",
                 fsize ) ;
         return NULL ;
     }
 
     n_bytes = read( fd, buf, fsize ) ;
     if ( n_bytes != fsize ) 
     {
         fprintf( stderr, 
                 "Unable to copy %lld byte(s) from text file (%d byte(s) copied)\n",
                 fsize, n_bytes) ;
         return NULL ;
     }
 
 #if APM_DEBUG
     printf( "Number of read bytes: %d\n", n_bytes ) ;
 #endif
 
     *size = n_bytes ;
 
 
     close( fd ) ;
 
 
     return buf ;
 }
 
 
 #define MIN3(a, b, c) ((a) < (b) ? ((a) < (c) ? (a) : (c)) : ((b) < (c) ? (b) : (c)))
 
 int levenshtein(char *s1, char *s2, int len, int * column) {
     unsigned int x, y, lastdiag, olddiag;
 
     for (y = 1; y <= len; y++)
     {
         column[y] = y;
     }
     for (x = 1; x <= len; x++) {
         column[0] = x;
         lastdiag = x-1 ;
         for (y = 1; y <= len; y++) {
             olddiag = column[y];
             column[y] = MIN3(
                     column[y] + 1, 
                     column[y-1] + 1, 
                     lastdiag + (s1[y-1] == s2[x-1] ? 0 : 1)
                     );
             lastdiag = olddiag;
 
         }
     }
     return(column[len]);
 }
 

int approx_factor = 0; // Global


 __global__ void cuda_call( int n_bytes, char *buf, char *pattern, int *column, int batch_size, int size_pattern, int *matches ) {
    /* Traverse the input data up to the end of the file */
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int start = tid * batch_size;
    int end = start + batch_size;
    if( end > n_bytes ) end = n_bytes;

    for (int j = start; j < end; j++ ) 
    {
      int distance = 0 ;
      int size ;
      size = size_pattern ;
      if ( n_bytes - j < size_pattern )
      {
        size = n_bytes - j ;
      }
      distance = levenshtein( pattern, &buf[j], size, column ) ;

      if ( distance <= approx_factor ) {
        matches[tid]++ ;
      }
    }

}

int sum( int *matches, int size ){
  int _sum = 0;
  for( int i = 0; i < size; i++ )
    _sum += matches[i];
  return _sum;
}
 
 int 
 main( int argc, char ** argv )
 {
   char ** pattern ;
   char * filename ;
   int nb_patterns = 0 ;
   int i, j ;
   char *buf, *dBuf;
   struct timeval t1, t2;
   double duration ;
   int n_bytes ;
   int * n_matches ;
 
   /* Check number of arguments */
   if ( argc < 4 ) 
   {
     printf( "Usage: %s approximation_factor "
             "dna_database pattern1 pattern2 ...\n", 
             argv[0] ) ;
     return 1 ;
   }
 
   /* Get the distance factor */
   approx_factor = atoi( argv[1] ) ;
 
   /* Grab the filename containing the target text */
   filename = argv[2] ;
 
   /* Get the number of patterns that the user wants to search for */
   nb_patterns = argc - 3 ;
 
   /* Fill the pattern array */
   pattern = (char **)malloc( nb_patterns * sizeof( char * ) ) ;
   if ( pattern == NULL ) 
   {
       fprintf( stderr, 
               "Unable to allocate array of pattern of size %d\n", 
               nb_patterns ) ;
       return 1 ;
   }
 
   /* Grab the patterns */
   for ( i = 0 ; i < nb_patterns ; i++ ) 
   {
       int l ;
 
       l = strlen(argv[i+3]) ;
       if ( l <= 0 ) 
       {
           fprintf( stderr, "Error while parsing argument %d\n", i+3 ) ;
           return 1 ;
       }
 
       pattern[i] = (char *)malloc( (l+1) * sizeof( char ) ) ;
       if ( pattern[i] == NULL ) 
       {
           fprintf( stderr, "Unable to allocate string of size %d\n", l ) ;
           return 1 ;
       }
 
       strncpy( pattern[i], argv[i+3], (l+1) ) ;
   }
 
   printf( "Approximate Pattern Mathing: "
           "looking for %d pattern(s) in file %s w/ distance of %d\n", 
           nb_patterns, filename, approx_factor ) ;
 
   buf = read_input_file( filename, &n_bytes ) ;
   cudaMalloc(&dBuf, n_bytes);
   cudaMemcpy(dBuf, buf, n_bytes, cudaMemcpyHostToDevice);

   if ( buf == NULL )
   {
       return 1 ;
   }
   printf("%d\n", n_bytes);
   /* Allocate the array of matches */
   n_matches = (int *)malloc( nb_patterns * sizeof( int ) ) ;
   if ( n_matches == NULL )
   {
       fprintf( stderr, "Error: unable to allocate memory for %ldB\n",
               nb_patterns * sizeof( int ) ) ;
       return 1 ;
   }
 
   /*****
    * BEGIN MAIN LOOP
    ******/
 
   /* Timer start */
   gettimeofday(&t1, NULL);


 
   /* Check each pattern one by one */
   for ( i = 0 ; i < nb_patterns ; i++ )
   {
      int size_pattern = strlen(pattern[i]) ;
      int * column, *dColumn;
      int matches, dMatches;
      int nb_threads = 1024, batch_size = 1000;
      int nb_blocks = (n_bytes / batch_size + nb_threads) / nb_threads;
      char *dPatern;

      n_matches[i] = 0 ;

      cudaMalloc(&dPatern, (size_pattern+1) * sizeof( char ) );
      cudaMemcpy(dPatern, pattern[i], size_pattern+1, cudaMemcpyHostToDevice);
 
      column = (int *)malloc( (size_pattern+1) * sizeof( int ) ) ;
      cudaMalloc(&dColumn, (size_pattern+1) * sizeof( int ));

      matches = (int *)malloc( (nb_blocks*nb_threads) * sizeof( int ) ) ;
      cudaMalloc(&dMatches, (nb_blocks*nb_threads) * sizeof( int ));
      cudaMemcpy(dMatches, matches, (nb_blocks*nb_threads) * sizeof( int ), cudaMemcpyHostToDevice);
      
      cuda_call<<<nb_blocks, nb_threads>>>(   , dBuf, dPatern, dColumn, batch_size, size_pattern, dMatches );
      cudaDeviceSynchronize();

      cudaMemcpy(matches, dMatches, (nb_blocks*nb_threads) * sizeof( int ), cudaMemcpyDeviceToHost);
    
      n_matches[i] = sum(matches,nb_blocks*nb_threads);

      free( column );
      cudaFree( dColumn );
      cudaFree( dPatern );
      free(matches);
      cudaFree( dMatches);
   }
   /* Timer stop */
   gettimeofday(&t2, NULL);
   duration = (t2.tv_sec -t1.tv_sec)+((t2.tv_usec-t1.tv_usec)/1e6);
 
   printf( "APM done in %lf s\n", duration ) ;
 
   /*****
    * END MAIN LOOP
    ******/
    
   free(buf);
   cudaFree(dBuf);
 
   for ( i = 0 ; i < nb_patterns ; i++ )
   {
       printf( "Number of matches for pattern <%s>: %d\n", 
               pattern[i], n_matches[i] ) ;
   }
 
   return 0 ;
 }
 