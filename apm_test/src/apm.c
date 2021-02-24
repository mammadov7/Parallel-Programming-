/**
 * APPROXIMATE PATTERN MATCHING
 *
 * INF560
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/time.h>
#include <mpi.h>
#define APM_DEBUG 0

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
        fprintf( stderr, "Unable to allocate %ld byte(s) for main array\n",
                fsize ) ;
        return NULL ;
    }

    n_bytes = read( fd, buf, fsize ) ;
    if ( n_bytes != fsize ) 
    {
        fprintf( stderr, 
                "Unable to copy %ld byte(s) from text file (%d byte(s) copied)\n",
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
#define MIN(a, b) ((a) < (b) ? (a) : (b) )


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

int main( int argc, char ** argv )
{
  char ** pattern ;
  char * filename ;
  char * local_buf;
  int local_buf_size;
  int buf_size; //buf size for each process without shadow cells (a part from rank 0)
  int max_pat = 0; // size of the largest Pattern
  int approx_factor = 0 ;
  int nb_patterns = 0 ;
  int i,j ;
  struct timeval t1, t2;
  double duration ;
  int n_bytes ;
  int * n_matches, *glob_matches ;
  int rank, size;
  MPI_Init (&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);	/* who am i */  
  MPI_Comm_size(MPI_COMM_WORLD, &size); /* number of processes */ 
  /* Check number of arguments */
  if ( argc < 4 ) 
  {
    if(rank == 0 )
      printf( "Usage: %s approximation_factor "
              "dna_database pattern1 pattern2 ...\n", 
              argv[0] ) ;
    MPI_Finalize();
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

      if( l > max_pat )
        max_pat = l;
      
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

  /* Allocate the array of matches */
  n_matches = (int *)malloc( nb_patterns * sizeof( int ) ) ;
  glob_matches = (int *)malloc( nb_patterns * sizeof( int ) ) ;
  if ( n_matches == NULL || glob_matches == NULL)
  {
      fprintf( stderr, "Error: unable to allocate memory for %ldB\n",
              nb_patterns * sizeof( int ) ) ;
      return 1 ;
  }



  // Reading and distributing the file by Root
  if( rank == 0 ){
    char * buf ;
    printf( "Approximate Pattern Mathing: "
            "looking for %d pattern(s) in file %s w/ distance of %d\n", 
            nb_patterns, filename, approx_factor ) ;

    buf = read_input_file( filename, &n_bytes ) ;
    if ( buf == NULL ) return 1 ;
    
    buf_size = n_bytes / size;
    //if (n_bytes % size ) buf_size++; Why ??

    // Sending size of the local_buf to each Proc
    for (int to = 1; to < size; to++)
      MPI_Send(&n_bytes,1, MPI_INT, to, 0, MPI_COMM_WORLD);
    
    local_buf_size = n_bytes - buf_size*( size - 1);
    local_buf = (char *)malloc(sizeof(char)*(local_buf_size));
    strncpy(local_buf, &( buf[ buf_size*( size - 1) ] ), local_buf_size );
    
    // Sending the part of the data to each Proc
    for (int to = 1; to < size; to++){
        int start = (to-1)*buf_size;
        int end = MIN(to*buf_size + max_pat - 1, n_bytes);
        MPI_Send(&buf[start], end - start, MPI_CHAR, to, 1, MPI_COMM_WORLD);
    }
    buf_size = local_buf_size; //only change after data is sent
    free(buf);
  }
  // The rest should receive his part of data
  else{
    MPI_Recv(&n_bytes, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, NULL);
    buf_size = n_bytes / size;
    int start = (rank-1)*buf_size;
    int end = MIN(rank*buf_size + max_pat - 1, n_bytes);
    local_buf_size = end - start;
    local_buf = (char *)malloc(sizeof(char)*local_buf_size);
    MPI_Recv(local_buf, local_buf_size, MPI_CHAR, 0, 1, MPI_COMM_WORLD, NULL);
  }
  /*****
   * BEGIN MAIN LOOP
   ******/

  /* Timer start */
  if (rank == 0 ) gettimeofday(&t1, NULL);

  /* Check each pattern one by one */
  for ( i = 0 ; i < nb_patterns ; i++ )
  {
      int size_pattern = strlen(pattern[i]) ;
      int * column ;
      
      /* Initialize the number of matches to 0 */
      n_matches[i] = 0 ;

      column = (int *)malloc( (size_pattern+1) * sizeof( int ) ) ;
      if ( column == NULL ) 
      {
          fprintf( stderr, "Error: unable to allocate memory for column (%ldB)\n",
                  (size_pattern+1) * sizeof( int ) ) ;
          return 1 ;
      }

      for ( j = 0 ; j < buf_size; j++ ) 
      {
          int distance = 0 ;
          int size_pat ;

#if APM_DEBUG
          if ( j % 100 == 0 )
          {
          printf( "Procesing byte %d (out of %d)\n", j, n_bytes ) ;
          }
#endif

          size_pat = size_pattern ;
          if ( local_buf_size < j + size_pattern )
          {
              size_pat = local_buf_size - j ;
          }

          distance = levenshtein( pattern[i], &local_buf[j], size_pat, column ) ;

          if ( distance <= approx_factor ) {
              n_matches[i]++ ;
          }
      }

      free( column );
  }
    MPI_Reduce(n_matches, glob_matches, nb_patterns, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  /* Timer stop */
  
  if( rank == 0 ){
    gettimeofday(&t2, NULL);
  
    duration = (t2.tv_sec -t1.tv_sec)+((t2.tv_usec-t1.tv_usec)/1e6);
  
    printf( "APM done in %lf s\n", duration ) ;
  
    /*****
     * END MAIN LOOP
     ******/
  
    for ( i = 0 ; i < nb_patterns ; i++ )
    {
        printf( "Number of matches for pattern <%s>: %d\n", 
                pattern[i], glob_matches[i] ) ;
    }
  }
  MPI_Finalize();

  return 0 ;
}
