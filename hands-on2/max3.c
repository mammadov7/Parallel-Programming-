/**
 * INF560 - TD2
 *
 * Part 2: Work Decomposition
 * Sequential Max
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char**argv) {
  int rank, size;
  int s ;
  int n ;
  int * tab ;
  int i;
  int max ;
  double t1, t2;

  /* MPI Initialization */
  MPI_Init(&argc, &argv);

  /* Get the rank of the current task and the number
   * of MPI processe
   */
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  /* Check the input arguments */
  if(argc <3) {
    printf("Usage: %s S N\n", argv[0]);
    printf( "\tS: seed for pseudo-random generator\n" ) ;
    printf( "\tN: size of the array\n" ) ;
    exit( 1 ) ;
  }

  s = atoi(argv[1]);
  n = atoi(argv[2]);
  srand48(s);

  /* Allocate the array */
  tab = malloc(sizeof(int) * n);
  if ( tab == NULL ) { 
	  fprintf( stderr, "Unable to allocate %d elements\n", n ) ;
	  return 1 ; 
  }

  /* Initialize the array */
  for(i=0; i<n; i++) {
    tab[i] = lrand48()%n;
  }

  /* start the measurement */
  t1=MPI_Wtime();

  /* search for the max value */
  max=tab[0];
  for(i=0; i<n; i++) {
    if(tab[i] > max) {
      max = tab[i];
    }
  }

  /* stop the measurement */
  t2=MPI_Wtime();

  printf("Computation time: %f s\n", t2-t1);

#if DEBUG
  printf("the array contains:\n");
  for(i=0; i<n; i++) {
    printf("%d  ", tab[i]);
  }
  printf("\n");
#endif

  printf("(Seed %d, Size %d) Max value = %d, Time = %g s\n", s, n, max, t2-t1);
  MPI_Finalize();
  return 0;
}

