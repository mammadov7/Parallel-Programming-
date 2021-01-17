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
	int global_max = 0;

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
  
  int local_start = rank * (n/size);
  int local_end = (rank + 1) * (n/size);
	if ( rank == (size - 1) )
		local_end = n;

  /* start the measurement */
	if( rank ==0 )
 		t1=MPI_Wtime();

  /* search for the max value */
  max=tab[0];
  for(i=local_start; i<local_end; i++) {
    if(tab[i] > max) {
      max = tab[i];
    }
  }

	// We could do the MPI_send from each proc to 0 proc
	// And compute the global max in proc O
	// But, doing MPI_Reduce is better

	MPI_Reduce(&max, &global_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

#if DEBUG
  printf("the array contains:\n");
  for(i=0; i<n; i++) {
    printf("%d  ", tab[i]);
  }
  printf("\n");
#endif
	
	if( rank == 0 ){
    /* stop the measurement */
    t2=MPI_Wtime();
    printf("Computation time: %f s\n", t2-t1);
  	printf("(Seed %d, Size %d) Max value = %d, Time = %g s\n", s, n, global_max, t2-t1);
	}// End if (rank == 0)

  MPI_Finalize();
  return 0;
}
