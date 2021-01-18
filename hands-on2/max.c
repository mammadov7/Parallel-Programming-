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
  int n, m;
  int i,j;
  double t1, t2;
	int *tab;
	int max;

  /* MPI Initialization */
  MPI_Init(&argc, &argv);

  /* Get the rank of the current task and the number
   * of MPI processe
   */
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  /* Check the input arguments */
  if(argc <4) {
    printf("Usage: %s S N\n", argv[0]);
    printf( "\tS: seed for pseudo-random generator\n" ) ;
    printf( "\tN: size of the array\n" ) ;
    printf( "\tM: number of arrays\n" ) ;
    exit( 1 ) ;
  }

  s = atoi(argv[1]);
  n = atoi(argv[2]);
  m = atoi(argv[3]);

  if ( size != m ){
    printf("Number of tasks should be equal to the number of the arrays\n");
    exit(1);
  }
 tab = (int *) malloc(sizeof(int) * n);

  // Root task
  if( rank == 0 ){
    srand48(s);
    /* Allocate the array */
//    tab = (int *) malloc(sizeof(int) * n);

    if ( tab == NULL ) { 
      fprintf( stderr, "Unable to allocate %d elements\n", n ) ;
      return 1 ; 
    }
    /* Initialize the arrays for sending to other tasks */
    for ( i = 1; i < m; i++ ){
      for(j = 0; j < n; j++ ){ 
        tab[j] = lrand48()%n; 
      } // End for j
      MPI_Send(tab, n, MPI_INT, i, 1, MPI_COMM_WORLD);
    } // End for i

    /* Initialize the array for root rask */
    for(j = 0; j < n; j++ )
        tab[j] = lrand48()%n; 
  } // End if (rank == 0)
  else{ // Other tasks
    MPI_Recv(tab, n, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  /* start the measurement */
	if( rank ==0 )
 		t1=MPI_Wtime();

  /* search for the max value */
    max=tab[0];
    for(j=0; j<n; j++)
      if(tab[j] > max) 
        max = tab[j];
  // Free the table
	free(tab);
  MPI_Send(&max, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);

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
    printf("General computation time: %f s\n", t2-t1);
    // Printing the results of the root task
    printf("(Seed %d, Size %d) Max value = %d, Array number 4\n", s, n, max);
    
    // Printing the rest
    for (i = 1; i < m; i++){
      MPI_Recv(&max, 1, MPI_INT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	  	printf("(Seed %d, Size %d) Max value = %d, Array number %d\n", s, n, max, i-1);
    }
	}// End if (rank == 0)

  MPI_Finalize();
  return 0;
}


