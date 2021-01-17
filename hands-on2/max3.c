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
	int **tab;
	int *max;
	int *global_max;

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

  srand48(s);

  max = (int *) malloc(sizeof(int) * m);
	global_max = (int *) malloc(sizeof(int) * m); 
  /* Allocate the array */
  tab =(int **) malloc(sizeof(int*) * m);
	for (i = 0; i < m; i++)
    tab[i] = (int *) malloc(sizeof(int) * n);

  if ( tab == NULL ) { 
	  fprintf( stderr, "Unable to allocate %d elements\n", n ) ;
	  return 1 ; 
  }
  /* Initialize the array */
  for ( i = 0; i < m; i++ )
    for(j = 0; j < n; j++ ) 
      tab[i][j] = lrand48()%n;

  int local_start = rank * (n/size);
  int local_end = (rank + 1) * (n/size);
	if ( rank == (size - 1) )
		local_end = n;

  /* start the measurement */
	if( rank ==0 )
 		t1=MPI_Wtime();

  /* search for the max value of each table for task's corresponding part of tables*/
  for(i = 0; i < m; i++){
    max[i]=tab[i][local_start];
    for(j=local_start; j<local_end; j++) {
      if(tab[i][j] > max[i]) {
        max[i] = tab[i][j];
      }
    }
  }
  // Free
  for(i = 0; i < m; i++)
		free(tab[i]);
	free(tab);

	// We could do the MPI_send from each proc to 0 proc
	// And compute the global max in proc O
	// But, doing MPI_Reduce is better
  MPI_Reduce(max, global_max, m, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

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
    for (i = 0; i < m; i++)
	  	printf("(Seed %d, Size %d) Max value = %d, Array number %d\n", s, n, global_max[i], i);
	}// End if (rank == 0)

  free(global_max);
  free(max);
  MPI_Finalize();
  return 0;
}


