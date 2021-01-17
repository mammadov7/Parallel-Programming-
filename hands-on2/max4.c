/**
 * INF560 - TD2
 *
 * Part 2: Work Decomposition
 * Sequential Max
 */



/**
* The goals of implementation:
* 1) You can use any number of tasks and arrays
* The Idea of implementation:
* 2) Root task will manage the distribution of arrays
*	Each task firstly waits a message from the root
*	to inter in the loop for receiving the array and
*	and the message which will indicate the end of data.
* If number of tasks more than nb arrays, the excess tasks
* will tarminate without doing anythig, cause they will
* receive directly end message. 
* If not, size = 5, M = 8: 
	Array	Task
	1	1
	2	2
	3	3
	4	4
	5	1
	6	2
	7	3
	8	4
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
  int end = 0;

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

  tab = (int *) malloc(sizeof(int) * n);

  // Root task
  if( rank == 0 ){
    srand48(s);
    /* Allocate the array */
  //  tab = (int *) malloc(sizeof(int) * n);

    if ( tab == NULL ) {
      fprintf( stderr, "Unable to allocate %d elements\n", n ) ;
      return 1 ;
    }
    /* Initialize the arrays for sending to other tasks */ 
    int to, rec = 0;
    for ( i = 1; i < m + 1; i++ ){
      for(j = 0; j < n; j++ ){
        tab[j] = lrand48()%n;
      } // End for j

      if( size < (m + 1) ) {
        to = (i + rec) % size;
        if ( !to ) { to++; rec++; }
      } else to = i;
	// Message for continuing
      end = 1;
      MPI_Send(&end, 1, MPI_INT, to, 2, MPI_COMM_WORLD);
      MPI_Send(tab, n, MPI_INT, to, 1, MPI_COMM_WORLD);
    } // End for i
  
    // Sending End Message to the tasks
    end = 0;
    for ( i = 1; i < size; i++ )
      MPI_Send(&end, 1, MPI_INT, i, 2, MPI_COMM_WORLD);
  
    /* start the measurement */
    t1=MPI_Wtime();
  } // End if (rank == 0)
  else{ // Other tasks
    MPI_Recv(&end, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    while( end ){
      MPI_Recv(tab, n, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    /* search for the max value */
      max=tab[0];
      for(j=0; j<n; j++)
        if(tab[j] > max)
          max = tab[j];
  
    MPI_Send(&max, 1, MPI_INT, 0, 3, MPI_COMM_WORLD);
    // Looking for new array
    MPI_Recv(&end, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

  // Free the table
    free(tab);
  }
#if DEBUG
  printf("the array contains:\n");
  for(i=0; i<n; i++) {
    printf("%d  ", tab[i]);
  }
  printf("\n");
#endif

  if( rank == 0 ){
    // Printing the maxs
    int from, rec = 0;
    for (i = 1 ; i < m + 1; i++){
      if( size < (m + 1) ) {
        from = (i + rec) % size ;
        if ( !from ) { from++; rec++; }
      } else from = i;

      MPI_Recv(&max, 1, MPI_INT, from, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      printf("(Seed %d, Size %d) Max value = %d, Array number %d, task %d\n", s, n, max, i-1,from);
    }
        /* stop the measurement */
    t2=MPI_Wtime();
    printf("General computation time: %f s\n", t2-t1);
  }// End if (rank == 0)

  MPI_Finalize();
  return 0;
}
