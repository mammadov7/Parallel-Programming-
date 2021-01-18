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
  int end = 0;
  int *tasks; // table for determining which task did which table
  int ready = 0;
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

    tasks = (int *) malloc(sizeof(int) * m);

    if ( tab == NULL ) {
      fprintf( stderr, "Unable to allocate %d elements\n", n ) ;
      return 1 ;
    }
    /* Initialize the arrays for sending to other tasks */ 
    for ( i = 1; i < m + 1; i++ ){
      for(j = 0; j < n; j++ ){
        tab[j] = lrand48()%n;
      } // End for j

    // Looking for ready task
    int done;
    MPI_Status sta;
    do{
      do {
        MPI_Iprobe(MPI_ANY_SOURCE, 4, MPI_COMM_WORLD, &done, &sta);
      } while (!done);
      MPI_Recv(&ready, 1, MPI_INT, sta.MPI_SOURCE, sta.MPI_TAG, MPI_COMM_WORLD, &sta);
    }
    while (!ready);
    tasks[i-1] = sta.MPI_SOURCE;
	// Message for continuing
      end = 1;
      MPI_Send(&end, 1, MPI_INT, sta.MPI_SOURCE, 2, MPI_COMM_WORLD);
      MPI_Send(tab, n,  MPI_INT,  sta.MPI_SOURCE, 1, MPI_COMM_WORLD);
      ready = 0;
    } // End for i
  
    // Sending End Message to the tasks
    end = 0;
    for ( i = 1; i < size; i++ )
      MPI_Send(&end, 1, MPI_INT, i, 2, MPI_COMM_WORLD);
  
    /* start the measurement */
    t1=MPI_Wtime();
  } // End if (rank == 0)
  else{ // Other tasks
  
    // Saying to root that I am ready
    MPI_Request request;
    ready = 1;
    MPI_Isend(&ready,1,MPI_INT,0,4,MPI_COMM_WORLD,&request);

    // Waiting Response from root for table
    MPI_Recv(&end, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    while( end ){
      MPI_Recv(tab, n, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    /* search for the max value */
      max=tab[0];
      for(j=0; j<n; j++)
        if(tab[j] > max)
          max = tab[j];
      // Sending to root the max of the array
      MPI_Send(&max, 1, MPI_INT, 0, 3, MPI_COMM_WORLD);
      // Saying to root that I am ready
      ready = 1;
      MPI_Isend(&ready,1,MPI_INT,0,4,MPI_COMM_WORLD,&request);
      // Waiting Response from root for new table
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
      MPI_Recv(&max, 1, MPI_INT,tasks[i-1], 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      printf("(Seed %d, Size %d) Max value = %d, Array number %d, task %d\n", s, n, max, i-1,tasks[i-1]);
    }
        /* stop the measurement */
    t2=MPI_Wtime();
    printf("General computation time: %f s\n", t2-t1);
  }// End if (rank == 0)

  MPI_Finalize();
  return 0;
}
