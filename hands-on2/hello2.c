#include <stdio.h>
#include <mpi.h>
#include <unistd.h>

int main(int argc, char **argv)
{
  int rank, size, ready = 0;
	char hostname[256];
  if(MPI_Init(&argc, &argv))
  {
    fprintf(stderr, "erreur MPI_Init!\n");
    return(1);
  }

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  // Root proc - 0
	if( !rank ){
		int i = 1;
		int nb_tasks = 1;
		for(i = 1; i < size; i++ ){
			MPI_Recv(&ready, 1, MPI_INT, i, 1, MPI_COMM_WORLD,NULL);
			if( ready == 1 )
				nb_tasks++;
		}//End of for
		printf("Hello World with %d ready task(s)\n",nb_tasks);
	}//end of if
	else{
		ready = 1;
		MPI_Send(&ready, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
	}


  MPI_Finalize();
  return 0 ;
}

