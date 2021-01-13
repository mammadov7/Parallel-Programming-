#include <stdio.h>
#include <mpi.h>
#include <unistd.h>

int main(int argc, char **argv)
{
  int rank, size;
	char hostname[256];
  if(MPI_Init(&argc, &argv))
  {
    fprintf(stderr, "erreur MPI_Init!\n");
    return(1);
  }

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
	gethostname(hostname,256);
  printf("Hello World from task %d out of %d on %s\n", rank, size, hostname );

  MPI_Finalize();
  return 0 ;
}
