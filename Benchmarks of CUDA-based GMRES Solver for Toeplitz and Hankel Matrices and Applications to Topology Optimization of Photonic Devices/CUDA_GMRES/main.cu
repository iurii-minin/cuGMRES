#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <string>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "GMRES_kernels.cuh"
#include "saveGPU.cuh"
#include "Fast_matvec.cuh"
#include "GMRES.cuh"
#include "launch_GMRES.cuh"


int main()
{
	printf("IN MAIN\n");
	time_t clock_time = clock();

	//discrete_gradient_numerical_method_v1();
	//discrete_gradient_numerical_method_v2();
	//discrete_gradient_numerical_method_v3();
	//greedy_numerical_method_v1();
	//discrete_gradient_multiple_points_numerical_method_v1();
	//discrete_gradient_multiple_points_numerical_method_v2();
	launch_GMRES();
	//double_resolute_by_size();
	//get_relative_errors();

	printf("Successful exit from Cuda\n");
	printf("Consumption time with OUTPUTTING = %f seconds \n", (float)(clock() - clock_time) / (float)(CLOCKS_PER_SEC));

	return 0;
}



