void G_x_fft_matvec(	cuComplex *dev_gamma_array, // For usual matvec (dev_mask is present)
			bool *dev_mask,
			cuComplex *dev_solution,
			cuComplex *dev_matmul_out_extended,
			cufftHandle plan,
			const unsigned int N)
{
	extend_by_zeros_kernel <<< dim3(THREADS_PER_BLOCK_M, THREADS_PER_BLOCK_M), dim3(Q, Q) >>> (	(bool *)dev_mask,
													(cuComplex *)dev_solution,
													(cuComplex *)dev_matmul_out_extended, N);
	cudacheckSYN();

	cufftcall(cufftExecC2C(	plan,
				(cuComplex *)dev_matmul_out_extended,
				(cuComplex *)dev_matmul_out_extended,
				CUFFT_FORWARD));
	cudacheckSYN();

	MatMul_ElemWise_kernel <<< dim3(THREADS_PER_BLOCK_M, THREADS_PER_BLOCK_M), dim3(Q, Q) >>> (	(cuComplex *)dev_gamma_array,
													(cuComplex *)dev_matmul_out_extended, N);	
	cudacheckSYN();

	cufftcall(cufftExecC2C(	plan,
				(cuComplex *)dev_matmul_out_extended,
				(cuComplex *)dev_matmul_out_extended,
				CUFFT_INVERSE));
	cudacheckSYN();
}


void G_x_fft_matvec(	cuComplex *dev_gamma_array, // For gradient matvec (dev_mask is absent)
			cuComplex *dev_solution,
			cuComplex *dev_matmul_out_extended,
			cufftHandle plan,
			const unsigned int N)
{	

	extend_by_zeros_kernel <<< dim3(THREADS_PER_BLOCK_M, THREADS_PER_BLOCK_M), dim3(Q, Q) >>> (	(cuComplex *)dev_solution,
													(cuComplex *)dev_matmul_out_extended, N);

	cufftcall(cufftExecC2C(	plan,
				(cuComplex *)dev_matmul_out_extended,
				(cuComplex *)dev_matmul_out_extended,
				CUFFT_FORWARD));
	cudacheckSYN();

	MatMul_ElemWise_kernel <<< dim3(THREADS_PER_BLOCK_M, THREADS_PER_BLOCK_M), dim3(Q, Q) >>> (	(cuComplex *)dev_gamma_array,
													(cuComplex *)dev_matmul_out_extended, N);	
	cudacheckSYN();

	cufftcall(cufftExecC2C(	plan,
				(cuComplex *)dev_matmul_out_extended,
				(cuComplex *)dev_matmul_out_extended,
				CUFFT_INVERSE));
	cudacheckSYN();
}

