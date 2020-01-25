void cuSolve_LES(	cuComplex *dev_A,
			cuComplex *dev_B,
			const int m,
			cusolverDnHandle_t cusolverH,
			cuComplex *dev_work,
			int *dev_Ipiv,
			int *dev_info,
			timespec *h_computation_times,
			unsigned int *clock_i_p);


void show_norm(	const char *description,
		cublasHandle_t handle,
		const cuComplex *dev_array,
		const unsigned int size_array);

void show_norm_F(	const char *description,
			cublasHandle_t handle,
			const float *dev_array,
			const unsigned int size_array);

void Fast_GMRES_with_CUDA(	const cuComplex *dev_gamma_array,
				const bool *dev_mask, cuComplex *dev_solution,
				float **dev_actual_residual,
				unsigned int *GMRES_n,
				cufftHandle plan,
				cublasHandle_t *handle_p,
				const float tolerance,
				const bool for_gradient,
				const unsigned int h_index_of_max,
				bool *h_res_vs_tol_p,
				unsigned int maxiter,
				cusolverDnHandle_t cusolverH,
				timespec *h_computation_times,
				const unsigned int N)
{
	unsigned int clock_i = 0;

	clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_0_

	dim3 blocks(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
	dim3 blocks_M(THREADS_PER_BLOCK_M, THREADS_PER_BLOCK_M);
	dim3 threads(Q, Q);
	dim3 blocksize(512);
	dim3 gridsize(N * N / blocksize.x);

	float h_residual_initial = 0.f;
	float h_actual_residual = 0.f;
	cuComplex h_Jtotal_0j = {};

	cuComplex *dev_orthogonal_basis		 = NULL;
	cuComplex *dev_HH			 = NULL;
	cuComplex *dev_Htemp			 = NULL;
	cuComplex *dev_extended			 = NULL;
	cuComplex *dev_vec_resudual		 = NULL;
	cuComplex *dev_H_			 = NULL;
	cuComplex *dev_alpha			 = NULL;
	cuComplex *dev_beta			 = NULL;
	cuComplex *dev_Jtotal			 = NULL;
	cuComplex *dev_cc 			 = NULL;
	cuComplex *dev_Givens_rotation 		 = NULL;
	cuComplex *dev_Givens_rotation_0	 = NULL;
	cuComplex *dev_Givens_rotation_1	 = NULL;
	cuComplex *dev_Givens_rotation_2	 = NULL;
	cuComplex *dev_Givens_rotation_3	 = NULL;
	cuComplex *dev_buffer_LES_cc		 = NULL;

	int *dev_info 				 = NULL;
	int *dev_Ipiv 				 = NULL;

	unsigned int GMRES_i = 0;
	const unsigned int maxiter_plus_1 = maxiter + 1;

	cudacall(cudaMalloc((void**)dev_actual_residual, 			    maxiter_plus_1 * sizeof(float)));
	cudacall(cudaMalloc((void**)&dev_orthogonal_basis, ((maxiter + 6) * N * N - 8 * N + 1) * sizeof(cuComplex)));
	cudacall(cudaMalloc((void**)&dev_info, 					       (maxiter + 1) * sizeof(int)));

	dev_vec_resudual = dev_H_ = dev_orthogonal_basis + maxiter_plus_1 * N * N;
	dev_Jtotal = dev_vec_resudual + maxiter * maxiter_plus_1;

	dev_extended = dev_HH = dev_Htemp = dev_vec_resudual + N * N;

	dev_cc = dev_extended + maxiter * maxiter;
	dev_Givens_rotation = dev_Jtotal + maxiter_plus_1 * maxiter_plus_1;
	dev_Givens_rotation_3 = dev_Givens_rotation + maxiter_plus_1 * maxiter_plus_1 - 1;
	dev_Givens_rotation_2 = dev_Givens_rotation_3 - 1;
	dev_Givens_rotation_1 = dev_Givens_rotation_2 - maxiter;
	dev_Givens_rotation_0 = dev_Givens_rotation_1 - 1;
	dev_alpha  = dev_Givens_rotation + maxiter_plus_1 * maxiter_plus_1;
	dev_beta   = dev_alpha + 1;
	dev_buffer_LES_cc = dev_beta + 1;

	dev_Ipiv = dev_info + 1;
							
	clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_1_ //Initialization
//========================================= BEGIN: get_residual_vector =======================================================
	if (for_gradient)
	{
		G_x_fft_matvec(	(cuComplex *)dev_gamma_array,
				(cuComplex *)dev_solution,
				(cuComplex *)dev_extended,
				(cufftHandle)plan, N);

		clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_2_

		_2D_to_1D_compared_kernel <<< blocks, threads >>> (	(bool *)dev_mask,
									(cuComplex *)dev_solution,
									(cuComplex*)dev_extended,
									(cuComplex*)dev_vec_resudual,
									h_index_of_max, N);
	}
	else
	{
		G_x_fft_matvec(	(cuComplex *)dev_gamma_array,
				(bool *)dev_mask,
				(cuComplex *)dev_solution,
				(cuComplex *)dev_extended,
				(cufftHandle)plan, N);

		clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_2_

		_2D_to_1D_compared_kernel <<< blocks, threads >>> (	(cuComplex *)dev_solution,
									(cuComplex*)dev_extended,
									(cuComplex*)dev_vec_resudual, N);
	}
	cudacheckSYN();

	clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_3_
//========================================== END: get_residual_vector =========================================================
	cublascall(cublasScnrm2(        (cublasHandle_t) *handle_p,
					N * N,
		                        (const cuComplex *)dev_vec_resudual, 1,
					(float  *)*dev_actual_residual));
	cudacheckSYN();

	clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_4_
//============================================= Begin: Condition to iterate ==========================================================
	cudacall(cudaMemcpyAsync(&h_residual_initial, *dev_actual_residual, sizeof(float), cudaMemcpyDeviceToHost));

	clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_5_
//=============================================== End: Condition to iterate ===========================================================
//============================================BEGIN:residual_normalization_kernel=======================================================
	residual_normalization_kernel <<< gridsize, blocksize >>> (	(cuComplex *)dev_vec_resudual,
									(float *)*dev_actual_residual,
									(cuComplex *)dev_orthogonal_basis);
	cudacheckSYN();

	set_alpha_beta_kernel <<< 4, 1 >>> ((cuComplex *)dev_alpha, (cuComplex *)dev_beta);
	//don't synchronize

	clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_6_
//============================================= END:residual_normalization_kernel ==================================================
	if (h_residual_initial > tolerance)
	{

		cudacall(cudaMemsetAsync((cuComplex *)dev_H_, 0, maxiter_plus_1 * maxiter * sizeof(cuComplex)));		//don't synchronize

		clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_7_

		if (for_gradient)
		{
			G_x_fft_matvec(	(cuComplex *)dev_gamma_array,
					(cuComplex *)dev_orthogonal_basis,
					(cuComplex *)dev_extended,
					(cufftHandle)plan, N);

			clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_8_

			_2D_to_1D_kernel <<< blocks, threads >>> (	(bool *)dev_mask,
									(cuComplex*)dev_orthogonal_basis,
									(cuComplex *)dev_extended,
									(cuComplex *)dev_orthogonal_basis + N * N, N);
		}
		else
		{
			G_x_fft_matvec(	(cuComplex *)dev_gamma_array,
					(bool *)dev_mask,
					(cuComplex *)dev_orthogonal_basis,
					(cuComplex *)dev_extended,
					(cufftHandle)plan, N);

			clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_8_

			_2D_to_1D_kernel <<< blocks, threads >>> (	(cuComplex*)dev_orthogonal_basis,
									(cuComplex *)dev_extended,
									(cuComplex *)dev_orthogonal_basis + N * N, N);
		}
		cudacheckSYN();

		clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_9_

		cublascall(cublasCdotc(		(cublasHandle_t) *handle_p, N * N,
						(const cuComplex *)dev_orthogonal_basis, 1,
						(const cuComplex *)dev_orthogonal_basis + N * N, 1,
						(cuComplex *)dev_H_));
		cudacheckSYN();

		clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_10_

		weight_subtract_kernel <<< gridsize, blocksize >>> (	(cuComplex *)dev_orthogonal_basis + N * N,
									(cuComplex *)dev_H_,
									(cuComplex *)dev_orthogonal_basis);
		cudacheckSYN();

		clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_11_

		cublascall(cublasScnrm2(        (cublasHandle_t) *handle_p,
						N * N,
				                (const cuComplex *)dev_orthogonal_basis + N * N, 1,
						(float  *)*dev_actual_residual + maxiter));
		cudacheckSYN();

		clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_12_
	//============================================== BEGIN: Fill Orthogonal Basis matrix ============================================
		get_complex_divided <<< 3, 1 >>> (	(const float *)*dev_actual_residual + maxiter,
							(cuComplex *)dev_H_ + maxiter,
							(float *)*dev_actual_residual + maxiter);
		cudacheckSYN();

		clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_13_

		cublascall(cublasCsscal(	(cublasHandle_t) *handle_p, N * N,
				            	(const float           *)*dev_actual_residual + maxiter,
				            	(cuComplex       *)dev_orthogonal_basis + N * N, 1));
		cudacheckSYN();

		clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_14_
	//============================================== END: Orthogonal Basis matrix  ==================================================
	//============================================= BEGIN: Create Givens_Rotation_Matrix ========================================
		set_Identity_matrix_kernel <<< dim3(maxiter_plus_1, maxiter_plus_1), dim3(1, 1) >>> ((cuComplex *)dev_Givens_rotation);
		cudacheckSYN();

		clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_15_
	//=============================================== END: Create Givens_Rotation_Matrix,  ========================================
	//============================================= BEGIN: Create Jtotal_Matrix ========================================
		set_first_Jtotal_kernel <<< maxiter_plus_1 * maxiter_plus_1 * 2, 1 >>> ((cuComplex *)dev_Jtotal,
											(cuComplex *)dev_H_,
											maxiter,
											maxiter_plus_1);
		cudacheckSYN();

		clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++);//_16_
	//=============================================== END: Create Jtotal_Matrix,  ========================================
	//===================================================== BEGIN: Update residual ======================================================
		cudacall(cudaMemcpy(&h_Jtotal_0j, dev_Jtotal + maxiter_plus_1, sizeof(cuComplex), cudaMemcpyDeviceToHost));
		h_actual_residual = h_residual_initial * sqrt( (pow((float)h_Jtotal_0j.x, 2.0f) + pow((float)h_Jtotal_0j.y, 2.f)));
		cudacall(cudaMemcpyAsync(*dev_actual_residual + 1, &h_actual_residual, sizeof(float), cudaMemcpyHostToDevice));

		clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_17_
	//======================================================= END: Update residual ======================================================
		GMRES_i ++;

		for(GMRES_i = 1; ((GMRES_i < maxiter)); GMRES_i ++) //(h_actual_residual > tolerance) &&
		{
			clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_18_

			if (for_gradient)
			{	
				G_x_fft_matvec(	(cuComplex *)dev_gamma_array,
						(cuComplex *)dev_orthogonal_basis + GMRES_i * N * N,
						(cuComplex *)dev_extended,
						(cufftHandle)plan, N);

				clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_19_

				_2D_to_1D_kernel <<< blocks, threads >>> (	(bool *)dev_mask,
										(cuComplex*)dev_orthogonal_basis + GMRES_i * N * N,
										(cuComplex *)dev_extended,
										(cuComplex *)dev_orthogonal_basis + (GMRES_i + 1) * N * N, N);
			}
			else
			{
				G_x_fft_matvec(	(cuComplex *)dev_gamma_array,
						(bool *)dev_mask,
						(cuComplex *)dev_orthogonal_basis + GMRES_i * N * N,
						(cuComplex *)dev_extended,
						(cufftHandle)plan, N);

				clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_19_

				_2D_to_1D_kernel <<< blocks, threads >>> (	(cuComplex*)dev_orthogonal_basis + GMRES_i * N * N,
										(cuComplex *)dev_extended,
										(cuComplex *)dev_orthogonal_basis + (GMRES_i + 1) * N * N, N);
			}
			cudacheckSYN();

			clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_20_
	
			for(unsigned int j = 0; j < GMRES_i + 1; j++)
			{
				cublascall(cublasCdotc(	(cublasHandle_t) *handle_p, N * N,
							(const cuComplex *)dev_orthogonal_basis + j * N * N, 1,
							(const cuComplex *)dev_orthogonal_basis + (GMRES_i + 1) * N * N, 1,
							(cuComplex *)dev_H_ + j * maxiter + GMRES_i));
				cudacheckSYN();
				clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_21_

				weight_subtract_kernel <<< gridsize, blocksize >>> (	(cuComplex *)dev_orthogonal_basis + (GMRES_i + 1) * N * N,
											(cuComplex *)dev_H_ + j * maxiter + GMRES_i,
											(cuComplex *)dev_orthogonal_basis + j * N * N);
				cudacheckSYN();
				clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_22_
			}

			cublascall(cublasScnrm2((cublasHandle_t) *handle_p,
						N * N,
						(const cuComplex *)dev_orthogonal_basis + (GMRES_i + 1) * N * N, 1,
						(float  *)*dev_actual_residual + maxiter));
			cudacheckSYN();

			clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_23_
		//============================================== BEGIN: Fill Orthogonal Basis m.============================================
			get_complex_divided <<< 3, 1 >>> (	(const float *)*dev_actual_residual + maxiter,
								(cuComplex *)dev_H_ + (GMRES_i + 1) * maxiter + GMRES_i,
								(float *)*dev_actual_residual + maxiter);
			cudacheckSYN();
			clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_24_

			cublascall(cublasCsscal(		(cublasHandle_t) *handle_p, N * N,
								(const float           *)*dev_actual_residual + maxiter,
								(cuComplex       *)dev_orthogonal_basis + (GMRES_i + 1) * N * N, 1));
			cudacheckSYN();
			clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_25_
		//===============================    END: Fill Orthogonal Basis m.  ===========================================
		//============================================== Begin: Least Squares Step =========================================================
		//================================ BEGIN: MATMUL (H_temp=Jtotal * H) ==============================================
			cublascall(cublasCgemm3m(	(cublasHandle_t)*handle_p,
							CUBLAS_OP_N,
							CUBLAS_OP_N,
							(unsigned int)GMRES_i + 1,
							(unsigned int)GMRES_i + 2,
							(unsigned int)GMRES_i + 2,
							(const cuComplex *)dev_alpha,
							(cuComplex *)dev_H_, (unsigned int)maxiter,
							(cuComplex *)dev_Jtotal,      (unsigned int)maxiter_plus_1,
							(const cuComplex *)dev_beta,
							(cuComplex *)dev_Htemp, (unsigned int)GMRES_i + 1));
			cudacheckSYN();
			clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_26_
		//================================== END: MATMUL (H_temp=Jtotal * H) ===============================================
		//================================================ END: Least Squares Step =========================================================
		//============================================= BEGIN: Create Givens_Rotation_Matrix ========================================
			set_4_Givens_rotation_matrix_elements_kernel <<< 8, 1 >>> (	(cuComplex *)dev_Htemp,
											maxiter_plus_1,
											(cuComplex *)dev_Givens_rotation_0,
											(cuComplex *)dev_Givens_rotation_1,
											(cuComplex *)dev_Givens_rotation_2,
											(cuComplex *)dev_Givens_rotation_3,
											GMRES_i + 1);
			cudacheckSYN();
			clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_27_
		//=============================================== END: Create Givens_Rotation_Matrix ========================================
		//================================================== BEGIN: Jtotal = J*Jtotal =================================================
			cublascall(cublasCgemm3m(	(cublasHandle_t)*handle_p,
							CUBLAS_OP_N,
							CUBLAS_OP_N,
							(unsigned int)GMRES_i + 2,
							(unsigned int)GMRES_i + 2,
							(unsigned int)GMRES_i + 2,
							(const cuComplex *)dev_alpha,
							(cuComplex *)dev_Jtotal,          (unsigned int)maxiter_plus_1,
							(cuComplex *)dev_Givens_rotation + (maxiter - 1 - GMRES_i) * (maxiter_plus_1 + 1), (unsigned int)maxiter_plus_1,
							(const cuComplex *)dev_beta,
							(cuComplex *)dev_Jtotal,          (unsigned int)maxiter_plus_1));
			cudacheckSYN();
			clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_28_
		//==================================================== END: Jtotal = J*Jtotal =================================================
		//===================================================== BEGIN: Update residual ======================================================
			cudacall(cudaMemcpy(&h_Jtotal_0j, dev_Jtotal + maxiter_plus_1 * (GMRES_i + 1), sizeof(cuComplex), cudaMemcpyDeviceToHost));
			h_actual_residual = h_residual_initial * sqrt( (pow((float)h_Jtotal_0j.x, 2.0f) + pow((float)h_Jtotal_0j.y, 2.f)));
			cudacall(cudaMemcpyAsync(*dev_actual_residual + GMRES_i + 1, &h_actual_residual, sizeof(float), cudaMemcpyHostToDevice));

			clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_29_
		//======================================================= END: Update residual ======================================================
		}
	//================================================== BEGIN: HH = (Jtotal*H)_resized ==========================================================
		cublascall(cublasCgemm3m(	(cublasHandle_t)*handle_p,
						CUBLAS_OP_T,
						CUBLAS_OP_T,
						(unsigned int)GMRES_i,
						(unsigned int)GMRES_i,
						(unsigned int)GMRES_i + 1,
						(const cuComplex *)dev_alpha,
						(cuComplex *)dev_Jtotal, (unsigned int)maxiter_plus_1,
						(cuComplex *)dev_H_, 	 (unsigned int)maxiter,
						(const cuComplex *)dev_beta,
						(cuComplex *)dev_HH, 	 (unsigned int)GMRES_i));
		cudacheckSYN();
		clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_30_
	//===================================================== END: HH = (Jtotal*H)_resized ==========================================================
	//================================================= BEGIN: cc = Jtotal * norm_res_vec =========================================================
		set_cc_kernel <<< GMRES_i, 1 >>> (	(cuComplex *)dev_cc,
							(cuComplex *)dev_Jtotal,
							(float *)*dev_actual_residual,
							maxiter_plus_1);
		cudacheckSYN();
		clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_31_
	//=================================================== END: cc = Jtotal * norm_res_vec =========================================================
		if (GMRES_i > 0)
		{
			if (GMRES_i < 2)
			{
				get_new_solution_kernel <<< 1, 1 >>> (	(cuComplex *)dev_cc,
									(cuComplex *)dev_HH);	
				cudacheckSYN();

				get_solution_kernel <<< gridsize, blocksize >>> (	(cuComplex *)dev_solution,
											(cuComplex *)dev_cc,
											(cuComplex *)dev_orthogonal_basis);
				cudacheckSYN();
			}
			else
			{
			//============================================ BEGIN: Find solution to the LES(cc_new) for HH*cc_new=cc ============================================
				cuSolve_LES(	(cuComplex *)dev_HH,
						(cuComplex *)dev_cc,
						GMRES_i,
						cusolverH,
						(cuComplex *)dev_buffer_LES_cc,
						(int *)dev_Ipiv,
						(int *)dev_info,
						(timespec *)h_computation_times,
						(unsigned int *)&clock_i);
			//============================================ END: Find solution to the LES(cc_new) for HH*cc_new=cc ===========================================
			//============================================ BEGIN: x = x0 + V * cc ===========================================
				for(unsigned int j = 0; j < GMRES_i; j++)
				{
					add_kernel <<< gridsize, blocksize >>> ((cuComplex *)dev_solution,
										(cuComplex *)dev_orthogonal_basis + j * N * N,
										(cuComplex *)dev_cc + j);
				//	cudacheckSYN();
					clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_34_
				}
				cudacheckSYN();
			}
		}
	}
	*GMRES_n	 = GMRES_i;
	*h_res_vs_tol_p	 = (h_actual_residual > tolerance);

	clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_35_

	cudacall(cudaFree((cuComplex *)dev_orthogonal_basis));
	cudacall(cudaFree((int *)dev_info));
	clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_36_
}



void cuSolve_LES(	cuComplex *dev_A, 
			cuComplex *dev_B, 
			const int m, 
			cusolverDnHandle_t cusolverH, 
			cuComplex *dev_work, 
			int *dev_Ipiv, 
			int *dev_info, 
			timespec *h_computation_times, 
			unsigned int *clock_i_p)
/*	dev_work <- malloc(SIZE_OF_WORKSPACE * sizeof(cuComplex)) 	device workspace for getrf */
/*	lwork size of workspace  <- #define SIZE_OF_WORKSPACE (N >= 1024)?N/1024*100:100 */
/*	dev_Ipiv <- malloc(maxiter * sizeof(int)) 			pivoting sequence */
/* 	dev_info <- malloc(sizeof(int))					error info */
{
	const int lda = m;
	const int ldb = m;
	int h_info = 0;


	clock_gettime(CLOCK_REALTIME, h_computation_times + (*clock_i_p)++); //_32_

	/* step 2: LU factorization */
	cusolvercall(cusolverDnCgetrf(      cusolverH,
					    m,
					    m,
					    dev_A,
					    lda,
					    dev_work,
					    dev_Ipiv,
					    dev_info));
	cudacheckSYN();

	cudacall(cudaMemcpy(&h_info, dev_info, sizeof(int), cudaMemcpyDeviceToHost));

	if ( h_info != 0 )
	{
		printf("cusolverDnCgetrf failed!\tinfo = %i\n", h_info);
		exit(1);
	}

	cusolvercall(cusolverDnCgetrs(  cusolverH,
					CUBLAS_OP_N,
					m,
					1, /* nrhs */
					dev_A,
					lda,
					dev_Ipiv,
					dev_B,
					ldb,
					dev_info));
	cudacheckSYN();

	cudacall(cudaMemcpy(&h_info, dev_info, sizeof(int), cudaMemcpyDeviceToHost));

	if ( h_info != 0 )
	{
		printf("cusolverDnCgetrs failed!\tinfo = %i\n", h_info);
		exit(1);
	}

	clock_gettime(CLOCK_REALTIME, h_computation_times + (*clock_i_p)++); //_33_
}

void show_norm(	const char *description, 
		cublasHandle_t handle, 
		const cuComplex *dev_array, 
		const unsigned int size_array)
{
	float h_residual_0;
	cublasPointerMode_t mode_current;

	cublascall(cublasGetPointerMode((cublasHandle_t)handle, (cublasPointerMode_t *)&mode_current));
	cublascall(cublasSetPointerMode((cublasHandle_t)handle, CUBLAS_POINTER_MODE_HOST));
	cublascall(cublasScnrm2((cublasHandle_t)handle,
					size_array,
		                        (const cuComplex *)dev_array, 1,
					(float  *)&h_residual_0));

	fprintf(stderr, "norm(%s) = %6.12f\n", description, h_residual_0);

	cublascall(cublasSetPointerMode((cublasHandle_t)handle, mode_current));
}


void show_norm_F(	const char *description, 
			cublasHandle_t handle, 
			const float *dev_array, 
			const unsigned int size_array)
{
	float h_residual_0;
	cublasPointerMode_t mode_current;

	cublascall(cublasGetPointerMode((cublasHandle_t)handle, (cublasPointerMode_t *)&mode_current));
	cublascall(cublasSetPointerMode((cublasHandle_t)handle, CUBLAS_POINTER_MODE_HOST));
	cublascall(cublasSnrm2((cublasHandle_t)handle,
					size_array,
		                        (const float *)dev_array, 1,
					(float  *)&h_residual_0));

	fprintf(stderr, "norm(%s) = %6.12f\n", description, h_residual_0);

	cublascall(cublasSetPointerMode((cublasHandle_t)handle, mode_current));
}
