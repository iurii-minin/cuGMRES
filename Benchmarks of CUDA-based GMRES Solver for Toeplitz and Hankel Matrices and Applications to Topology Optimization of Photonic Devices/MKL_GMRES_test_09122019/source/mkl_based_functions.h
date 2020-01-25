void G_x_fft_matvec(complex8 *mkl_gamma_array, bool *mkl_mask, complex8 *mkl_solution, complex8 **mkl_solution_matrix_extended)
{
	extend_by_zeros_MKL((bool *)mkl_mask, (complex8 *)mkl_solution, (complex8 *)*mkl_solution_matrix_extended);

	DFTI_DESCRIPTOR_HANDLE my_desc1_handle;
	MKL_LONG l[2] = {2 * N - 1, 2 * N - 1};

	dfticall(DftiCreateDescriptor( &my_desc1_handle, DFTI_SINGLE, DFTI_COMPLEX, 2, l));
	dfticall(DftiCommitDescriptor( my_desc1_handle));
	dfticall(DftiComputeForward( my_desc1_handle, *mkl_solution_matrix_extended));

	vcMul((const MKL_INT)(2 * N - 1) * (2 * N - 1), (MKL_Complex8 *)mkl_gamma_array, (MKL_Complex8 *)*mkl_solution_matrix_extended, (MKL_Complex8 *)*mkl_solution_matrix_extended);

	dfticall(DftiComputeBackward( my_desc1_handle, *mkl_solution_matrix_extended));
	dfticall(DftiFreeDescriptor(&my_desc1_handle));
}


void get_gamma_array(complex8 **mkl_gamma_array)
{
	*mkl_gamma_array = (complex8 *) malloc((2 * N - 1) * (2 * N - 1) * sizeof(complex8));

	Green_matrix_create_MKL((complex8*)(*mkl_gamma_array));	
	
	DFTI_DESCRIPTOR_HANDLE my_desc1_handle;
	MKL_LONG l[2] = {2 * N - 1, 2 * N - 1};

	dfticall(DftiCreateDescriptor( &my_desc1_handle, DFTI_SINGLE, DFTI_COMPLEX, 2, l));
	dfticall(DftiCommitDescriptor( my_desc1_handle));

	dfticall(DftiComputeForward( my_desc1_handle, *mkl_gamma_array));

	dfticall(DftiFreeDescriptor(&my_desc1_handle));
}

void get_resized(complex8 **mkl_to_be_resized, unsigned int old_size_i, unsigned int old_size_j, unsigned int new_size_i, unsigned int new_size_j)
{
	complex8 *mkl_resized = (complex8 *)malloc(new_size_i * new_size_j * sizeof(complex8));
	
	resize_MKL((complex8 *)(*mkl_to_be_resized), (unsigned int)old_size_i, (unsigned int)old_size_j, (unsigned int)new_size_i,  (unsigned int)new_size_j, (complex8 *)mkl_resized);

	free((complex8 *)*mkl_to_be_resized);

	(*mkl_to_be_resized) = mkl_resized;
}

void usual_MatMul_MKL(complex8 *A, complex8 *B, complex8 *C, unsigned int n, unsigned int k, unsigned int m)
{
	unsigned int lda = k, ldb = m;
	unsigned int ldc = (lda > ldb) ? ldb : lda;
	complex8 alf;
	alf.x = 1.f;
	alf.y = 0.f;
	complex8 bet;
	bet.x = 0.f;
	bet.y = 0.f;
	const complex8 *alpha = &alf;
	const complex8 *beta = &bet;

	cblas_cgemm3m(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, m, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

void get_new_solution_MKL(complex8 *mkl_cc, complex8 *mkl_HH)
{
	float dominant = pow((float)(mkl_HH->x), 2.f) + pow((float)(mkl_HH->y), 2.f);
	complex8 current;
	current.x = (mkl_cc->x * mkl_HH->x + mkl_cc->y * mkl_HH->y) / dominant;
	current.y = (mkl_cc->y * mkl_HH->x - mkl_cc->x * mkl_HH->y) / dominant;
	(*mkl_cc) = current;
}

void mkl_Solve_LES(complex8 *mkl_A, complex8 *mkl_B, const int m, int *mkl_Ipiv, timespec *h_computation_times, unsigned int *clock_i_p)
{
	const int lda = m;
	const int ldb = m;

	clock_gettime(CLOCK_REALTIME, h_computation_times + (*clock_i_p)++); //_32_

	/* LU factorization */
	lapackcall(LAPACKE_cgetrf(LAPACK_COL_MAJOR, m, m, (MKL_Complex8 *)mkl_A, lda , (int *)mkl_Ipiv));

	/* computing the solution */
	lapackcall(LAPACKE_cgetrs(LAPACK_COL_MAJOR, 'T', m, 1, (const MKL_Complex8 *)mkl_A, lda, (const int *)mkl_Ipiv, (MKL_Complex8 *)mkl_B, ldb));

	clock_gettime(CLOCK_REALTIME, h_computation_times + (*clock_i_p)++); //_33_
}
