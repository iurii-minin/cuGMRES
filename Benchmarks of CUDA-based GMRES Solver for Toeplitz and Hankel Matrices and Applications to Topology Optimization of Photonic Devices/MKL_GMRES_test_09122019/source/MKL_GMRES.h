void print_matrix_F(const char *desc, int m, int n, float *mkl_mat, int lda );

void FFT_GMRES_with_MKL(const complex8 *mkl_gamma_array, 
			const bool     *mkl_mask, 
			complex8       *mkl_solution, 
			float         **mkl_actual_residual, 
			unsigned int   *GMRES_n, 
			const float     tolerance,
			bool           *mkl_res_vs_tol_p,
			unsigned int    maxiter,
			timespec       *h_computation_times)
{
	unsigned int clock_i = 0;

	clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_0_

	complex8 *mkl_orthogonal_basis		 = NULL;
	complex8 *mkl_HH			 = NULL;
	complex8 *mkl_Htemp			 = NULL;
	complex8 *mkl_extended			 = NULL; //mkl_w_extended? //mkl_solution_matrix_extended?
//	complex8 *mkl_vec_resudual		 = NULL;
	complex8 *mkl_H_			 = NULL;
	complex8 *mkl_Jtotal			 = NULL;
	complex8 *mkl_cc 			 = NULL;
	complex8 *mkl_Givens_rotation 		 = NULL; //mkl_w?
	complex8 *mkl_Givens_rotation_0	 	 = NULL;
	complex8 *mkl_Givens_rotation_1	 	 = NULL;
	complex8 *mkl_Givens_rotation_2	 	 = NULL;
	complex8 *mkl_Givens_rotation_3	 	 = NULL;
	complex8 *mkl_alpha			 = NULL;
	complex8 *mkl_beta			 = NULL;

	int *mkl_info 				 = NULL;
	int *mkl_Ipiv 				 = NULL;

	unsigned int GMRES_i = 0;
	const unsigned int maxiter_plus_1 = maxiter + 1;

	*mkl_actual_residual = (float *)malloc(maxiter_plus_1 * sizeof(float));
	mkl_orthogonal_basis = (complex8 *)malloc(maxiter_plus_1 * N * N * sizeof(complex8) );
	mkl_info = (int *)malloc(maxiter_plus_1 * N * N * sizeof(int));


	mkl_H_ = (complex8 *)malloc((6 * N * N - 8 * N + 1) * sizeof(complex8));
//	mkl_H_ = mkl_orthogonal_basis + maxiter_plus_1 * N * N;
	mkl_Jtotal = mkl_H_ + maxiter * maxiter_plus_1;

//	mkl_vec_resudual = mkl_H_;
	
	mkl_extended = mkl_HH = mkl_Htemp = mkl_H_ + N * N;

	mkl_cc = mkl_extended + maxiter * maxiter;
	mkl_Givens_rotation = mkl_Jtotal + maxiter_plus_1 * maxiter_plus_1;
	mkl_Givens_rotation_3 = mkl_Givens_rotation + maxiter_plus_1 * maxiter_plus_1 - 1;
	mkl_Givens_rotation_2 = mkl_Givens_rotation_3 - 1;
	mkl_Givens_rotation_1 = mkl_Givens_rotation_2 - maxiter;
	mkl_Givens_rotation_0 = mkl_Givens_rotation_1 - 1;
	mkl_alpha  = mkl_Givens_rotation + maxiter_plus_1 * maxiter_plus_1;
	mkl_beta   = mkl_alpha + 1;

	mkl_Ipiv = mkl_info + 1;
							
	clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_1_ //Initialization
//========================================= BEGIN: get_residual_vector =======================================================
	G_x_fft_matvec(	(complex8 *)mkl_gamma_array, 
			(bool *)mkl_mask, 
			(complex8 *)mkl_solution, 
			(complex8 **)&mkl_extended);

	clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_2_

	_2D_to_1D_compared_MKL(	(complex8 *)mkl_solution, 
				(complex8 *)mkl_extended, 
				(complex8 *)mkl_orthogonal_basis);

	clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_3_
//========================================== END: get_residual_vector =========================================================
//============================================BEGIN:Get norm of residual =======================================================
	(*mkl_actual_residual)[0] = cblas_scnrm2(	N * N, 
							(const complex8 *)mkl_orthogonal_basis, 1);



	fprintf(stderr, "check: B\n");

	clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_4_
//============================================END:Get norm of residual =======================================================
//============================================= Begin: Condition to iterate ==========================================================
	float mkl_residual_initial = (*mkl_actual_residual)[0];

	clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_5_



//=============================================== End: Condition to iterate ===========================================================
//============================================BEGIN:residual_normalization =======================================================	
	cblas_csscal(N * N, (const float)(1.f / mkl_residual_initial), (complex8 *)mkl_orthogonal_basis, 1);

	mkl_alpha->x = 1.f;
	mkl_alpha->y = 0.f;

	mkl_beta->x = 0.f;
	mkl_beta->y = 0.f;

	clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_6_


	fprintf(stderr, "mkl_residual_initial = %f\n", mkl_residual_initial);


	fprintf(stderr, "check: C\n");
//============================================END:residual_normalization =======================================================
	if (mkl_residual_initial > tolerance)
	{
		fprintf(stderr, "GMRES_i = %i\n", GMRES_i);

		memset(mkl_H_, 0, maxiter_plus_1 * maxiter * sizeof(complex8));

		clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_7_

		G_x_fft_matvec(	(complex8 *)mkl_gamma_array, 
				(bool *)mkl_mask, 
				(complex8 *)mkl_orthogonal_basis, 
				(complex8 **)&mkl_extended);

		clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_8_

		_2D_to_1D_MKL(	(complex8 *)mkl_orthogonal_basis, 
				(complex8 *)mkl_extended, 
				(complex8 *)mkl_orthogonal_basis + N * N);

		clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_9_

		cblas_cdotc_sub(	N * N,
					(const complex8 *)mkl_orthogonal_basis, 1,
					(const complex8 *)mkl_orthogonal_basis + N * N, 1,
					(complex8 *)mkl_H_);

		clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_10_

		complex8 temporary = *mkl_H_;
		temporary.x = -temporary.x;
		temporary.y = -temporary.y;

		cblas_caxpy(	N * N, 
				(const complex8 *)&temporary, 
				(const complex8 *)mkl_orthogonal_basis, 1, 
				(complex8 *)mkl_orthogonal_basis + N * N, 1);
		




		clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_11_

		mkl_H_[maxiter].x = cblas_scnrm2(N * N, 
						(const complex8 *)mkl_orthogonal_basis + N * N, 1);
		mkl_H_[maxiter].y = 0.f;

		clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_12_
	//============================================== BEGIN: Fill Orthogonal Basis matrix ============================================
		temporary.x = 1.f / mkl_H_[maxiter].x - 1;
		temporary.y = 0.f;

		clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_13_

		cblas_caxpy(	N * N, 
				(const complex8 *)&temporary, 
				(const complex8 *)mkl_orthogonal_basis + N * N, 1, 
				(complex8 *)mkl_orthogonal_basis + N * N, 1);

		clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_14_
	//============================================== END: Orthogonal Basis matrix  ==================================================
	//============================================= BEGIN: Create Givens_Rotation_Matrix ========================================
		set_Identity_matrix_kernel(	(complex8 *)mkl_Givens_rotation, 
						(unsigned int)maxiter_plus_1);

		clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_15_
	//=============================================== END: Create Givens_Rotation_Matrix,  ========================================


	//============================================= BEGIN: Create Jtotal_Matrix ========================================
		set_first_Jtotal_kernel((complex8 *)mkl_Jtotal,
					(complex8 *)mkl_H_,
					maxiter,
					maxiter_plus_1);

		clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_16_
	//=============================================== END: Create Jtotal_Matrix,  ========================================
	//===================================================== BEGIN: Update residual ======================================================
		(*mkl_actual_residual)[1] =(*mkl_actual_residual)[0] * sqrt( (pow((float)(mkl_Jtotal[maxiter_plus_1].x), 2.0f) + pow((float)(mkl_Jtotal[maxiter_plus_1].y), 2.0f)));

		clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_17_
	//======================================================= END: Update residual ======================================================
		GMRES_i ++;


		fprintf(stderr, "residual = %f\n", (*mkl_actual_residual)[0] );
		fprintf(stderr, "residual = %f\n", (*mkl_actual_residual)[1] );
		fprintf(stderr, "mkl_H_[maxiter].x = %f\n", mkl_H_[maxiter].x );



		for(GMRES_i = 1; ((GMRES_i < maxiter)); GMRES_i ++) //((*mkl_actual_residual)[?] > tolerance) && 
		{
			clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_18_
			fprintf(stderr, "GMRES_i = %i\n", GMRES_i);
		//======================================================= BEGIN: w = A*v (w_equal_to_A_v) ======================================================
			G_x_fft_matvec(	(complex8 *)mkl_gamma_array, 
					(bool *)mkl_mask, 
					(complex8 *)(mkl_orthogonal_basis + GMRES_i * N * N), 
					(complex8 **)&mkl_extended);

			clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_19_

			_2D_to_1D_MKL(	(complex8 *)mkl_orthogonal_basis + GMRES_i * N * N, 
					(complex8 *)mkl_extended, 
					(complex8 *)mkl_orthogonal_basis + (GMRES_i + 1) * N * N);




			clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_20_
		//======================================================= END: w = A*v (w_equal_to_A_v) ======================================================
			for(unsigned int j = 0; j < GMRES_i + 1; j++)
			{
				cblas_cdotc_sub(	N * N, 
							(const complex8 *)mkl_orthogonal_basis + j * N * N, 1, 
							(const complex8 *)mkl_orthogonal_basis + (GMRES_i + 1) * N * N, 1, 
							(complex8 *)mkl_H_ + j * maxiter + GMRES_i);

				clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_21_

				complex8 temporary = mkl_H_[j * maxiter + GMRES_i];
				temporary.x = -temporary.x;
				temporary.y = -temporary.y;
/*
				if (GMRES_i == 2)
				{
					fprintf(stderr, "mkl_H_[j * maxiter + GMRES_i] = %f\n", temporary.x);
				}
				
*/
				cblas_caxpy(	N * N, 
						(const complex8 *)&temporary, 
						(const complex8 *)mkl_orthogonal_basis + j * N * N, 1, 
						(complex8 *)mkl_orthogonal_basis + (GMRES_i + 1) * N * N, 1);

				clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_22_
			}

			//Next, fill Hessenberg matrix
			mkl_H_[(GMRES_i + 1) * maxiter + GMRES_i].x = cblas_scnrm2(	N * N, 
											(const complex8 *)mkl_orthogonal_basis + (GMRES_i + 1) * N * N, 1);
			mkl_H_[(GMRES_i + 1) * maxiter + GMRES_i].y = 0.f;



			clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_23_
		//============================================== BEGIN: Fill Orthogonal Basis m.============================================
			temporary.x = 1.f / mkl_H_[(GMRES_i + 1) * maxiter + GMRES_i].x - 1;
			temporary.y = 0.f;

			clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_24_

			cblas_caxpy(	N * N, 
					(const complex8 *)&temporary, 
					(const complex8 *)mkl_orthogonal_basis + (GMRES_i + 1) * N * N, 1, 
					(complex8 *)mkl_orthogonal_basis + (GMRES_i + 1) * N * N, 1);

			clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_25_
		//===============================    END: Fill Orthogonal Basis m.  ===========================================
		//============================================== Begin: Least Squares Step =========================================================
		//================================ BEGIN: MATMUL (H_temp=Jtotal * H) ==============================================
			cblas_cgemm3m(	CblasRowMajor, 
					CblasNoTrans, 
					CblasNoTrans, 
					(unsigned int)GMRES_i + 2,
					(unsigned int)GMRES_i + 1,
					(unsigned int)GMRES_i + 2,
					(const complex8 *)mkl_alpha, 
					(complex8 *)mkl_Jtotal, (unsigned int)maxiter_plus_1, 
					(complex8 *)mkl_H_, 	(unsigned int)maxiter, 
					(const complex8 *)mkl_beta, 
					(complex8 *)mkl_Htemp, 	(unsigned int)GMRES_i + 1);

			clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_26_
		//================================== END: MATMUL (H_temp=Jtotal * H) ===============================================
		//================================================ END: Least Squares Step =========================================================
		//============================================= BEGIN: Create Givens_Rotation_Matrix ========================================
			set_4_Givens_rotation_matrix_elements_kernel(	(complex8 *)mkl_Htemp,
									maxiter_plus_1,
									(complex8 *)mkl_Givens_rotation_0,
									(complex8 *)mkl_Givens_rotation_1,
									(complex8 *)mkl_Givens_rotation_2,
									(complex8 *)mkl_Givens_rotation_3,
									GMRES_i + 1);







			clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_27_
		//=============================================== END: Create Givens_Rotation_Matrix ========================================
		//================================================== BEGIN: Jtotal = J*Jtotal =================================================
			cblas_cgemm3m(	CblasRowMajor,
					CblasNoTrans,
					CblasNoTrans, 
					(unsigned int)GMRES_i + 2,
					(unsigned int)GMRES_i + 2,
					(unsigned int)GMRES_i + 2,
					(const complex8 *)mkl_alpha, 
					(complex8 *)mkl_Givens_rotation + (maxiter - 1 - GMRES_i) * (maxiter_plus_1 + 1), 
					(unsigned int)maxiter_plus_1, 
					(complex8 *)mkl_Jtotal,	(unsigned int)maxiter_plus_1, 
					(const complex8 *)mkl_beta, 
					(complex8 *)mkl_Jtotal,	(unsigned int)maxiter_plus_1);

			clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_28_


		//==================================================== END: Jtotal = J*Jtotal =================================================
		//===================================================== BEGIN: Update residual ======================================================
			(*mkl_actual_residual)[GMRES_i + 1] = (*mkl_actual_residual)[0] * 
								sqrt( 	 (pow((float)(mkl_Jtotal[maxiter_plus_1 * (GMRES_i + 1)].x), 2.0f) 
									+ pow((float)(mkl_Jtotal[maxiter_plus_1 * (GMRES_i + 1)].y), 2.0f)));

			clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_29_


			fprintf(stderr, "residual = %f\n", (*mkl_actual_residual)[GMRES_i + 1]);
		//======================================================= END: Update residual ======================================================
		}

	//================================================== BEGIN: HH = (Jtotal*H)_resized ==========================================================
		cblas_cgemm3m(	CblasRowMajor, 
				CblasNoTrans, 
				CblasNoTrans, 
				(unsigned int)GMRES_i,
				(unsigned int)GMRES_i,
				(unsigned int)GMRES_i + 1,
				(const complex8 *)mkl_alpha, 
				(complex8 *)mkl_Jtotal,	(unsigned int)maxiter_plus_1, 
				(complex8 *)mkl_H_, (unsigned int)maxiter, 
				(const complex8 *)mkl_beta, 
				(complex8 *)mkl_HH,	(unsigned int)GMRES_i);

		clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_30_
	//===================================================== END: HH = (Jtotal*H)_resized ==========================================================
	//================================================= BEGIN: cc = Jtotal * norm_res_vec =========================================================
		get_cc_MKL(	(complex8 *)mkl_cc, 
				(complex8 *)mkl_Jtotal, 
				(float *)(*mkl_actual_residual), GMRES_i);

		clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_31_
	//=================================================== END: cc = Jtotal * norm_res_vec =========================================================
		if (GMRES_i > 0)
		{
			if (GMRES_i < 2)
			{
				get_new_solution_MKL((complex8 *)mkl_cc, (complex8 *)mkl_HH);

				get_solution_MKL((complex8 *)mkl_solution, (complex8 *)mkl_cc, (complex8 *)mkl_orthogonal_basis);
			}
			else
			{
			//============================================ BEGIN: Find solution to the LES(cc_new) for HH*cc_new=cc ============================================
				mkl_Solve_LES(	(complex8 *)mkl_HH,
						(complex8 *)mkl_cc, GMRES_i,
						(int *)mkl_Ipiv,
						(timespec *)h_computation_times,
						(unsigned int *)&clock_i);
			//============================================ END: Find solution to the LES(cc_new) for HH*cc_new=cc ===========================================
			//============================================ BEGIN: x = x0 + V * cc ===========================================
				for(unsigned int j = 0; j < GMRES_i; j++)
				{
					add_MKL((complex8 *)mkl_solution, (complex8 *)mkl_orthogonal_basis + j * N * N, (complex8 *)mkl_cc + j);

					clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_34_
				}
			}
		}
	}
	*GMRES_n	  = GMRES_i;
	*mkl_res_vs_tol_p = ((*mkl_actual_residual)[GMRES_i - 1] > tolerance);

	clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_35_

	free((complex8 *)mkl_orthogonal_basis);
	free((int *)mkl_info);
	free((complex8 *)mkl_H_);
	clock_gettime(CLOCK_REALTIME, h_computation_times + clock_i++); //_36_
}


void print_matrix_F(const char *desc, int m, int n, float *mkl_mat, int lda ) {
        int i, j;

        printf( "\nFor %s\n", desc );
        for( i = 0; i < m; i++ ) {
                for( j = 0; j < n; j++ ) 
			if (mkl_mat[i * lda + j] == 0){
				printf( "\t." );
			}else{
				printf( "\t%6.2f", mkl_mat[i * lda + j] );
			}
                printf( "\n" );
        }
}
