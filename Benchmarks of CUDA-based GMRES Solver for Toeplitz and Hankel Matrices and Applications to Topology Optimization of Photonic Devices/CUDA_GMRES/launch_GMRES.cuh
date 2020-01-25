#include <sstream>
#include <unistd.h>

unsigned int *get_n_timestamps_array_improved(unsigned int max_maxiter);

void launch_GMRES()
{		
	char buffer[1024];
	float tolerance = 0.001f;//0.2f;

	unsigned int max_st = 0;
	unsigned int max_en = 100;

	unsigned int pow_st =  8;
	unsigned int pow_en = 13;

	unsigned int max_maxiter = 50;

	unsigned int *n_timestamps_array = get_n_timestamps_array_improved((unsigned int)max_maxiter);


	cuComplex **p_h_anal_sols = (cuComplex **) malloc((pow_en - pow_st + 1) * sizeof(cuComplex *));
	bool **p_h_masks = (bool **) malloc((pow_en - pow_st + 1) * sizeof(bool *));
	cuComplex **p_h_gamma_arrays = (cuComplex **) malloc((pow_en - pow_st + 1) * sizeof(cuComplex *));

	for (unsigned int pow_cur = pow_st; pow_cur < pow_en + 5; pow_cur ++)
	{
		unsigned int N = 1 << pow_cur;
		p_h_anal_sols[pow_cur - pow_st] = (cuComplex *) malloc( N * N * sizeof(cuComplex) );

		std::string line;
		sprintf(buffer, "input/analytical_solution_%u.txt", N);
		std::ifstream analytical_solution_file (buffer);//Python_analytical_solution_%u
		if (analytical_solution_file.is_open())
		{
			unsigned int index = 0;
			while ( getline (analytical_solution_file, line) )
			{
				std::istringstream in_string_stream(line);

				in_string_stream >> p_h_anal_sols[pow_cur - pow_st][index].x >> p_h_anal_sols[pow_cur - pow_st][index].y;

				index++;
	
			}
			analytical_solution_file.close();
		}
		else
		{
			fprintf(stderr, "Unable to open file: %s\n", buffer);
			exit(1);
		}


		p_h_masks[pow_cur - pow_st] = (bool *) malloc(N * N * sizeof(bool));

		sprintf(buffer, "input/cylinder_%u.txt", N);
		std::ifstream myfile (buffer);
		if (myfile.is_open())
		{
			unsigned int index = 0;
			while ( getline (myfile,line) )
			{
				p_h_masks[pow_cur - pow_st][index++] = (line == "1");
			}
			myfile.close();
		}
		else {
			fprintf(stderr, "Unable to open file: %s\n", buffer);
			exit(1);
		}


		p_h_gamma_arrays[pow_cur - pow_st] = (cuComplex *)malloc((2 * N - 1) * (2 * N - 1) * sizeof(cuComplex));
	
		sprintf(buffer, "input/G_prep_%u.txt", N);
		get_array_C_to_CPU((cuComplex *)p_h_gamma_arrays[pow_cur - pow_st], (const char *)buffer);
	}


	for (unsigned int repetition_i = max_st; repetition_i < max_en; repetition_i ++)
	{	// int maxiter = 28;
		for (unsigned int maxiter = 3; maxiter < max_maxiter; maxiter ++)
		{
			for (unsigned int pow_cur = pow_st; pow_cur < pow_en + 1; pow_cur = pow_cur + 5) //Characteristic size of square matrix
			{
				unsigned int N = 1 << pow_cur;

				fprintf(stderr, "%i\n", N);
	
				dim3 blocks(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
				dim3 threads(Q, Q);
				cufftHandle plan;
				cublasHandle_t handle;
				cublascall(cublasCreate_v2(&handle));
				cufftcall(cufftPlan2d(&plan, 2 * N - 1, 2 * N - 1, CUFFT_C2C));
				cudaStream_t stream = NULL;
				cusolverDnHandle_t cusolverH = NULL;
				cusolvercall(cusolverDnCreate(&cusolverH));
				cudacall(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
				cusolvercall(cusolverDnSetStream(cusolverH, stream));

				bool *dev_mask;
				bool *h_mask = p_h_masks[pow_cur - pow_st];
				bool h_res_vs_tol = true;
				cuComplex *h_gamma_array = p_h_gamma_arrays[pow_cur - pow_st];
				cuComplex *h_analytical_solution = p_h_anal_sols[pow_cur - pow_st];
				cuComplex *dev_gamma_array;
				cuComplex *dev_analytical_solution;
				cuComplex *dev_solution;
				float *dev_actual_residual;
				float h_result = 0.f;
				float h_norm_analytical_solution = 0.f;
				unsigned int GMRES_n = 0;
				timespec *h_computation_times = (timespec *) malloc(n_timestamps_array[maxiter] * sizeof(timespec));
				cudacall(cudaSetDevice(0));

				cudacall(cudaMalloc((void**)&dev_mask, N * N * sizeof(bool)));
				cudacall(cudaMalloc((void**)&dev_solution, N * N * sizeof(cuComplex)));
				cudacall(cudaMalloc((void**)&dev_analytical_solution, N * N * sizeof(cuComplex)));




				cudacall(cudaMemcpy(dev_analytical_solution, h_analytical_solution, N * N * sizeof(cuComplex), cudaMemcpyHostToDevice));


				cublascall(cublasScnrm2(handle, N * N,
							(const cuComplex *)dev_analytical_solution, 1, 
							(float  *)&h_norm_analytical_solution));


				cudacall(cudaMemcpy(dev_mask, h_mask, N * N * sizeof(bool), cudaMemcpyHostToDevice));

			//	get_gamma_array((cuComplex **)&dev_gamma_array, (cufftHandle)plan);
			//	cudacall(cudaMemcpy(h_gamma_array, dev_gamma_array, (2 * N - 1) * (2 * N - 1) * sizeof(cuComplex), cudaMemcpyDeviceToHost));
			//==================================== Begin: get_gamma_array connected to MKL 2D Green's function values in Bessel function =========================
				cudacall(cudaMalloc((void**)&dev_gamma_array,  (2 * N - 1) * (2 * N - 1) * sizeof(cuComplex)));
				cudacall(cudaMemcpy(dev_gamma_array, h_gamma_array, (2 * N - 1) * (2 * N - 1) * sizeof(cuComplex), cudaMemcpyHostToDevice));

				cufftcall(cufftExecC2C(plan, (cuComplex *)dev_gamma_array, (cuComplex *)dev_gamma_array, CUFFT_FORWARD));
				cudacheckSYN();
			//==================================== End: get_gamma_array connected to MKL 2D Green's function values in Bessel function =========================

				time_t clock_time;
				float diff_time = 0.f;
				float diff_average = 0.f;
				cuComplex alpha;
				alpha.x = -1.f;
				alpha.y = 0.f;
				const cuComplex *p_alpha = &alpha;

				{

					cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
					fprintf(stderr, "maxiter = %i\trepetition_i = %i\n", maxiter, repetition_i);

					init_x0_kernel <<< blocks, threads >>> ((cuComplex *)dev_solution, N);
					cudacheckSYN();

					memset(h_computation_times, 0, n_timestamps_array[maxiter] * sizeof(timespec));

					clock_time = clock();

					Fast_GMRES_with_CUDA(	(const cuComplex *)dev_gamma_array,
								(const bool *)dev_mask,
								(cuComplex *)dev_solution,
								(float **)&dev_actual_residual,
								(unsigned int *)&GMRES_n,
								(cufftHandle)plan,
								(cublasHandle_t *)&handle,
								tolerance, false, 0,
								(bool *)&h_res_vs_tol,
 								maxiter,
								(cusolverDnHandle_t)cusolverH,
								(timespec *)h_computation_times, N);

					diff_time = (float)(clock() - clock_time) / (float)(CLOCKS_PER_SEC);
				}

				{
					fprintf(stderr, "Files writing\n");
		
					sprintf(buffer, "time_%u/solution_sample", N);
					save_test_GPU((char *)buffer, (cuComplex *)dev_solution, maxiter * 100 + repetition_i, N * N);
					fprintf(stderr, "diff_time = %f\n", diff_time);

					sprintf(buffer, "time_%u/maxiter", N);
					save_test_F_CPU((char *)buffer, (float *)&diff_time, maxiter * 100 + repetition_i, 1);
					sprintf(buffer, "time_%u/residual", N);
					save_test_F_GPU((char *)buffer, (float *)dev_actual_residual + GMRES_n, maxiter * 100 + repetition_i, 1);
					sprintf(buffer, "time_%u/times", N);
					save_test_timespec_CPU((char *)buffer, (timespec *)h_computation_times, maxiter * 100 + repetition_i, n_timestamps_array[maxiter]);

					cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);

					cublascall(cublasScnrm2(handle, N * N,
								    (const cuComplex *)dev_solution, 1, (float  *)&h_result));

					fprintf(stderr, "Norm of solution = %f\n", h_result);


					cublascall(cublasCaxpy(handle, N * N,
								   (const cuComplex *)p_alpha,
								   (const cuComplex *)dev_analytical_solution, 1,
								   (cuComplex *)dev_solution, 1));


					cublascall(cublasScnrm2(handle, N * N,
								    (const cuComplex *)dev_solution, 1, (float  *)&h_result));

					fprintf(stderr, "Norm of diff = %f\n", h_result);

					h_result = h_result / h_norm_analytical_solution;

					fprintf(stderr, "File relative_error writing\t%f\n", h_result);
					sprintf(buffer, "time_%u/relative_error", N);
					save_test_F_CPU((char *)buffer, (float *)&h_result, maxiter * 100 + repetition_i, 1);
				}

				fprintf(stderr, "diff = %f\n", diff_average);

	//			saveGPUrealtxt_C(dev_solution, "/output/solution.txt", N * N);

				cudacall(cudaFree((bool *)dev_mask));
				cudacall(cudaFree((cuComplex *)dev_solution));
				cudacall(cudaFree((cuComplex *)dev_gamma_array));
				cudacall(cudaFree((cuComplex *)dev_analytical_solution));
				cufftcall(cufftDestroy(plan));
				cusolverDnDestroy(cusolverH);
				free((timespec *)h_computation_times);
				cublascall(cublasDestroy_v2(handle));
                                cudacall(cudaFree((float *)dev_actual_residual));
			}
		}
	}

	free(n_timestamps_array);
}





unsigned int get_n_timestamps_val_improved(unsigned int maxiter) //Comparables/new
{
    unsigned int n_timestamps  = 1; //short_indexed_text_array = []
    n_timestamps ++; //short_indexed_text_array.append("Initialization (malloc)") #_1_ !_
    n_timestamps ++; //short_indexed_text_array.append("G_x_fft_matvec for A*x0") #_2_ !_
    n_timestamps ++; //short_indexed_text_array.append("2D_to_1D for A*x0-x0") #_3_
    n_timestamps ++; //short_indexed_text_array.append("Norm(residual_vec)") #_4_
    n_timestamps ++; //short_indexed_text_array.append("Condition to iterate") #_5_ !_
    n_timestamps ++; //short_indexed_text_array.append("Residual_normalization & set_a,b") #_6_
    
    unsigned int GMRES_i = 1;
    
    if (1)
    {
        n_timestamps ++; //short_indexed_text_array.append("Memset(H, 0)") #_7_ !_
        n_timestamps ++; //short_indexed_text_array.append("G_x_fft_matvec for w=A*v iteration(" + str(GMRES_i) + ")") #_8_
        n_timestamps ++; //short_indexed_text_array.append("2D_to_1D for w=A*v iteration(" + str(GMRES_i) + ")") #_9_
        n_timestamps ++; //short_indexed_text_array.append("H_jk = (V_j, w) iteration(" + str(GMRES_i) + ")") #_10_
        n_timestamps ++; //short_indexed_text_array.append("w = w - H*v iteration(" + str(GMRES_i) + ")") #_11_ !_    
        n_timestamps ++; //short_indexed_text_array.append("H_jj+1 = norm(w) iteration(" + str(GMRES_i) + ")") #_12_    
        n_timestamps ++; //short_indexed_text_array.append("1/H_jj+1 iteration(" + str(GMRES_i) + ")") #_13_    
        n_timestamps ++; //short_indexed_text_array.append("w = w/H_jj+1 iteration(" + str(GMRES_i) + ")") #_14_
        n_timestamps ++; //short_indexed_text_array.append("Set(J) iteration(" + str(GMRES_i) + ")") #_15_ !_
        n_timestamps ++; //short_indexed_text_array.append("Set(Jtotal) iteration(" + str(GMRES_i) + ")") #_16_ !_
        n_timestamps ++; //short_indexed_text_array.append("Update residual iteration(" + str(GMRES_i) + ")") #_17_ !_
        
        for (GMRES_i = 1; GMRES_i < maxiter; GMRES_i ++)
        {  
            n_timestamps ++; //short_indexed_text_array.append("Condition_check iteration(" + str(GMRES_i) + ")") #_18_
            n_timestamps ++; //short_indexed_text_array.append("G_x_fft_matvec for w=A*v iteration(" + str(GMRES_i) + ")") #_19_        
            n_timestamps ++; //short_indexed_text_array.append("2D_to_1D for w=A*v iteration(" + str(GMRES_i) + ")") #_20_     
                
            for (unsigned int j = 0; j < GMRES_i + 1; j ++)
            {
                n_timestamps ++; //short_indexed_text_array.append("H_jk = (V_j, w) iteration(" + str(GMRES_i) + ", j = " + str(j) + ")") #_21_
                n_timestamps ++; //short_indexed_text_array.append("w = w - H_jk * V_j iteration(" + str(GMRES_i) + ", j = " + str(j) + ")") #_22_  
            }       
                
            n_timestamps ++; //short_indexed_text_array.append("H_jj+1 = norm(w) iteration(" + str(GMRES_i) + ")") #_23_
            n_timestamps ++; //short_indexed_text_array.append("1/H_jj+1 iteration(" + str(GMRES_i) + ")") #_24_
            n_timestamps ++; //short_indexed_text_array.append("w = w/H_jj+1 iteration(" + str(GMRES_i) + ")") #_25_    
            n_timestamps ++; //short_indexed_text_array.append("H_temp=Jtotal * H iteration(" + str(GMRES_i) + ")") #_26_
            n_timestamps ++; //short_indexed_text_array.append("Set(J) iteration(" + str(GMRES_i) + ")") #_27_ !_
            n_timestamps ++; //short_indexed_text_array.append("Jtotal = J*Jtotal iteration(" + str(GMRES_i) + ")") #_28_
            n_timestamps ++; //short_indexed_text_array.append("Update residual iteration(" + str(GMRES_i) + ")") #_29_ !_
        }
    }
            
    n_timestamps ++; //short_indexed_text_array.append("HH = Jtotal * H") #_30_
    n_timestamps ++; //short_indexed_text_array.append("cc <- Jtotal") #_31_
    n_timestamps ++; //short_indexed_text_array.append("Initialize_small_LES(HH, cc)") #_32_
    n_timestamps ++; //short_indexed_text_array.append("Process_small_LES(HH, cc)") #_33_
    
    for (unsigned int j = 0; j < GMRES_i; j++)
    {        
        n_timestamps ++; //short_indexed_text_array.append("Add iteration(j = " + str(j) + ")") #_34_
    }
        
    n_timestamps ++; //short_indexed_text_array.append("set(Output_p)") #_35_        
    n_timestamps ++; //short_indexed_text_array.append("Free in postprocessing") #_36_
    
    return n_timestamps;
}

unsigned int *get_n_timestamps_array_improved(unsigned int max_maxiter)
{
    unsigned int *n_timestamps_array = (unsigned int *)malloc(max_maxiter * sizeof(unsigned int));

    for (unsigned int maxiter = 0; maxiter < max_maxiter; maxiter ++)
    {
        n_timestamps_array[maxiter] = get_n_timestamps_val_improved((unsigned int)maxiter);
    }
    return n_timestamps_array;
}
