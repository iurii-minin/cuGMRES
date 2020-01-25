#define BILLION 1000000000LL

void print_matrix(const char *desc, int m, int n, double* a, int lda ) {
        int i, j;
        printf( "\n %s\n", desc );
        for( i = 0; i < m; i++ ) {
                for( j = 0; j < n; j++ ) printf( " %6.2f", a[i+j*lda] );
                printf( "\n" );
        }
}

void print_matrix_C(const char *desc, int m, int n, complex8 *a, int lda ) {
        int i, j;
        fprintf(stderr, "\nRe for %s\n", desc );
        for( i = 0; i < m; i++ ) {
                for( j = 0; j < n; j++ ) fprintf(stderr, " %6.2f", a[i+j*lda].x );
                fprintf(stderr, "\n" );
        }

        fprintf(stderr, "\nIm for %s\n", desc );

        for( i = 0; i < m; i++ ) {
                for( j = 0; j < n; j++ ) fprintf(stderr, " %6.2f", a[i+j*lda].y);
                fprintf(stderr, "\n" );
        }
}

void print_int_vector(const char *desc, int n, int* a ) {
        int j;
        printf( "\n %s\n", desc );
        for( j = 0; j < n; j++ ) printf( " %6i", a[j] );
        printf( "\n" );
}

complex8 my_cexpf(complex8 z) //FOR ONLY z.x = 0.f;
{
	complex8 res;
	res.x = E0 * cos(z.y);
	res.y = E0 * sin(z.y);
	return res;
}

void init_x0_MKL(complex8 *mkl_x0_c) {
	complex8 Input_Field;

	for(unsigned int i = 0; i < N; i ++)
	{
		for(unsigned int j = 0; j < N; j ++)
		{
			Input_Field.y = - WAVE_NUMBER * (i * cos(ALPHA) + j * sin(ALPHA));
			Input_Field = my_cexpf(Input_Field);
			mkl_x0_c[i * N + j] = Input_Field;
		}
	}
}

void extend_by_zeros_MKL(bool *mkl_mask, complex8 *mkl_usual, complex8 *mkl_extended)
{
	complex8 current;
	unsigned int index = 0;
	unsigned int index_extended = 0;

	memset(mkl_extended, 0, (2 * N - 1) * (2 * N - 1) * sizeof(complex8));

	for(unsigned int i = 0; i < N; i ++)
	{
		for(unsigned int j = 0; j < N; j ++)
		{
			if (mkl_mask[index])
			{
				current.x = CHI * mkl_usual[index].x / (2 * N - 1);
				current.y = CHI * mkl_usual[index].y / (2 * N - 1);

				mkl_extended[index_extended] = current;
			}

			index++;
			index_extended ++;
		}
		index_extended += N - 1; 
	}
}


void get_mask_on_path(const char *file_path, bool *mkl_mask)
{
//	std::string line;
//	std::ifstream myfile (file_path);
//	if (myfile.is_open())
//	{
//		unsigned int index = 0;
//		while ( getline (myfile,line) )
//		{
//			mkl_mask[index++] = (line == "1");
//		}
//		myfile.close();
//	}
	FILE *fp;
	char ch;

	fp = fopen(file_path, "r");

	if (fp == NULL)
	{
		perror("Error while opening the file.\n");
		exit(EXIT_FAILURE);
	}

	unsigned int index = 0;
	while((ch = fgetc(fp)) != EOF)
	{
		if ((ch == '0') || (ch == '1'))
		{
			if (ch == '1')
			{
				mkl_mask[index] = true;
			}
			else
			{
				mkl_mask[index] = false;
			}
			index++;
		}
	}
}


void Green_matrix_create_MKL(complex8 *mkl_gamma_array)
{
	complex8 current;
	unsigned int index_extended = 0;
	float kr_ij;
	
	for(unsigned int i = 0; i < N; i ++)
	{
		for(unsigned int j = 0; j < N; j ++)
		{

			kr_ij = WAVE_NUMBER * sqrt(pow((float)(i) - 0.5f, 2.f) + pow((float)(j), 2.f));

			current.x = -0.25f * y0(kr_ij);
			current.y =  0.25f * j0(kr_ij);
			mkl_gamma_array[index_extended] = current;

//			if ((i == 0) && (j == 0))
//			{
//				fprintf(stderr, "%f + i * %f\n", current.x, current.y);
//			}

			if ((i > 0) and (j > 0))
			{
				mkl_gamma_array[(2 * N - 1) * (2 * N - 1 - i) + (2 * N - 1 - j)] = current;
			}
			    
			if (i > 0)
			{
				mkl_gamma_array[(2 * N - 1) * (2 * N - 1 - i) + j] = current;
			}
			    
			if (j > 0)
			{
				mkl_gamma_array[(2 * N - 1) * i + (2 * N - 1 - j)] = current;
			}
			index_extended ++;
		}
		index_extended += N - 1;
	}
}

void mkl_array_C_to_file(const char *filepath, complex8 *mkl_array_c, unsigned int datasize)
{
	FILE *fp;

	fp = fopen(filepath, "w+");

//	datasize = 1;
	for(unsigned int i = 0; i < datasize; i ++)
	{
		fprintf(fp, "%.12f %.12f\n", mkl_array_c[i].x, mkl_array_c[i].y);
	}

	fclose(fp);
}

void mkl_array_F_to_file(const char *filepath, float *mkl_array_f, unsigned int datasize)
{
	FILE *fp;

	fp = fopen(filepath, "w+");

//	datasize = 1;
	for(unsigned int i = 0; i < datasize; i ++)
	{
		fprintf(fp, "%.12f\n", mkl_array_f[i]);
	}

	fclose(fp);
}

void mkl_array_I_to_file(const char *filepath, unsigned int *mkl_array_f, unsigned int datasize)
{
	FILE *fp;

	fp = fopen(filepath, "w+");

//	datasize = 1;
	for(unsigned int i = 0; i < datasize; i ++)
	{
		fprintf(fp, "%i\n", mkl_array_f[i]);
	}

	fclose(fp);
}

void mkl_array_B_to_file(const char *filepath, bool *mkl_array_b, unsigned int datasize)
{
	FILE *fp;

	fp = fopen(filepath, "w+");

	for(unsigned int i = 0; i < datasize; i ++)
	{
		fprintf(fp, "%u\n", (mkl_array_b[i]) ? 1 : 0);
	}

	fclose(fp);
}

void _2D_to_1D_compared_MKL(complex8 *sol, complex8 *sol_ext, complex8 *mkl_residual)
{
	unsigned int _1D_index = 0;
	unsigned int _2D_index = 0;

	complex8 current_2D = sol_ext[_2D_index];
	complex8 arg_old = sol[_1D_index];
	complex8 Input_Field;

	for(unsigned int i = 0; i < N; i++)
	{
		for(unsigned int j = 0; j < N; j++)
		{
			current_2D = sol_ext[_2D_index];
			arg_old = sol[_1D_index];

			current_2D.x /= (2 * N - 1);
			current_2D.y /= (2 * N - 1);

		//	current_2D.x /= (float)(2.f * N - 1.f);
		//	current_2D.y /= (float)(2.f * N - 1.f);
			Input_Field.y = - WAVE_NUMBER * (i * cos(ALPHA) + j * sin(ALPHA));

			Input_Field = my_cexpf(Input_Field);
			//float sigma = 400.f;
			//Input_Field.x = Input_Field.x * exp(-pow((float)((float)(j) - 512.f), 2.f)/pow(sigma, 2.f))/(sigma * sqrt(7.28f));
			//Input_Field.y = Input_Field.y * exp(-pow((float)((float)(j) - 512.f), 2.f)/pow(sigma, 2.f))/(sigma * sqrt(7.28f));
			//Input_Field.x = Input_Field.x * (exp(-pow((float)((float)(j) - 341.f), 2.f)/pow(sigma, 2.f)) + exp(-pow((float)((float)(j) - 682.f), 2.f)/pow(sigma, 2.f)))/(sigma * sqrt(7.28f));
			//Input_Field.y = Input_Field.y * (exp(-pow((float)((float)(j) - 341.f), 2.f)/pow(sigma, 2.f)) + exp(-pow((float)((float)(j) - 682.f), 2.f)/pow(sigma, 2.f)))/(sigma * sqrt(7.28f));
			
			current_2D.x += Input_Field.x - arg_old.x;
			current_2D.y += Input_Field.y - arg_old.y;
			mkl_residual[_1D_index] = current_2D;
			_1D_index ++;
			_2D_index ++;
		}

		_2D_index += N - 1;
	}
}

void _2D_to_1D_MKL(complex8 *sol, complex8 *sol_ext, complex8 *_1D_out)
{
	unsigned int _1D_index = 0;
	unsigned int _2D_index = 0;

	complex8 current_2D;
	complex8 arg_old;

	for(unsigned int i = 0; i < N; i++)
	{
		for(unsigned int j = 0; j < N; j++)
		{
			current_2D = sol_ext[_2D_index];
			arg_old = sol[_1D_index];

			arg_old.x -= current_2D.x / (2 * N  - 1);
			arg_old.y -= current_2D.y / (2 * N  - 1);
			_1D_out[_1D_index] = arg_old;

			_1D_index ++;
			_2D_index ++;
		}

		_2D_index += N - 1;
	}	
}

void create_Givens_rotation_matrix_MKL(complex8 *mkl_Givens_rotation, complex8 *Htemp, unsigned int characteristic_size)
{
	unsigned int index = 0;

	//fprintf(stderr, "givens\n");

	for(unsigned int i = 0; i < characteristic_size; i ++)
	{

		for(unsigned int j = 0; j < characteristic_size; j ++)
		{
			if ((i < characteristic_size - 2) && (i == j))
			{
				mkl_Givens_rotation[index].x = 1.f;
				mkl_Givens_rotation[index].y = 0.f;
			}
			else
			{
				if ((i == characteristic_size - 2) && (j == characteristic_size - 2))
				{	
					unsigned int ind1 = index - i;
					unsigned int ind2 = index + 1;

					//fprintf(stderr, "1:\t%i\t%i\t%i\n", index, ind1, ind2);
					float denominator = sqrt(pow((float)Htemp[ind1].x, 2.f) + pow((float)Htemp[ind2].x, 2.f) + pow((float)Htemp[ind1].y, 2.f) + pow((float)Htemp[ind2].y, 2.f));
					mkl_Givens_rotation[index].x = Htemp[ind1].x / denominator;
					mkl_Givens_rotation[index].y = Htemp[ind1].y / denominator;
				}
				else
				{	
					if ((i == characteristic_size - 2) && (j == characteristic_size - 1))
					{
						unsigned int ind2 = index - j;

						//fprintf(stderr, "2:\t%i\t%i\t%i\n", index, index, ind2);
						float denominator = sqrt(pow((float)Htemp[index].x, 2.f) + pow((float)Htemp[ind2].x, 2.f) + pow((float)Htemp[index].y, 2.f) + pow((float)Htemp[ind2].y, 2.f));
						mkl_Givens_rotation[index].x = Htemp[index].x / denominator;
						mkl_Givens_rotation[index].y = Htemp[index].y / denominator;
					}
					else
					{
						if ((i == characteristic_size - 1) && (j == characteristic_size - 2))
						{
							unsigned int ind1 = index - i;
							unsigned int ind2 = ind1  - i;


							//fprintf(stderr, "3:\t%i\t%i\t%i\n", index, ind1, ind2);
							float denominator = sqrt(pow((float)Htemp[ind1].x, 2.f) + pow((float)Htemp[ind2].x, 2.f) + pow((float)Htemp[ind1].y, 2.f) + pow((float)Htemp[ind2].y, 2.f));
							mkl_Givens_rotation[index].x = - Htemp[ind1].x / denominator;
							mkl_Givens_rotation[index].y =   Htemp[ind1].y / denominator;
						}
						else
						{
							if ((i == characteristic_size - 1) && (j == characteristic_size - 1))
							{

								unsigned int ind2 = index - i - 1;
								unsigned int ind1 = ind2  - i;


								//fprintf(stderr, "4:\t%i\t%i\t%i\n", index, ind1, ind2);
								float denominator = sqrt(pow((float)Htemp[ind1].x, 2.f) + pow((float)Htemp[ind2].x, 2.f) + pow((float)Htemp[ind1].y, 2.f) + pow((float)Htemp[ind2].y, 2.f));
								mkl_Givens_rotation[index].x = Htemp[ind1].x / denominator;
								mkl_Givens_rotation[index].y = Htemp[ind1].y / denominator;	
							}
							else
							{
								mkl_Givens_rotation[index].x = 0.f;
								mkl_Givens_rotation[index].y = 0.f;
							}
						}
					}
				}
			}

			index ++;
		}
	}
}

void resize_MKL(complex8 *mkl_data, unsigned int current_size_i, unsigned int current_size_j, unsigned int new_size_i, unsigned int new_size_j, complex8 *mkl_resized_data)
{
	complex8 zero_complex;
	zero_complex.x = 0.f;
	zero_complex.y = 0.f;

	for(unsigned int i = 0; i < new_size_i; i ++)
	{
		for(unsigned int j = 0; j < new_size_j; j ++)
		{
			mkl_resized_data[new_size_j * i + j] = ((i < current_size_i) && (j < current_size_j)) ? mkl_data[current_size_j * i + j] : zero_complex;
		}
	}
}

void Jtotal_resize_MKL(complex8 *data, unsigned int current_size_ij, complex8 *mkl_resized_data)
{

	unsigned int index_new = 0;
	unsigned int index_cur = 0;
	unsigned int new_sizeij = current_size_ij + 1;

	for(unsigned int i = 0; i < new_sizeij; i ++)
	{
		for(unsigned int j = 0; j < new_sizeij; j ++)
		{
			if ((i < current_size_ij) && (j < current_size_ij))
			{
				mkl_resized_data[index_new] = data[index_cur];

				index_cur++;
			}
			else
			{
				if ((i == new_sizeij - 1) && (i == j))
				{
					mkl_resized_data[index_new].x = 1.f;
					mkl_resized_data[index_new].y = 0.f;
				}
				else
				{
					mkl_resized_data[index_new].x = 0.f;
					mkl_resized_data[index_new].y = 0.f;
				}
			}

			index_new ++;
		}
	}
}

void get_cc_MKL(complex8 *mkl_cc, complex8 *mkl_Jtotal, float *old_norm_res_vec, unsigned int size)
{	
	unsigned int index = 0;
	for (unsigned int i = 0; i < size; i ++)
	{
		mkl_cc[i].x = mkl_Jtotal[index].x * (*old_norm_res_vec);
		mkl_cc[i].y = mkl_Jtotal[index].y * (*old_norm_res_vec);
		index += size + 1;
	}
}

void get_solution_MKL(complex8 *mkl_solution, complex8 *mkl_cc, complex8 *mkl_orthogonal_basis)
{
	complex8 current;

	for(unsigned int index = 0; index < N * N; index ++)
	{
		current = mkl_orthogonal_basis[index];
		mkl_solution[index].x += current.x * mkl_cc->x - current.y * mkl_cc->y;
		mkl_solution[index].y += current.x * mkl_cc->y + current.y * mkl_cc->x;
	}
}

void add_MKL(complex8 *mkl_solution, complex8 *mkl_add_x, complex8 *mul)
{
	for(unsigned int index = 0; index < N * N; index ++)
	{ 
		mkl_solution[index].x += mul->x * mkl_add_x[index].x - mul->y * mkl_add_x[index].y;
		mkl_solution[index].y += mul->y * mkl_add_x[index].x + mul->x * mkl_add_x[index].y;
	}
}

void save_test_MKL_C(const char *describtion, complex8 *mkl_array, unsigned int iteration_number, unsigned int size_array)
{
	char buffer[1024];

	sprintf(buffer, "%s_%i.txt", describtion, iteration_number);//test_data
	mkl_array_C_to_file(buffer, mkl_array, size_array);
}

void saveCPUrealtxt_time_t(const time_t *h_inx, const char *filename, const int M)
{
	std::ofstream outfile;
	outfile.open(filename);	
	for (int i = 0; i < M; i++) outfile << h_inx[i] << "\n";
	outfile.close();
}

void saveCPUrealtxt_timespec(const timespec *h_inx, const char *filename, const int M)
{
	std::ofstream outfile;
	outfile.open(filename);	
	for (int i = 0; i < M; i++) outfile << h_inx[i].tv_sec * BILLION + h_inx[i].tv_nsec << "\n";
	outfile.close();
}


void save_test_MKL_timespec(const char *describtion, timespec *mkl_array, unsigned int iteration_number, unsigned int size_array)
{
	char buffer[1024];

	sprintf(buffer, "%s_%i.txt", describtion, iteration_number);//test_data
	saveCPUrealtxt_timespec(mkl_array, buffer, size_array);
}


void save_test_MKL_time_t(const char *describtion, time_t *mkl_array, unsigned int iteration_number, unsigned int size_array)
{
	char buffer[1024];

	sprintf(buffer, "%s_%i.txt", describtion, iteration_number);//test_data
	saveCPUrealtxt_time_t(mkl_array, buffer, size_array);
}

void save_test_MKL_F(const char *describtion, float *mkl_array, unsigned int iteration_number, unsigned int size_array)
{
	char buffer[1024];

	sprintf(buffer, "%s_%i.txt", describtion, iteration_number);//test_data
	mkl_array_F_to_file(buffer, mkl_array, size_array);
}


void get_last_maxiter(const char *file_path, unsigned int *maxiter)
{
	std::string line;
	std::ifstream myfile (file_path);
	if (myfile.is_open())
	{
		while ( getline (myfile,line) )
		{
			std::istringstream in_string_stream(line);

			in_string_stream >> *maxiter;
		}
		myfile.close();
	}
	else fprintf(stderr, "Unable to open %s file", file_path);
}



void set_Identity_matrix_kernel(complex8 *mkl_Identity_matrix, unsigned int maxiter_plus_1)
{
	complex8 current;

	for (unsigned int i = 0; i < maxiter_plus_1; i ++)
	{
		for (unsigned int j = 0; j < maxiter_plus_1; j ++)
		{
			current.x = i == j ? 1.f : 0.f;
			current.y = 0.f;

			mkl_Identity_matrix[i *maxiter_plus_1 + j] = current;
		}
	}
}



void set_first_Jtotal_kernel(	complex8 *mkl_Jtotal,
				complex8 *Htemp,
				const unsigned int maxiter,
				const unsigned int maxiter_plus_1)
{
	for (unsigned int index = 0; index < maxiter_plus_1 * maxiter_plus_1 * 2; index ++)
	{
		switch(index)
		{	
			case 0 :
			{
				float denominator = sqrt(pow((float)Htemp->x, 2.f) + pow((float)Htemp[maxiter].x, 2.f) + pow((float)Htemp->y, 2.f) + pow((float)Htemp[maxiter].y, 2.f));
				mkl_Jtotal[0].x = Htemp->x / denominator;
				break;
			}	
			case 1 :
			{
				float denominator = sqrt(pow((float)Htemp->x, 2.f) + pow((float)Htemp[maxiter].x, 2.f) + pow((float)Htemp->y, 2.f) + pow((float)Htemp[maxiter].y, 2.f));
				mkl_Jtotal[0].y = Htemp->y / denominator;
				break;
			}
			case 2 :
			{
				float denominator = sqrt(pow((float)Htemp->x, 2.f) + pow((float)Htemp[maxiter].x, 2.f) + pow((float)Htemp->y, 2.f) + pow((float)Htemp[maxiter].y, 2.f));
				mkl_Jtotal[1].x = Htemp[maxiter].x / denominator;
				break;
			}
			case 3 :
			{
				float denominator = sqrt(pow((float)Htemp->x, 2.f) + pow((float)Htemp[maxiter].x, 2.f) + pow((float)Htemp->y, 2.f) + pow((float)Htemp[maxiter].y, 2.f));;
				mkl_Jtotal[1].y = Htemp[maxiter].y / denominator;
				break;
			}
			default:
			{
				switch(index - (maxiter_plus_1 << 1))
				{
					case 0:
					{
						float denominator = sqrt(pow((float)Htemp->x, 2.f) + pow((float)Htemp[maxiter].x, 2.f) + pow((float)Htemp->y, 2.f) + pow((float)Htemp[maxiter].y, 2.f));
						mkl_Jtotal[maxiter_plus_1].x = - Htemp[maxiter].x / denominator;
						break;
					}
					case 1:
					{
						float denominator = sqrt(pow((float)Htemp->x, 2.f) + pow((float)Htemp[maxiter].x, 2.f) + pow((float)Htemp->y, 2.f) + pow((float)Htemp[maxiter].y, 2.f));
						mkl_Jtotal[maxiter_plus_1].y =   Htemp[maxiter].y / denominator;
						break;
					}
					case 2:
					{
						float denominator = sqrt(pow((float)Htemp->x, 2.f) + pow((float)Htemp[maxiter].x, 2.f) + pow((float)Htemp->y, 2.f) + pow((float)Htemp[maxiter].y, 2.f));
						mkl_Jtotal[maxiter_plus_1 + 1].x = Htemp->x / denominator;
						break;
					}
					case 3:
					{
						float denominator = sqrt(pow((float)Htemp->x, 2.f) + pow((float)Htemp[maxiter].x, 2.f) + pow((float)Htemp->y, 2.f) + pow((float)Htemp[maxiter].y, 2.f));
						mkl_Jtotal[maxiter_plus_1 + 1].y = Htemp->y / denominator;
						break;
					}
					default:
					{
						if (index % 2)
						{
							unsigned int index2 = index / 2;
							mkl_Jtotal[index2].x = index2 % (maxiter_plus_1 + 1) ? 0.f : 1.f;
						}else
						{
							mkl_Jtotal[index / 2].y = 0.f;
						}
					}
				}
			}		
		}
	}
}




void set_4_Givens_rotation_matrix_elements_kernel(	complex8 *mkl_Htemp,
							const unsigned int characteristic_size,
							complex8 *mkl_Givens_rotation_0,
							complex8 *mkl_Givens_rotation_1,
							complex8 *mkl_Givens_rotation_2,
							complex8 *mkl_Givens_rotation_3,
							const unsigned int GMRES_i_plus_1)
{

	const unsigned int index_H_0 = GMRES_i_plus_1 * GMRES_i_plus_1 - 1;
	const unsigned int index_H_1 = index_H_0 + GMRES_i_plus_1;

	float denominator = sqrt(pow((float)mkl_Htemp[index_H_0].x, 2.f) + pow((float)mkl_Htemp[index_H_1].x, 2.f) + pow((float)mkl_Htemp[index_H_0].y, 2.f) + pow((float)mkl_Htemp[index_H_1].y, 2.f));

	for (unsigned int index = 0; index < 8; index ++)
	{
		switch(index)
		{
			case 0:
			{
				mkl_Givens_rotation_0 -> x = mkl_Htemp[index_H_0].x / denominator;
				break;
			}
			case 1:
			{
				mkl_Givens_rotation_0 -> y = mkl_Htemp[index_H_0].y / denominator;
				break;
			}
			case 2:
			{
				mkl_Givens_rotation_1 -> x = mkl_Htemp[index_H_1].x / denominator;
				break;
			}
			case 3:
			{
				mkl_Givens_rotation_1 -> y = mkl_Htemp[index_H_1].y / denominator;
				break;
			}
			case 4:
			{
				mkl_Givens_rotation_2 -> x = - mkl_Htemp[index_H_1].x / denominator;
				break;
			}
			case 5:
			{
				mkl_Givens_rotation_2 -> y =   mkl_Htemp[index_H_1].y / denominator;
				break;
			}
			case 6:
			{
				mkl_Givens_rotation_3 -> x = mkl_Htemp[index_H_0].x / denominator;
				break;
			}
			case 7:
			{
				mkl_Givens_rotation_3 -> y = mkl_Htemp[index_H_0].y / denominator;
			}
		}
	}
}
