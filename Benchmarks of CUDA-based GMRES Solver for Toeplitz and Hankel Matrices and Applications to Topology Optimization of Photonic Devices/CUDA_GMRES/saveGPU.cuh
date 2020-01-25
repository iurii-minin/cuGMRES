#include <string>
#include <fstream>
#include <sstream>

#define BILLION 1000000000LL

void saveGPUrealtxt_C(const cuComplex * d_in, const char *filename, const int M) {

	cuComplex *h_in = (cuComplex *)malloc(M * sizeof(cuComplex));

	cudacall(cudaMemcpy(h_in, d_in, M * sizeof(cuComplex), cudaMemcpyDeviceToHost));
	std::ofstream outfile;
	outfile.open(filename);	
	for (int i = 0; i < M; i++) outfile << std::setprecision(PRECISION_TO_SAVE_DATA_TO_FILE) << h_in[i].x << " " << h_in[i].y << "\n";
	outfile.close();

	free((cuComplex *) h_in);
}

void saveCPUrealtxt_C(const cuComplex * h_in, const char *filename, const int M) {
	std::ofstream outfile;
	outfile.open(filename);	
	for (int i = 0; i < M; i++) outfile << std::setprecision(PRECISION_TO_SAVE_DATA_TO_FILE) << h_in[i].x << " " << h_in[i].y << "\n";
	outfile.close();
}


void saveGPUrealtxt_F(const float *d_inx, const char *filename, const int M) {

	float *h_inx = (float *)malloc(M * sizeof(float));

	cudacall(cudaMemcpy(h_inx, d_inx, M * sizeof(float), cudaMemcpyDeviceToHost));
	std::ofstream outfile;
	outfile.open(filename);	
	for (int i = 0; i < M; i++) outfile << std::setprecision(PRECISION_TO_SAVE_DATA_TO_FILE) << h_inx[i] << "\n";
	outfile.close();

	free((float *) h_inx);
}

void saveGPUrealtxt_I(const unsigned int *d_inx, const char *filename, const int M) {

	unsigned int *h_inx = (unsigned int *)malloc(M * sizeof(unsigned int));

	cudacall(cudaMemcpy(h_inx, d_inx, M * sizeof(unsigned int), cudaMemcpyDeviceToHost));
	std::ofstream outfile;
	outfile.open(filename);	
	for (int i = 0; i < M; i++) outfile << std::setprecision(PRECISION_TO_SAVE_DATA_TO_FILE) << h_inx[i] << "\n";
	outfile.close();

	free((unsigned int *) h_inx);
}


void saveGPUrealtxt_B(const bool *d_inx, const char *filename, const int M) 
{
	bool *h_inx = (bool *)malloc(M * sizeof(bool));

	cudacall(cudaMemcpy(h_inx, d_inx, M * sizeof(bool), cudaMemcpyDeviceToHost));
	std::ofstream outfile;
	outfile.open(filename);	
	for (int i = 0; i < M; i++) outfile << std::setprecision(PRECISION_TO_SAVE_DATA_TO_FILE) << (h_inx[i] ? 1 : 0) << "\n";
	outfile.close();

	free((unsigned int *) h_inx);
}

void saveCPUrealtxt_F(const float *h_inx, const char *filename, const int M)
{
	std::ofstream outfile;
	outfile.open(filename);	
	for (int i = 0; i < M; i++) outfile << std::setprecision(PRECISION_TO_SAVE_DATA_TO_FILE) << h_inx[i] << "\n";
	outfile.close();
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



void saveCPUrealtxt_B(const bool *h_inx, const char *filename, const int M)
{
	std::ofstream outfile;
	outfile.open(filename);	
	for (int i = 0; i < M; i++) outfile << std::setprecision(PRECISION_TO_SAVE_DATA_TO_FILE) << (h_inx[i] ? 1 : 0) << "\n";
	outfile.close();
}

void saveCPUrealtxt_I(const unsigned int *h_inx, const char *filename, const int M)
{
	std::ofstream outfile;
	outfile.open(filename);	
	for (int i = 0; i < M; i++) outfile << std::setprecision(PRECISION_TO_SAVE_DATA_TO_FILE) << h_inx[i] << "\n";
	outfile.close();
}

void saveGPUrealtxt(bool *h_mask, cuComplex *dev_solution, float *dev_intensity_global, unsigned int optimization_number, const unsigned int N)
{
	char buffer[1024];

	sprintf(buffer, "data/greedy/mask_%i.txt", optimization_number);
	saveCPUrealtxt_B(h_mask, buffer, N * N);

	sprintf(buffer, "data/greedy/field_%i.txt", optimization_number);
	saveGPUrealtxt_C((cuComplex *)dev_solution, buffer, N * N);

	sprintf(buffer, "data/greedy/intensity_maximum_%i.txt", optimization_number);
	saveGPUrealtxt_F(dev_intensity_global, buffer, 1);
}

void save_test_GPU(char *describtion, cuComplex *dev_array, unsigned int iteration_number, unsigned int size_array)
{
	char buffer[1024];

	sprintf(buffer, "output/%s_%i.txt", describtion, iteration_number);//CUDA_GMRES_test
	saveGPUrealtxt_C(dev_array, buffer, size_array);
}


void save_test_F_GPU(char *describtion, float *dev_array, unsigned int iteration_number, unsigned int size_array)
{
	char buffer[1024];

	sprintf(buffer, "output/%s_%i.txt", describtion, iteration_number);//CUDA_GMRES_test
	saveGPUrealtxt_F(dev_array, buffer, size_array);
}

void save_test_F_CPU(char *describtion, float *dev_array, unsigned int iteration_number, unsigned int size_array)
{
	char buffer[1024];

	sprintf(buffer, "output/%s_%i.txt", describtion, iteration_number);//CUDA_GMRES_test
	saveCPUrealtxt_F(dev_array, buffer, size_array);
}

void save_test_time_t_CPU(char *describtion, time_t *h_array, unsigned int iteration_number, unsigned int size_array)
{
	char buffer[1024];

	sprintf(buffer, "output/%s_%i.txt", describtion, iteration_number);//CUDA_GMRES_test
	saveCPUrealtxt_time_t(h_array, buffer, size_array);
}

void save_test_timespec_CPU(char *describtion, timespec *h_array, unsigned int iteration_number, unsigned int size_array)
{
	char buffer[1024];

	sprintf(buffer, "output/%s_%i.txt", describtion, iteration_number);//CUDA_GMRES_test
	saveCPUrealtxt_timespec(h_array, buffer, size_array);
}

/*
void saveGPUrealtxt_discrete_gradient(bool *dev_mask, cuComplex *dev_solution, float h_intensity_max, unsigned int optimization_number, const unsigned int N)
{
	char buffer[1024];	

	sprintf(buffer, "data/discrete_gradient/mask_%i.txt", optimization_number);
	saveGPUrealtxt_B(dev_mask, buffer, N * N);

	sprintf(buffer, "data/discrete_gradient/field_%i.txt", optimization_number);
	saveGPUrealtxt_C((cuComplex *)dev_solution, buffer, N * N);

	sprintf(buffer, "data/discrete_gradient/intensity_maximum_%i.txt", optimization_number);
	saveCPUrealtxt_F(&h_intensity_max, buffer, 1);
}
*/

void get_array_C_to_CPU(cuComplex *h_array, const char *filename)
{
	std::string line;
	std::ifstream analytical_solution_file (filename);
	if (analytical_solution_file.is_open())
	{
		unsigned int index = 0;
		while ( getline (analytical_solution_file, line) )
		{
			std::istringstream in_string_stream(line);

			in_string_stream >> h_array[index].x >> h_array[index].y;

			index ++;
		}
		analytical_solution_file.close();
	}
	else fprintf(stderr, "Unable to open %s file", filename);
}

void print_matrix_C(const char *desc, int m, int n, cuComplex *dev_mat, int lda ) {
        int i, j;
	cuComplex *h_mat = (cuComplex *)malloc(lda * m * sizeof(cuComplex));

	cudacall(cudaMemcpy(h_mat, dev_mat, lda * m * sizeof(cuComplex), cudaMemcpyDeviceToHost));

        printf( "\nRe for %s\n", desc );
        for( i = 0; i < m; i++ ) {
                for( j = 0; j < n; j++ ) 
			if (h_mat[i * lda + j].x == 0){
				printf( "\t." );
			}else{
				printf( "\t%2.6f", h_mat[i * lda + j].x );
			}
                printf( "\n" );
        }

        printf( "\nIm for %s\n", desc );

        for( i = 0; i < m; i++ ) {
                for( j = 0; j < n; j++ ) 
			if (h_mat[i * lda + j].y == 0){
				printf( "\t." );
			}else{
				printf( "\t%2.6f", h_mat[i * lda + j].y );
			}
                printf( "\n" );
        }

	free((cuComplex *)h_mat);
}

void print_matrix_F(const char *desc, int m, int n, float *dev_mat, int lda ) {
        int i, j;
	float *h_mat = (float *)malloc(lda * m * sizeof(float));

	cudacall(cudaMemcpy(h_mat, dev_mat, lda * m * sizeof(float), cudaMemcpyDeviceToHost));

        printf( "\nFor %s\n", desc );
        for( i = 0; i < m; i++ ) {
                for( j = 0; j < n; j++ ) 
			if (h_mat[i * lda + j] == 0){
				printf( "\t." );
			}else{
				printf( "\t%6.2f", h_mat[i * lda + j] );
			}
                printf( "\n" );
        }

	free((float *)h_mat);
}
