#include <cufft.h>
#include <cublas_v2.h>
#include <fstream>
#include <iomanip>
#include <curand_kernel.h>

//#pragma once
//#ifdef __INTELLISENSE__
//void __syncthreads();
//#endif

#define WAVE_NUMBER 2*3.14f/(N/6.f)
#define Q 32
#define THREADS_PER_BLOCK N / Q
#define THREADS_PER_BLOCK_M THREADS_PER_BLOCK * 2
#define E0 1
#define ALPHA 3.14*0/180
#define EPSILON 2.25f
#define CHI (EPSILON-1)*WAVE_NUMBER*WAVE_NUMBER
#define PRECISION_TO_SAVE_DATA_TO_FILE 9
#define cudacall(call)                                                                                                          \
    do                                                                                                                          \
    {                                                                                                                           \
        cudaError_t err = (call);                                                                                               \
        if(cudaSuccess != err)                                                                                                  \
        {                                                                                                                       \
            fprintf(stderr,"CUDA Error:\nFile = %s\nLine = %d\nReason = %s\n", __FILE__, __LINE__, cudaGetErrorString(err));    \
            cudaDeviceReset();                                                                                                  \
            exit(EXIT_FAILURE);                                                                                                 \
        }                                                                                                                       \
    }                                                                                                                           \
    while (0)

#define cudacheckSYN()                                                                                                          \
    do                                                                                                                          \
    {                                                                                                                           \
        cudaError_t err = cudaGetLastError();                                                                                   \
        if(cudaSuccess != err)                                                                                                  \
        {                                                                                                                       \
            fprintf(stderr,"GetL Error:\nFile = %s\nLine = %d\nReason = %s\n", __FILE__, __LINE__, cudaGetErrorString(err));    \
            cudaDeviceReset();                                                                                                  \
            exit(EXIT_FAILURE);                                                                                                 \
        }                                                                                                                       \
        err = cudaDeviceSynchronize();                                                                                          \
        if(cudaSuccess != err)                                                                                                  \
        {                                                                                                                       \
            fprintf(stderr,"DevSyn ERR:\nFile = %s\nLine = %d\nReason = %s\n", __FILE__, __LINE__, cudaGetErrorString(err));    \
            cudaDeviceReset();                                                                                                  \
            exit(EXIT_FAILURE);                                                                                                 \
        }                                                                                                                       \
    }                                                                                                                           \
    while (0)

#define cublascall(call)                                                                                        \
    do                                                                                                          \
    {                                                                                                           \
        cublasStatus_t status = (call);                                                                         \
        if(CUBLAS_STATUS_SUCCESS != status)                                                                     \
        {                                                                                                       \
            fprintf(stderr,"CUBLAS Error:\nFile = %s\nLine = %d\nCode = %d\n", __FILE__, __LINE__, status);     \
            cudaDeviceReset();                                                                                  \
            exit(EXIT_FAILURE);                                                                                 \
        }                                                                                                       \
                                                                                                                \
    }                                                                                                           \
    while(0)

#define cufftcall(call)                                                                                         \
    do                                                                                                          \
    {                                                                                                           \
        cufftResult_t status = (call);                                                                          \
        if(CUFFT_SUCCESS != status)                                                                             \
        {                                                                                                       \
            fprintf(stderr,"CUFFT Error:\nFile = %s\nLine = %d\nCode = %d\n", __FILE__, __LINE__, status);      \
            cudaDeviceReset();                                                                                  \
            exit(EXIT_FAILURE);                                                                                 \
        }                                                                                                       \
                                                                                                                \
    }                                                                                                           \
    while(0)

#define cusolvercall(call)                                                                                      \
    do                                                                                                          \
    {                                                                                                           \
        cusolverStatus_t status = (call);                                                                       \
        if(CUSOLVER_STATUS_SUCCESS != status)                                                                   \
        {                                                                                                       \
            fprintf(stderr,"CUSOLVER Error:\nFile = %s\nLine = %d\nCode = %d\n", __FILE__, __LINE__, status);   \
            cudaDeviceReset();                                                                                  \
            exit(EXIT_FAILURE);                                                                                 \
        }                                                                                                       \
                                                                                                                \
    }                                                                                                           \
    while(0)



__global__ void extend_by_zeros_kernel(bool *dev_mask, cuComplex *dev_usual, cuComplex *dev_extended, const unsigned int N) //For usual matvec (mask is present)
{
	unsigned int i = Q * blockIdx.x + threadIdx.x;
	unsigned int j = Q * blockIdx.y + threadIdx.y;
	unsigned int size_limit = (N << 1) - 1;
	cuComplex current;

	if ((i <  size_limit) && (j < size_limit ))
	{	
		unsigned int Ni = N * i;
		unsigned int index = Ni + j;
		unsigned int index_extended = index + Ni - i;
		if ((i < N) && (j < N) && (dev_mask[index]))
		{
			current.x = CHI * dev_usual[index].x;
			current.y = CHI * dev_usual[index].y;
		}
		else
		{
			current.x = current.y = 0.f;
		}
		dev_extended[index_extended] = current;
	}
}


__global__ void extend_by_zeros_kernel(	cuComplex *dev_usual,  //For Gradient matvec (mask is absent)
					cuComplex *dev_extended,
					const unsigned int N)
{
	unsigned int i = Q * blockIdx.x + threadIdx.x;
	unsigned int j = Q * blockIdx.y + threadIdx.y;
	unsigned int size_limit = (N << 1) - 1;
	cuComplex current;

	if ((i <  size_limit) && (j < size_limit ))
	{	
		unsigned int Ni = N * i;
		unsigned int index = Ni + j;
		unsigned int index_extended = index + Ni - i;
		if ((i < N) && (j < N))
		{
			current.x = CHI * dev_usual[index].x;
			current.y = CHI * dev_usual[index].y;
		}
		else
		{
			current.x = current.y = 0.f;
		}
		dev_extended[index_extended] = current;
	}
}

__global__ void MatMul_ElemWise_kernel(	cuComplex *dev_bttb_sur,
					cuComplex *dev_vec2D,
					const unsigned int N)
{
	unsigned int i = Q * blockIdx.x + threadIdx.x;
	unsigned int j = Q * blockIdx.y + threadIdx.y;
	unsigned int size_limit = (N << 1) - 1;

	if (( i < size_limit ) && ( j < size_limit ))
	{
		unsigned int index = size_limit * i + j;
		cuComplex curr_bttb = dev_bttb_sur[index];
		cuComplex curr_out_mul = dev_vec2D[index];
		dev_vec2D[index].x = (curr_bttb.x * curr_out_mul.x - curr_out_mul.y * curr_bttb.y);
		dev_vec2D[index].y = (curr_out_mul.x * curr_bttb.y + curr_out_mul.y * curr_bttb.x);
	}
}

__device__ __forceinline__ cuComplex my_cexpf(cuComplex z) //FOR ONLY z.x = 0.f;
{
	cuComplex res;
	sincosf(z.y, &res.y, &res.x);
	res.x *= E0;
	res.y *= E0;
	return res;
}


__global__ void _2D_to_1D_compared_kernel(	cuComplex *dev_input_mul,  //For GMRES to compute LES for Helmholtz equation (mask and target index are absent)
						cuComplex *dev_2D_in,
						cuComplex *dev_residual,
						const unsigned int N)
{
	unsigned int i = Q * blockIdx.x + threadIdx.x;
	unsigned int j = Q * blockIdx.y + threadIdx.y;
	unsigned int size_limit = (N << 1) - 1;

	if ((i < size_limit) && (j < size_limit))
	{
		unsigned int Ni = N * i;
		unsigned int _1D_index = Ni + j;
		unsigned int _2D_index = _1D_index + Ni - i;
		cuComplex current_2D = dev_2D_in[_2D_index];
		cuComplex arg_old = dev_input_mul[_1D_index];
		cuComplex Input_Field;

		Input_Field.y = - WAVE_NUMBER * (i * cos(ALPHA) + j * sin(ALPHA));
		Input_Field = my_cexpf(Input_Field);
		//float sigma = 400.f;
		//Input_Field.x = Input_Field.x * exp(-pow((float)((float)(j) - 512.f), 2.f)/pow(sigma, 2.f))/(sigma * sqrt(7.28f));
		//Input_Field.y = Input_Field.y * exp(-pow((float)((float)(j) - 512.f), 2.f)/pow(sigma, 2.f))/(sigma * sqrt(7.28f));
		//Input_Field.x = Input_Field.x * (exp(-pow((float)((float)(j) - 341.f), 2.f)/pow(sigma, 2.f)) + exp(-pow((float)((float)(j) - 682.f), 2.f)/pow(sigma, 2.f)))/(sigma * sqrt(7.28f));
		//Input_Field.y = Input_Field.y * (exp(-pow((float)((float)(j) - 341.f), 2.f)/pow(sigma, 2.f)) + exp(-pow((float)((float)(j) - 682.f), 2.f)/pow(sigma, 2.f)))/(sigma * sqrt(7.28f));
		Input_Field.x += current_2D.x / ((N << 1) - 1) / ((N << 1) - 1) - arg_old.x;
		Input_Field.y += current_2D.y / ((N << 1) - 1) / ((N << 1) - 1) - arg_old.y;
		dev_residual[_1D_index] = Input_Field;
	}
}



__global__ void _2D_to_1D_compared_kernel(	bool *dev_mask,  //For Gradient of Helmholtz equation-based LES GMRES (mask and target index are present) //#CHANGED
						cuComplex *dev_input_mul,
						cuComplex *dev_2D_in,
						cuComplex *dev_residual,
						const unsigned int h_index_of_max,
						const unsigned int N)
{
	unsigned int i = Q * blockIdx.x + threadIdx.x;
	unsigned int j = Q * blockIdx.y + threadIdx.y;
	unsigned int size_limit = (N << 1) - 1;

	if ((i < size_limit) && (j < size_limit))
	{
		unsigned int Ni = N * i;
		unsigned int _1D_index = Ni + j;
		unsigned int _2D_index = _1D_index + Ni - i;

		cuComplex arg_old = dev_input_mul[_1D_index];
		cuComplex current;

		current.x =  (h_index_of_max == _1D_index) ? 1.f - arg_old.x : - arg_old.x;	//"=" operation
		current.y = - arg_old.y;

		if (dev_mask[_1D_index])
		{
			cuComplex current_2D = dev_2D_in[_2D_index];
			
			current.x += current_2D.x / ((N << 1) - 1) / ((N << 1) - 1);
			current.y += current_2D.y / ((N << 1) - 1) / ((N << 1) - 1);
		}
		dev_residual[_1D_index] = current;
	}
}


__global__ void residual_normalization_kernel(	cuComplex *dev_residual_vec,
						float *dev_norm_res_vec,
						cuComplex *dev_orthogonal_basis)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	cuComplex current = dev_residual_vec[index];
	current.x = current.x / (*dev_norm_res_vec);
	current.y = current.y / (*dev_norm_res_vec);
	dev_orthogonal_basis[index] = current;
}


__global__ void set_alpha_beta_kernel(cuComplex *cu_alpha, cuComplex *cu_beta)
{
	switch(blockIdx.x)
	{
		case 0 :
		{
			cu_alpha->x = 1.f;
			break;
		}
		case 1 :
		{
			cu_alpha->y = 0.f;
			break;
		}
		case 2 :
		{
			cu_beta->x = 0.f;
			break;
		}
		case 3 :
		{
			cu_beta->y = 0.f;
			break;
		}
	}
}

__global__ void _2D_to_1D_kernel(	cuComplex *dev_input_mul,
					cuComplex *dev_2D_in,
					cuComplex *dev_1D_out,
					const unsigned int N)
{
	unsigned int i = Q * blockIdx.x + threadIdx.x;
	unsigned int j = Q * blockIdx.y + threadIdx.y;
	unsigned int Ni = N * i;
	unsigned int _1D_index = Ni + j;
	unsigned int _2D_index = _1D_index + Ni - i;
	cuComplex current = dev_input_mul[_1D_index];
	cuComplex new_arg = dev_2D_in[_2D_index];
	
	current.x -= new_arg.x / ((N << 1) - 1) / ((N << 1) - 1);
	current.y -= new_arg.y / ((N << 1) - 1) / ((N << 1) - 1);

	dev_1D_out[_1D_index] = current;
}

__global__ void _2D_to_1D_kernel(	bool *dev_mask, //For gradient computations
					cuComplex *dev_input_mul,
					cuComplex *dev_2D_in,
					cuComplex *dev_1D_out,
					const unsigned int N)
{
	unsigned int i = Q * blockIdx.x + threadIdx.x;
	unsigned int j = Q * blockIdx.y + threadIdx.y;
	unsigned int Ni = N * i;
	unsigned int _1D_index = Ni + j;
	unsigned int _2D_index = _1D_index + Ni - i;
	cuComplex current = dev_input_mul[_1D_index];

	if (dev_mask[_1D_index])
	{
		cuComplex new_arg = dev_2D_in[_2D_index];

		current.x -= new_arg.x / ((N << 1) - 1) / ((N << 1) - 1);
		current.y -= new_arg.y / ((N << 1) - 1) / ((N << 1) - 1);
	}
	dev_1D_out[_1D_index] = current;
}

__global__ void weight_subtract_kernel(	cuComplex *dev_weight,
					cuComplex *dev_Hjk,
					cuComplex *dev_vj)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	cuComplex current = dev_weight[i];
	cuComplex current_vj = dev_vj[i];
	
	current.x -= (dev_Hjk->x) * current_vj.x - (dev_Hjk->y) * current_vj.y;
	current.y -= (dev_Hjk->y) * current_vj.x + (dev_Hjk->x) * current_vj.y;
	dev_weight[i] = current;
}


__global__ void get_complex_divided(	const float *dev_Hjk_norm_part,
					cuComplex *dev_Hj,
					float *dev_divided)
{
	switch(blockIdx.x)
	{
		case 0 :
		{	
			*dev_divided = 1.f / *dev_Hjk_norm_part;
			break;
		}
		case 1 :
		{
			dev_Hj->y      = 0.f;
			break;
		}
		case 2 :
		{
			dev_Hj->x = *dev_Hjk_norm_part;
			break;
		}
	}
}

__global__ void set_first_Jtotal_kernel(	cuComplex *dev_Jtotal,
						cuComplex *Htemp,
						const unsigned int maxiter,
						const unsigned int characteristic_size)
{
	switch(blockIdx.x)
	{	
		case 0 :
		{
			float denominator = sqrt(pow((float)Htemp->x, 2.f) + pow((float)Htemp[maxiter].x, 2.f) + pow((float)Htemp->y, 2.f) + pow((float)Htemp[maxiter].y, 2.f));
			dev_Jtotal[0].x = Htemp->x / denominator;
			break;
		}	
		case 1 :
		{
			float denominator = sqrt(pow((float)Htemp->x, 2.f) + pow((float)Htemp[maxiter].x, 2.f) + pow((float)Htemp->y, 2.f) + pow((float)Htemp[maxiter].y, 2.f));
			dev_Jtotal[0].y = Htemp->y / denominator;
			break;
		}
		case 2 :
		{
			float denominator = sqrt(pow((float)Htemp->x, 2.f) + pow((float)Htemp[maxiter].x, 2.f) + pow((float)Htemp->y, 2.f) + pow((float)Htemp[maxiter].y, 2.f));
			dev_Jtotal[1].x = Htemp[maxiter].x / denominator;
			break;
		}
		case 3 :
		{
			float denominator = sqrt(pow((float)Htemp->x, 2.f) + pow((float)Htemp[maxiter].x, 2.f) + pow((float)Htemp->y, 2.f) + pow((float)Htemp[maxiter].y, 2.f));;
			dev_Jtotal[1].y = Htemp[maxiter].y / denominator;
			break;
		}
		default:
		{
			switch(blockIdx.x - (characteristic_size << 1))
			{
				case 0:
				{
					float denominator = sqrt(pow((float)Htemp->x, 2.f) + pow((float)Htemp[maxiter].x, 2.f) + pow((float)Htemp->y, 2.f) + pow((float)Htemp[maxiter].y, 2.f));
					dev_Jtotal[characteristic_size].x = - Htemp[maxiter].x / denominator;
					break;
				}
				case 1:
				{
					float denominator = sqrt(pow((float)Htemp->x, 2.f) + pow((float)Htemp[maxiter].x, 2.f) + pow((float)Htemp->y, 2.f) + pow((float)Htemp[maxiter].y, 2.f));
					dev_Jtotal[characteristic_size].y =   Htemp[maxiter].y / denominator;
					break;
				}
				case 2:
				{
					float denominator = sqrt(pow((float)Htemp->x, 2.f) + pow((float)Htemp[maxiter].x, 2.f) + pow((float)Htemp->y, 2.f) + pow((float)Htemp[maxiter].y, 2.f));
					dev_Jtotal[characteristic_size + 1].x = Htemp->x / denominator;
					break;
				}
				case 3:
				{
					float denominator = sqrt(pow((float)Htemp->x, 2.f) + pow((float)Htemp[maxiter].x, 2.f) + pow((float)Htemp->y, 2.f) + pow((float)Htemp[maxiter].y, 2.f));
					dev_Jtotal[characteristic_size + 1].y = Htemp->y / denominator;
					break;
				}
				default:
				{
					if (blockIdx.x % 2)
					{
						unsigned int index = blockIdx.x / 2;
						dev_Jtotal[index].x = index % (characteristic_size + 1) ? 0.f : 1.f;
					}else
					{
						dev_Jtotal[blockIdx.x / 2].y = 0.f;
					}
				}
			}
		}		
	}
}

__global__ void set_Identity_matrix_kernel(cuComplex *dev_Identity_matrix)
{
	unsigned int i = blockIdx.x;
	unsigned int j = blockIdx.y;
	cuComplex current;

	current.x = i == j ? 1.f : 0.f;
	current.y = 0.f;

	dev_Identity_matrix[i * gridDim.x + j] = current;
}

__global__ void set_4_Givens_rotation_matrix_elements_kernel(	cuComplex *dev_Htemp,
								const unsigned int characteristic_size,
								cuComplex *dev_Givens_rotation_0,
								cuComplex *dev_Givens_rotation_1,
								cuComplex *dev_Givens_rotation_2,
								cuComplex *dev_Givens_rotation_3,
								const unsigned int GMRES_i_plus_1)
{

	const unsigned int index_H_0 = GMRES_i_plus_1 * GMRES_i_plus_1 - 1;
	const unsigned int index_H_1 = index_H_0 + GMRES_i_plus_1;

	float denominator = sqrt(pow((float)dev_Htemp[index_H_0].x, 2.f) + pow((float)dev_Htemp[index_H_1].x, 2.f) + pow((float)dev_Htemp[index_H_0].y, 2.f) + pow((float)dev_Htemp[index_H_1].y, 2.f));

	switch(blockIdx.x)
	{
		case 0:
		{
			dev_Givens_rotation_0 -> x = dev_Htemp[index_H_0].x / denominator;
			break;
		}
		case 1:
		{
			dev_Givens_rotation_0 -> y = dev_Htemp[index_H_0].y / denominator;
			break;
		}
		case 2:
		{
			dev_Givens_rotation_1 -> x = dev_Htemp[index_H_1].x / denominator;
			break;
		}
		case 3:
		{
			dev_Givens_rotation_1 -> y = dev_Htemp[index_H_1].y / denominator;
			break;
		}
		case 4:
		{
			dev_Givens_rotation_2 -> x = - dev_Htemp[index_H_1].x / denominator;
			break;
		}
		case 5:
		{
			dev_Givens_rotation_2 -> y =   dev_Htemp[index_H_1].y / denominator;
			break;
		}
		case 6:
		{
			dev_Givens_rotation_3 -> x = dev_Htemp[index_H_0].x / denominator;
			break;
		}
		case 7:
		{
			dev_Givens_rotation_3 -> y = dev_Htemp[index_H_0].y / denominator;
		}
	}
}


__global__ void set_cc_kernel(	cuComplex *dev_cc,
				cuComplex *dev_Jtotal,
				float *dev_old_norm_res_vec,
				const unsigned int characteristic_size)
{	
	unsigned int index = blockIdx.x * characteristic_size;

	dev_cc[blockIdx.x].x = dev_Jtotal[index].x * (*dev_old_norm_res_vec);
	dev_cc[blockIdx.x].y = dev_Jtotal[index].y * (*dev_old_norm_res_vec);
}



__global__ void get_new_solution_kernel(	cuComplex *dev_cc,
						cuComplex *dev_HH)
{
	float dominant = pow((float)(dev_HH->x), 2.f) + pow((float)(dev_HH->y), 2.f);
	cuComplex current;
	current.x = (dev_cc->x * dev_HH->x + dev_cc->y * dev_HH->y) / dominant;
	current.y = (dev_cc->y * dev_HH->x - dev_cc->x * dev_HH->y) / dominant;
	(*dev_cc) = current;
}


__global__ void get_solution_kernel(	cuComplex *dev_solution,
					cuComplex *dev_cc,
					cuComplex *dev_orthogonal_basis)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	cuComplex current = dev_orthogonal_basis[index];
	atomicAdd((float *)&(dev_solution[index].x), current.x * dev_cc->x - current.y * dev_cc->y);
	atomicAdd((float *)&(dev_solution[index].y), current.x * dev_cc->y + current.y * dev_cc->x);
}


__global__ void add_kernel(	cuComplex *dev_solution,
				cuComplex *dev_add_x,
				cuComplex *dev_mul)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	atomicAdd((float *)&(dev_solution[index].x), dev_mul->x * dev_add_x[index].x - dev_mul->y * dev_add_x[index].y);
	atomicAdd((float *)&(dev_solution[index].y), dev_mul->y * dev_add_x[index].x + dev_mul->x * dev_add_x[index].y);
}


__global__ void init_x0_kernel(	cuComplex *dev_input,
				const unsigned int N)
{
	unsigned int i = Q * blockIdx.x + threadIdx.x;
	unsigned int j = Q * blockIdx.y + threadIdx.y;
	cuComplex Input_Field;

	Input_Field.y = - WAVE_NUMBER * (i * cos(ALPHA) + j * sin(ALPHA));
	Input_Field = my_cexpf(Input_Field);
	//float sigma = 400.f;
	//Input_Field.x = Input_Field.x * exp(-pow((float)((float)(j) - 512.f), 2.f)/pow(sigma, 2.f))/(sigma * sqrt(7.28f));
	//Input_Field.y = Input_Field.y * exp(-pow((float)((float)(j) - 512.f), 2.f)/pow(sigma, 2.f))/(sigma * sqrt(7.28f));
	//Input_Field.x = Input_Field.x * (exp(-pow((float)((float)(j) - 341.f), 2.f)/pow(sigma, 2.f)) + exp(-pow((float)((float)(j) - 682.f), 2.f)/pow(sigma, 2.f)))/(sigma * sqrt(7.28f));
	//Input_Field.y = Input_Field.y * (exp(-pow((float)((float)(j) - 341.f), 2.f)/pow(sigma, 2.f)) + exp(-pow((float)((float)(j) - 682.f), 2.f)/pow(sigma, 2.f)))/(sigma * sqrt(7.28f));
	dev_input[i * N + j] = Input_Field;
}
