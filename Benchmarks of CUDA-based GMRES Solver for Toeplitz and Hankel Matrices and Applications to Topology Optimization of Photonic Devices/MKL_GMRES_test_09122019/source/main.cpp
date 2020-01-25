#include "parameters.h"
#include "loop_functions.h"
#include "mkl_based_functions.h"
#include "MKL_GMRES.h"
#include <time.h>

unsigned int *get_n_timestamps_array_improved(unsigned int max_maxiter);

int main() {
    complex8 *mkl_gamma_array;
    bool *mkl_mask = (bool *)malloc(N * N * sizeof(bool));
    complex8 *mkl_solution = (complex8 *)malloc(N * N * sizeof(complex8));
    float *mkl_GMRES_residuals;
    float time_consumption;
    unsigned int GMRES_n = 0;
    unsigned int min_maxiter = 44;
    unsigned int max_maxiter = 45;
    unsigned int min_repetition = 0;
    unsigned int max_repetition = 1;
    time_t clock_time = 0;

    bool mkl_res_vs_tol_p = false;


    unsigned int *n_timestamps_array = get_n_timestamps_array_improved((unsigned int)max_maxiter);

    fprintf(stderr, "\n\n\nin MAIN\n\n");

    const float tolerance = 0.00000001f;

    get_mask_on_path("source/cylinder_256.txt", (bool *)mkl_mask);

    get_gamma_array((complex8 **)&mkl_gamma_array);

    for(unsigned int maxiter = min_maxiter; maxiter < max_maxiter; maxiter ++) // upper bound was equal to 100
    {
        for(unsigned int repetition_i = min_repetition; repetition_i < max_repetition; repetition_i ++) // upper bound was equal to 100
        {

            timespec *h_computation_times = (timespec *) malloc(n_timestamps_array[maxiter] * sizeof(timespec));

            init_x0_MKL((complex8 *)mkl_solution);
            memset(h_computation_times, 0, n_timestamps_array[maxiter] * sizeof(timespec));

            clock_time = clock();

            fprintf(stderr, "\n\nFFT_GMRES_with_MKL\n");
            FFT_GMRES_with_MKL(    (const complex8 *) mkl_gamma_array,
                        (const bool     *) mkl_mask,
                        (complex8       *) mkl_solution,
                        (float         **)&mkl_GMRES_residuals,
                        (unsigned int   *)&GMRES_n,
                        (const float     ) tolerance,
                        (bool           *)&mkl_res_vs_tol_p,
                        (unsigned int    ) maxiter,
                        (timespec     *) h_computation_times);

            time_consumption = (float)(clock() - clock_time) / (float)(CLOCKS_PER_SEC);

            fprintf(stderr, "maxiter = %i, repetition_i = %i, time_consumption = %f\n", maxiter, repetition_i, time_consumption);

            fprintf(stderr, "File writing: time = %f\n", time_consumption);
            save_test_MKL_F((const char *)"_results/test_residuals/time_256", (float *)&time_consumption, (unsigned int)maxiter * 100 + repetition_i, (unsigned int)1);
            
            fprintf(stderr, "File writing: residual = %f\n", mkl_GMRES_residuals[maxiter]);
            save_test_MKL_F((const char *)"_results/test_residuals/residual_256", (float *)mkl_GMRES_residuals + maxiter, (unsigned int)maxiter * 100 + repetition_i, (unsigned int)1);
            free((float *)mkl_GMRES_residuals);
            mkl_GMRES_residuals = NULL;

            fprintf(stderr, "File writing: solution\n");
            save_test_MKL_C((const char *)"_results/test_residuals/solution_256", (complex8 *)mkl_solution, (unsigned int)maxiter * 100 + repetition_i, (unsigned int)N * N);
            save_test_MKL_timespec((const char *)"_results/test_residuals/times_256", (timespec *)h_computation_times, maxiter * 100 + repetition_i, n_timestamps_array[maxiter]);


            free((time_t *)h_computation_times);
        }
    }
    mkl_array_C_to_file((const char *)"_results/mkl_solution_matrix.txt", mkl_solution, N * N);

    printf("\n\nSuccessful exit from MKL\n\n");    
    free((bool *)mkl_mask);
    free((complex8 *)mkl_solution);
    free((complex8 *)mkl_gamma_array);

    return 0;
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

    for (unsigned int maxiter; maxiter < max_maxiter; maxiter ++)
    {
        n_timestamps_array[maxiter] = get_n_timestamps_val_improved((unsigned int)maxiter);
    }
    return n_timestamps_array;
}
