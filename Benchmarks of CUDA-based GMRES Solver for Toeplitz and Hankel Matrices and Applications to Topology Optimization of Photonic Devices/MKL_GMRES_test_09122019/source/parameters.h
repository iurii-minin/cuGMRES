#include <stdlib.h>
#include <stdio.h>
#include <mkl_types.h>
#include <math.h>
#include <stdbool.h>
#include <string.h> /* memset */
#include <unistd.h> /* close */
#include "mkl_dfti.h"
#include <mkl_vml_functions.h>
#include <ctype.h>
#include <mkl.h>
#include <time.h>
#include <string>
#include <fstream>
#include <sstream>

/* Parameters */
#define N 256
#define WAVE_NUMBER 2*3.14f/(N/6.f)
#define E0 1
#define ALPHA 3.14*0/180
#define EPSILON 2.25f
#define CHI (EPSILON-1)*WAVE_NUMBER*WAVE_NUMBER
#define PRECISION_TO_SAVE_DATA_TO_FILE 2

#define dfticall(call)                                                                                                          \
    do                                                                                                                          \
    {                                                                                                                           \
        MKL_LONG err = (call);                                                                                                  \
        if(DFTI_NO_ERROR != err)                                                                                                \
        {                                                                                                                       \
            fprintf(stderr,"DFTI Error:\nFile = %s\nLine = %d\nReason = %s\n", __FILE__, __LINE__, DftiErrorMessage(err));      \
            exit(EXIT_FAILURE);                                                                                                 \
        }                                                                                                                       \
    }                                                                                                                           \
    while (0)

#define lapackcall(call)                                                                                                        \
    do                                                                                                                          \
    {                                                                                                                           \
        lapack_int err = (call);                                                                                                \
        if(0 != err)                                                                                                            \
        {                                                                                                                       \
            fprintf(stderr,"LAPACK Error:\nFile = %s\nLine = %d\n\n", __FILE__, __LINE__);                                      \
            exit(EXIT_FAILURE);                                                                                                 \
        }                                                                                                                       \
    }                                                                                                                           \
    while (0)

typedef struct{ float x; float y; } complex8;
