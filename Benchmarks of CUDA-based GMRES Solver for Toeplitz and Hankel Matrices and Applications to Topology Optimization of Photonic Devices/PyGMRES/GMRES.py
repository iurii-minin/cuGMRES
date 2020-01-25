import numpy as np
import copy

from scipy.special import hankel1

prefix_input = "/gpfs/gpfs0/i.minin/input/"
prefix_output = "numpy_savings/"

rep_st  = 99 
rep_end = 100


def get_plane_wave(k,size):
    y,x = np.mgrid[:size, :size]
    a = np.pi*0/180
    sigma = 400
    E = np.exp(-1j*k*(x*np.cos(a)+y*np.sin(a))) #*  np.exp(-((y - 512)/sigma)**2)/(sigma * np.sqrt(7.28));
    return(E)

def get_greenfun(r,k):
    return (1j/4)*hankel1(0, k* r)

def get_green_matrix(k,size):
    j,i = np.mgrid[:size, :size]
    ij_block = np.sqrt((i-1/2)**2+j**2)
    green_mat = get_greenfun(ij_block,k)
    return green_mat


def G_matvec(vec,k, qweq = False, g = None):
    size = int(np.sqrt(vec.shape[0]))
    if g is None:
        G_block = get_green_matrix(k,size)
        G = get_toeplitz_mat(G_block)
    else:
        G = g
    mat = np.zeros((2*size-1,2*size-1),dtype = np.complex64)
    mat_block = vec.reshape((-1, size))
    mat[:size,:size] = mat_block
    out_mat = np.fft.ifft2(np.fft.fft2(G) * np.fft.fft2(mat))
    if qweq == True:
        return G_block, G
    out = out_mat[:size,:size].reshape((-1,1))
    return out


def matvec(x,eps,k, qweq = False, transpose = False, g = None):
    x = x.reshape((-1,1))
    #print(x)
    size = x.shape[0]
    chi = k**2*eps
    if qweq == True:
        return G_matvec(x * chi, k, qweq, g)
    if transpose:
        return x - chi*G_matvec(x,k, qweq, g)
    else:
        return x - G_matvec(x*chi,k, qweq, g)
    

def get_eps_from_mask(e,mask):
    return (e-1)*mask.reshape((-1,1))    


def get_toeplitz_mat(ij_block):
    ij_block = copy.deepcopy(ij_block)
    T1 = np.hstack((ij_block,ij_block[:,:0:-1]))
    T2 = np.hstack((ij_block[:0:-1,:],ij_block[:0:-1,:0:-1]))
    T = np.vstack((T1,T2))
    return T

def get_complex_array(filename):
    
    N_big = 0
    mynumbers = []
    with open(filename) as f:
        for line in f:
            N_big += 1
            mynumbers.append([float(n) for n in line.strip().split(' ')])

    complex_array = np.zeros(N_big, dtype = np.complex64)
    i = 0
#     print("N = ", N_big)
    for pair in mynumbers:
        try:
            complex_array[i] = pair[0] + 1j * pair[1]
    #         if pair[0] < 0.4 and pair[1] < 0.1:
    #   i         print(i)
            i +=1
            # Do Something with x and y
        except IndexError:
            print("A line in the file doesn't have enough entries.")
    return complex_array


for repetition in range(rep_st, rep_end):
    print("\n")
    for power in range(8, 14):
        print("\n")
        N = 2 ** power

        k = 2*3.14/(N/6)
        e = 2.25

        cylinder_mask = np.zeros((N, N))
        x, y = np.mgrid[0:N, 0:N]
        cylinder_mask[(y- N/3)**2 + (x - N / 2)**2 <= (N/ 6)**2 ] = 1


        eps = get_eps_from_mask(e, cylinder_mask)


        x0 = get_plane_wave(k, N).reshape(N * N)
        #np.ones((N, N)) + 1j* np.ones((N, N)) #np.ones((N, N), dtype = np.complex64)
        A_x = matvec(x0, eps, k)
        r0 = x0.reshape(-1) - A_x.reshape(-1)
        normr0 = np.linalg.norm(r0)
        v = r0 / normr0

        GMRES_i = 0
        residual = 1


        tol = 1e-12

        V = v

        if (residual > tol):
            H = np.zeros((2, 1), dtype = np.complex64)
            w = matvec(v, eps, k).reshape(-1)
            H[0, 0] = np.inner(w, v.conj())
            w = w - H[0, 0] * v
            H[1, 0] = np.linalg.norm(w)
            v = w / H[1, 0]
            V = np.hstack((V.reshape(N**2, 1), v.reshape(N**2, 1)))
            Htemp = H
            J = np.zeros((2, 2), dtype = np.complex64)

            denominator = np.linalg.norm(Htemp)
            J[1, 1] = J[0, 0] = Htemp[0, 0] / denominator
            J[0, 1] =           Htemp[1, 0] / denominator
            J[1, 0] =         - Htemp[1, 0].conj() / denominator
            Jtotal = J

        #     HH = np.dot(Jtotal, H)
            bb = np.zeros((2, 1), dtype = np.complex64)
            bb[0] = normr0
            c = np.dot(Jtotal, bb)
            residual = abs(c[0, 0])
            print(residual)
            GMRES_i = 1

        residual_set = []
        rel_error_set =[]

        x_reference1 = get_complex_array(prefix_input + "analytical_solution_" + str(N) + ".txt")
        x_reference1 = x_reference1.reshape(N, N)

        x_reference = np.zeros_like(x_reference1, dtype = np.complex64)
        for i in range(N):
            for j in range(N):
                x_reference[i, j] = x_reference1[j, i]

        norm_ref = np.linalg.norm(x_reference)


        while ((residual > tol) and (GMRES_i < 50)):
            print("N = ", N, "repetition = ", repetition, "GMRES_i = ", GMRES_i, )
            H_new = np.zeros((GMRES_i + 2, GMRES_i + 1), dtype = np.complex64)
            H_new[0:GMRES_i + 1, 0:GMRES_i] = H
            H = H_new
            w = matvec(v, eps, k).reshape(-1)

            for j in range(GMRES_i + 1):
                H[j, GMRES_i] = np.inner(w, V[:, j].conj())
                w = w - H[j, GMRES_i] * V[:, j]

            H[GMRES_i + 1, GMRES_i] = np.linalg.norm(w)
            v = w / H[GMRES_i + 1, GMRES_i]
            V = np.hstack((V.reshape(N**2, GMRES_i + 1), v.reshape(N**2, 1)))

            Jtotal = np.hstack((Jtotal, np.zeros(GMRES_i+1).reshape(GMRES_i+1, 1)))
            Jtotal = np.vstack((Jtotal, np.zeros(GMRES_i+2).reshape(1, GMRES_i+2)))
            Jtotal[GMRES_i+1, GMRES_i+1] = 1

            Htemp = np.dot(Jtotal, H)
            J = np.eye(GMRES_i + 2, dtype = np.complex64)

            denominator = np.linalg.norm(np.asarray([Htemp[GMRES_i, GMRES_i], Htemp[GMRES_i + 1, GMRES_i]]))
            J[GMRES_i + 1, GMRES_i + 1] = J[GMRES_i, GMRES_i] = Htemp[GMRES_i    , GMRES_i] / denominator
            J[GMRES_i, GMRES_i + 1] =           Htemp[GMRES_i + 1, GMRES_i] / denominator
            J[GMRES_i + 1, GMRES_i] =         - Htemp[GMRES_i + 1, GMRES_i].conj() / denominator

            Jtotal = np.dot(J, Jtotal)
            bb = np.zeros((GMRES_i + 2, 1), dtype = np.complex64)
            bb[0] = normr0
            c = np.dot(Jtotal, bb)
            residual = abs(c[GMRES_i, 0])
            print(residual)

            if GMRES_i > 3:
                residual_set.append(residual)

            GMRES_i +=1

            GMRES_i_plus_1 = GMRES_i

            HH = np.dot(Jtotal, H)
            HH = HH[0 : GMRES_i_plus_1, :]
            cc = c[0 : GMRES_i_plus_1, 0:1]
            cc_new = np.linalg.solve(HH.reshape(GMRES_i_plus_1, GMRES_i_plus_1), cc.reshape(GMRES_i_plus_1, 1))

            x_add = np.dot(V[:, 0:GMRES_i_plus_1], cc_new)
            x = x0 + x_add.reshape(-1)

            rel_error = np.linalg.norm(x.reshape(-1) - x_reference.reshape(-1)) / norm_ref
            rel_error_set.append(rel_error)

            print("rel_error = %f" % rel_error)

        np.save(prefix_output + 'residuals_' + str(N) + '_' + str(repetition), residual_set)
        np.save(prefix_output + 'relative_errors_' + str(N) + '_' + str(repetition), rel_error_set)
