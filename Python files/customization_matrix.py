import heterocl as hcl
import numpy as np

def top_matrix(P, Q, R, S, T, alpha, beta, gamma, dtype=hcl.Int(), target=None):
    hcl.init(dtype)
    A = hcl.placeholder((P, Q), "A")
    B = hcl.placeholder((Q, R), "B")
    C = hcl.placeholder((R, S), "C")
    D = hcl.placeholder((S, T), "D")
    E = hcl.placeholder((P, T), "E")
    F = hcl.placeholder((P, T), "F")

    def compute_matrix(A,B,C,D,E):
        r = hcl.reduce_axis(0, Q, "r")
        s = hcl.reduce_axis(0, S, "s")

        out_AB = hcl.compute((P, R),
                         lambda x, y: hcl.sum(A[x, r] * B[r, y], dtype=dtype, axis=r),
                         name="out_AB")
        out_CD = hcl.compute((R, T),
                         lambda x, y: hcl.sum(C[x, s] * D[s, y], dtype=dtype, axis=s),
                         name="out_CD")
       
        F = hcl.compute((P,T),
                        lambda x, y: alpha * out_AB[x, y] * beta * out_CD[x, y] + gamma * E[x, y],
                        name="F")

        return F

    s1 = hcl.create_schedule([A, B, C, D, E], compute_matrix)

    AB = compute_matrix.out_AB
    CD = compute_matrix.out_CD
    
    ###   Applying customizations   ###
    s1[AB].compute_at(s1[CD],CD.axis[0])
    s1[AB].parallel(AB.axis[0])                 
    s1[CD].parallel(CD.axis[0])                     
    s1[AB].pipeline(AB.axis[0])     
    s1[CD].pipeline(CD.axis[0]) 
    sm = hcl.create_scheme([A, B, C, D, E], compute_matrix)
    sm.downsize([compute_matrix.out_AB], hcl.Int(32))
    sm.downsize([compute_matrix.out_CD], hcl.Int(32))
    s2 = hcl.create_schedule_from_scheme(sm)

    return hcl.build(s2,target=target)
    return hcl.build(s1,target=target)
    
def main(P=2, Q=2, R=2, S=2, T=2, alpha=0.1, beta=0.1, gamma=0.1, dtype=hcl.Float(), target=None):
    hcl.init(dtype)
    f1 = top_matrix(P, Q, R, S, T, alpha, beta, gamma, dtype, target)

# Define input numpy arrays
    A = np.random.randint(10, size=(P, Q)).astype(np.float32)
    B = np.random.randint(10, size=(Q, R)).astype(np.float32)
    C = np.random.randint(10, size=(R, S)).astype(np.float32)
    D = np.random.randint(10, size=(S, T)).astype(np.float32)
    E = np.random.randint(10, size=(P, T)).astype(np.float32)
    F = np.zeros((P, T), dtype=np.float32) 

    hcl_A = hcl.asarray(A, dtype=dtype)
    hcl_B = hcl.asarray(B, dtype=dtype)
    hcl_C = hcl.asarray(C, dtype=dtype)
    hcl_D = hcl.asarray(D, dtype=dtype)
    hcl_E = hcl.asarray(E, dtype=dtype)
    hcl_F = hcl.asarray(F, dtype=dtype) 
    print(A)
    print(hcl_A)

    f1(hcl_A, hcl_B, hcl_C, hcl_D, hcl_E, hcl_F)

    result = hcl_F.asnumpy()

    golden= alpha* np.matmul(A, B) * beta * np.matmul(C, D) + gamma * E

    print(result)
    print(golden)
    
    if (
        np.allclose(golden, result)
        ):
        print("passed")
    else:
        print("failed")


if __name__ == "__main__":
    main()
