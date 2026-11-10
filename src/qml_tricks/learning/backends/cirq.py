import numpy as np
import cirq

def state(theta, x, n, depth):
    q = cirq.LineQubit.range(n); c = cirq.Circuit()
    for l in range(depth):
        for i in range(n):
            c.append(cirq.ry(float(x[i]))(q[i]))
            c.append(cirq.rz(float(x[i]/2))(q[i]))
            c.append(cirq.ry(float(theta[l*n+i]))(q[i]))
        for i in range(n-1): c.append(cirq.CZ(q[i],q[i+1]))
    return cirq.Simulator(dtype=np.complex128).simulate(c,qubit_order=q).final_state_vector
