"""Independent Cirq circuits for reduced application-model adaptations."""
import numpy as np
import cirq


def kernel_state(x, topology='full'):
    n = len(x); q = cirq.LineQubit.range(n); c = cirq.Circuit()
    if topology in ('x','y','z'):
        for i in range(n):
            if topology == 'z': c.append(cirq.H(q[i]))
            c.append({'x':cirq.rx,'y':cirq.ry,'z':cirq.rz}[topology](float(x[i]))(q[i]))
    else:
        for i in range(n): c.append(cirq.H(q[i])); c.append(cirq.rz(float(x[i]))(q[i]))
        if topology == 'full': edges = [(i,j) for i in range(n) for j in range(i+1,n)]
        else:
            edges = [(i,i+1) for i in range(n-1)]
            if topology == 'circular' and n > 2: edges.append((n-1,0))
        for i,j in edges:
            c.append(cirq.CNOT(q[i],q[j])); c.append(cirq.rz(float(x[i]*x[j]))(q[j])); c.append(cirq.CNOT(q[i],q[j]))
    return cirq.Simulator(dtype=np.complex128).simulate(c,qubit_order=q).final_state_vector


def qvc_state(amplitudes, theta):
    n = theta.shape[1]; q = cirq.LineQubit.range(n); c = cirq.Circuit()
    for layer in theta:
        for i in range(n):
            c.append(cirq.rz(float(layer[i,0]))(q[i])); c.append(cirq.ry(float(layer[i,1]))(q[i])); c.append(cirq.rz(float(layer[i,2]))(q[i]))
        for i in range(n-1): c.append(cirq.CZ(q[i],q[i+1]))
    # Simulator state access is explicit; physical loading is not free.
    return cirq.Simulator(dtype=np.complex128).simulate(c,initial_state=np.asarray(amplitudes,dtype=complex),qubit_order=q).final_state_vector


def entity_state(angles, mixing, strong):
    n = len(angles); q = cirq.LineQubit.range(n); c = cirq.Circuit()
    edges = [(0,1)] if n == 2 else [(i,(i+1)%n) for i in range(n)]
    for layer in mixing:
        for i in range(n): c.append(cirq.rx(float(angles[i]))(q[i]))
        for k,(i,j) in enumerate(edges):
            c.append(cirq.CNOT(q[i],q[j])); c.append(cirq.rz(float(layer[k]))(q[j])); c.append(cirq.CNOT(q[i],q[j]))
        for i in range(n): c.append(cirq.ry(float(layer[len(edges)+i]))(q[i]))
    for i in range(n): c.append(cirq.rx(float(angles[i]))(q[i]))
    for layer in strong:
        for i in range(n):
            c.append(cirq.rz(float(layer[i,0]))(q[i])); c.append(cirq.ry(float(layer[i,1]))(q[i])); c.append(cirq.rz(float(layer[i,2]))(q[i]))
        for i in range(n): c.append(cirq.CNOT(q[i],q[(i+1)%n]))
    return cirq.Simulator(dtype=np.complex128).simulate(c,qubit_order=q).final_state_vector
