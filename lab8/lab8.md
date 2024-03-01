```python
import matplotlib.pyplot as plt
import scipy
import sympy
import numpy as np
import control 
import os
import array_to_latex as a2l
from LatexGapFiller import LatexGapFiller, TextSheet
```

# Task 1

Задание 1 про управляемость


```python
def check_eigenvalues_controllable(A, B) -> np.array:
    assert len(A.shape) == 2
    assert A.shape[0] == A.shape[1]
    assert A.shape[0] == B.shape[0]
    
    n = A.shape[0]
    is_controllable = []

  
    eigen_values, _ = np.linalg.eig(A)
    for eigen_value in eigen_values:
        M = np.concatenate((A - eigen_value * np.eye(n), B.reshape(-1, 1)), axis=1)
        is_controllable.append(np.linalg.matrix_rank(M) == n)

    return np.array(is_controllable)
```


```python
def get_observation_matrix(A, C):
    assert len(A.shape) == 2
    assert A.shape[0] == A.shape[1]
    assert A.shape[0] == C.shape[1]

    n = A.shape[0]
    U = np.zeros((0, n))
    for i in range(n):
        new_column = (C @ np.linalg.matrix_power(A, i))
        U = np.concatenate((U, new_column))

    return U
    
```


```python
def check_system_observable(A, C, method = 'rank_criteria'):
    assert len(A.shape) == 2
    assert A.shape[0] == A.shape[1]
    assert A.shape[0] == C.shape[1]
    
    is_observable = None
    n = A.shape[0]
    if method == 'rank_criteria':
        V = get_observation_matrix(A, C)
        is_observable = np.linalg.matrix_rank(V) == n
    else:
        raise NotImplementedError
    
    return is_observable
```


```python
def create_g_matrix(eigvalues):
    
    return np.diag(eigvalues)
```


```python
def modeling(A, B, K):
    ss = control.ss((A + B @ K), B*0, A * 0, B * 0)

    time = np.linspace(0, 3, 1000)
    output = control.forced_response(ss, X0=[1, 1, 1, 1], T=time).states

    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4,hspace=0.4)
    for i in range(4):
        axs[i//2, i % 2].plot(time, output[i], linewidth=4)
        # axs[i//2, i % 2].scatter(t[-1], x1[i], color='r', linewidth=4, label='goal')
        axs[i//2, i % 2].set_title(f'x_{i} = x_{i}(t)', fontsize=12)
        axs[i//2, i % 2].set_xlabel(f"t, [c]", fontsize=12)
        axs[i//2, i % 2].grid(True)
        axs[i//2, i % 2].legend()
        
        
    # axs[1, 1].plot(t, np.array(U).reshape(-1), linewidth=4)
    # axs[1, 1].set_title(f'u = u(t)', fontsize=12)
    # axs[1, 1].set_xlabel(f"t, [c]", fontsize=12)
    # axs[1, 1].grid(True)
```


```python
gap_filler = LatexGapFiller()
A = np.array([
    [-5, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 2, 6],
    [0, 0, -6, 2],
])
B = np.array([
    [0],
    [1],
    [0],
    [3]
])

```


```python
with open('./task1.txt', 'r') as file:
        data = file.read()

text = TextSheet(0, data)
text.fill_gap({"A": f"{i}"})
gap_filler.add_sheet(text)
```

    {A} 0



    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[35], line 7
          5 text = TextSheet(0, data)
          6 text.fill_gap({"A": f"{i}"})
    ----> 7 gap_filler.add_sheet(text)


    NameError: name 'gap_filler' is not defined



```python
check_eigenvalues_controllable(A, B)
```




    array([ True,  True, False,  True])




```python
np.linalg.eig(A)[0]
```




    array([ 2.+6.j,  2.-6.j, -5.+0.j,  0.+0.j])




```python
desired_eigvalues = [-5, -5, -5, -5]
G = create_g_matrix(desired_eigvalues)
G[0, 1] = 1
G[1, 2] = 1
G[2, 3] = 1
G
```




    array([[-5,  1,  0,  0],
           [ 0, -5,  1,  0],
           [ 0,  0, -5,  1],
           [ 0,  0,  0, -5]])




```python
np.linalg.eig(G)[0]
```




    array([-5., -5., -5., -5.])




```python
Y = np.array([[0, 1, 0, 0]])
```


```python
check_system_observable(G, Y)
```




    False




```python
P = scipy.linalg.solve_sylvester(A, -G, B@Y)
P
```




    array([[ 0.        ,  0.        ,  0.        ,  0.        ],
           [ 0.        ,  0.2       ,  0.04      ,  0.008     ],
           [ 0.        , -0.21176471, -0.03487889, -0.00325341],
           [ 0.        ,  0.24705882,  0.00539792, -0.0020175 ]])




```python
K = -Y @ np.linalg.pinv(P)
K
```




    array([[ 0.        , -3.125     , -4.40277778, -5.29166667]])




```python
np.linalg.eig(A+B@K)[0]
```




    array([-5.00003305+5.72532191e-05j, -5.00003305-5.72532191e-05j,
           -4.99993389+0.00000000e+00j, -5.        +0.00000000e+00j])




```python
modeling(A, B, K)
```

    No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.
    No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.
    No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.
    No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.



    
![png](lab8_files/lab8_19_1.png)
    

