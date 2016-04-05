# computation module
from scipy import *
from scipy.linalg import inv, det, norm, svd
from itertools import product, chain

# integrate func over [-1, 1]^2 using k-order gauss quad
def dbl_gauss_quad(func, k):
    if k == 1:
        return 4 * func(0, 0)
    if k == 2:
        p = sqrt(1/3)
        return func(-p, -p) + func(-p, p) + func(p, -p) + func(p, p)
    if k == 3:
        p = [-sqrt(3/5), 0, sqrt(3/5)]
        w = [5/9, 8/9, 5/9]
        sum = func(0, 0) - func(0, 0)   # get the 0 element of the addition
        for i, j in product(range(3), range(3)):
            sum += func(p[i], p[j]) * w[i] * w[j]
        return sum
    print('integration of order ', k, 'not supported.')
    assert(0)

def compute_stiffness_matrix(type, nodes, material, thickness):
    if type == 'RECTANGLE4':
        return compute_stiffness_matrix_quad4(nodes, material, thickness)
    if type == 'RECTANGLE9':
        return compute_stiffness_matrix_quad9(nodes, material, thickness)
    print('unsupported element type')
    assert(0)
    
def compute_stiffness_matrix_quad4(nodes, material, thickness):

    # N1 = 1/4 (1-\xi) (1-\eta)     shape:
    # N2 = 1/4 (1+\xi) (1-\eta)         4 ---- 3
    # N3 = 1/4 (1+\xi) (1+\eta)         |      |
    # N4 = 1/4 (1-\xi) (1+\eta)         1 ---- 2
    
    # the sub-process specific to this type of elements, calculates (4.2.6) at (\xi, \eta)
    def Jacobi(xi, eta):
        partials = 0.25 * array([
            [eta-1, 1-eta, 1+eta, -1-eta],
            [xi-1,  -1-xi, 1+xi,  1-xi]
        ])
        coords = row_stack(nodes)
        return dot(partials, nodes)
    
    # computes (dN_i/dx, dN_i/dy) at (\xi, \eta), (4.2.7)
    def Partial_xy(xi, eta):
        J = Jacobi(xi, eta)
        partial_xe = 0.25 * array([
            [eta-1, 1-eta, 1+eta, -1-eta],
            [xi-1,  -1-xi, 1+xi,  1-xi]
        ])
        return dot(inv(J), partial_xe)
    
    # computes matrix B, eq (2.2.15), the multiplication of constant D_0 is delayed to final stage
    def compute_B(xi, eta):
        p = Partial_xy(xi, eta)
        return array([
            [ p[0,0],   0,      p[0,1], 0,      p[0,2], 0,      p[0,3], 0       ],
            [ 0,        p[1,0], 0,      p[1,1], 0,      p[1,2], 0,      p[1,3]  ],
            [ p[1,0],   p[0,0], p[1,1], p[0,1], p[1,2], p[0,2], p[1,3], p[0,3]  ],
        ])
    
    # elasticity matrix D, see table 1.2 on P42
    E, v = material
    D = array([
        [1,     v,      0        ],
        [v,     1,      0        ],
        [0,     0,      0.5*(1-v)]
    ])
    
    # integrand of eq (4.4.1)
    def integrand(xi, eta):
        J = Jacobi(xi, eta)
        B = compute_B(xi, eta)
        ans = dot(dot(transpose(B), D), B) * abs(det(J))
        # print(det(ans), norm((ans[0,:] + ans[2,:] + ans[4,:] + ans[6,:])))
        return ans
    
    i = (E / (1-v**2)) * dbl_gauss_quad(integrand, 2)
    return thickness * i
        
def compute_stiffness_matrix_quad9(nodes, material, thickness):
    # follow the same pattern as above quad4-method
    
    # N1 = 1/4 xi(xi-1) eta(eta-1)
    # N2 = 1/4 xi(xi+1) eta(eta-1)      shape:
    # N3 = 1/4 xi(xi+1) eta(eta+1)              4 ---- 7 ---- 3
    # N4 = 1/4 xi(xi-1) eta(eta+1)              |             |
    # N5 = 1/2 (1-xi^2) eta(eta-1)              8      9      6  
    # N6 = 1/2 xi(xi+1) (1-eta^2)               |             |
    # N7 = 1/2 (1-xi^2) eta(eta+1)              1 ---- 5 ---- 2
    # N8 = 1/2 xi(xi-1) (1-eta^2)
    # N9 = (1-xi^2) (1-eta^2)
    
    def Partials(xi, eta):
        return 0.25 * array([
            [ (2*xi-1)*eta*(eta-1), (2*xi+1)*eta*(eta-1), (2*xi+1)*eta*(eta+1), (2*xi-1)*eta*(eta+1), -4*xi*eta*(eta-1),    2*(2*xi+1)*(1-eta**2), -4*xi*eta*(1+eta),     2*(2*xi-1)*(1-eta**2), -8*xi*(1-eta**2) ],
            [ xi*(xi-1)*(2*eta-1),  xi*(xi+1)*(2*eta-1),  xi*(xi+1)*(2*eta+1),  xi*(xi-1)*(2*eta+1),  2*(1-xi**2)*(2*eta-1), -4*xi*(xi+1)*eta,      2*(1-xi**2)*(2*eta+1), -4*xi*(xi-1)*eta,      -8*(1-xi**2)*eta ]
        ])
        
    def Jacobi(xi, eta):
        partials = Partials(xi, eta)
        coords = row_stack(nodes)
        return dot(partials, nodes)
    
    def Partial_xy(xi, eta):
        J = Jacobi(xi, eta)
        partial_xe = Partials(xi, eta)
        return dot(inv(J), partial_xe)
        
    def compute_B(xi, eta):
        p = Partial_xy(xi, eta)
        zeros9 = [0 for i in range(9)]
        # the list(chain(*zip(~,~))) operation interleaves two array, if that is confusing to you, see quad4's implementation
        return array([
            list(chain(*zip(p[0,:], zeros9))),
            list(chain(*zip(zeros9, p[1,:]))),
            list(chain(*zip(p[1,:], p[0,:]))),
        ])
    
    E, v = material
    D = array([
        [1,     v,      0        ],
        [v,     1,      0        ],
        [0,     0,      0.5*(1-v)]
    ])
    
    def integrand(xi, eta):
        J = Jacobi(xi, eta)
        B = compute_B(xi, eta)
        ans = dot(dot(transpose(B), D), B) * abs(det(J))
        return ans
    
    i = (E / (1-v**2)) * dbl_gauss_quad(integrand, 3)
    return thickness * i
    
def test():
    nodes = array([ [0,0], [2,0], [2,2], [0,2], [1,0], [2,1], [1,2], [0,1], [1,1] ])
    compute_stiffness_matrix_quad9(nodes, (1, 0.3), 1)
    
# test()