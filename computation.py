# computation module. rely on scipy library.
from scipy import *                             # basic vector operations
from scipy.linalg import inv, det, eig, solve   # 4 linear algebra functions  
from itertools import product, chain, starmap   # functional programming tools

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

# notations for simplicity
# elasticity matrix D, see table 1.2 on P42
def get_D(E, v):
    return (E / (1-v**2)) * array([
        [1,     v,      0        ],
        [v,     1,      0        ],
        [0,     0,      0.5*(1-v)]
    ])
    
############# RECT 4 #######################
# N1 = 1/4 (1-\xi) (1-\eta)     shape:
# N2 = 1/4 (1+\xi) (1-\eta)         4 ---- 3
# N3 = 1/4 (1+\xi) (1+\eta)         |      |
# N4 = 1/4 (1-\xi) (1+\eta)         1 ---- 2
    
# dN_i/d xi, dN_i/d eta
def partials_quad4(xi, eta):
    return 0.25 * array([
        [eta-1, 1-eta, 1+eta, -1-eta],
        [xi-1,  -1-xi, 1+xi,  1-xi]
    ])

############# RECT 9 ####################### 
# N1 = 1/4 xi(xi-1) eta(eta-1)
# N2 = 1/4 xi(xi+1) eta(eta-1)   shape:
# N3 = 1/4 xi(xi+1) eta(eta+1)     4 ---- 7 ---- 3
# N4 = 1/4 xi(xi-1) eta(eta+1)     |             |
# N5 = 1/2 (1-xi^2) eta(eta-1)     8      9      6  
# N6 = 1/2 xi(xi+1) (1-eta^2)      |             |
# N7 = 1/2 (1-xi^2) eta(eta+1)     1 ---- 5 ---- 2
# N8 = 1/2 xi(xi-1) (1-eta^2)
# N9 = (1-xi^2) (1-eta^2)

def partials_quad9(xi, eta):
    # better use a large screen to read this function :-)
    return 0.25 * array([
        [ (2*xi-1)*eta*(eta-1), (2*xi+1)*eta*(eta-1), (2*xi+1)*eta*(eta+1), (2*xi-1)*eta*(eta+1), -4*xi*eta*(eta-1),     2*(2*xi+1)*(1-eta**2), -4*xi*eta*(1+eta),     2*(2*xi-1)*(1-eta**2), -8*xi*(1-eta**2) ],
        [ xi*(xi-1)*(2*eta-1),  xi*(xi+1)*(2*eta-1),  xi*(xi+1)*(2*eta+1),  xi*(xi-1)*(2*eta+1),  2*(1-xi**2)*(2*eta-1), -4*xi*(xi+1)*eta,      2*(1-xi**2)*(2*eta+1), -4*xi*(xi-1)*eta,      -8*(1-xi**2)*eta ]
    ])
    
############################################

# interface for Problem classes.
def compute_stiffness_matrix(type, nodes, material, thickness):
    if type == 'RECTANGLE4':
        return compute_stiffness_matrix_quad(nodes, material, thickness, partials_quad4)
    if type == 'RECTANGLE9':
        return compute_stiffness_matrix_quad(nodes, material, thickness, partials_quad9, integral_order=3)
    print('unsupported element type')
    assert(0)


'''
Implementation detail, things become generic now. This pattern fits all types of nodes
'''
    
def get_matrix(B=False, Jacobi=False, nodes, partials, xi, eta):
    assert(B or Jacobi)
    coords = row_stack(nodes)
    p_xe = partials(xi,eta)
    J = dot(p_xe, nodes)
    if not B:
        return J
    else:
        # partials with respect to x, y
        p = dot(inv(J), p_xe)
        _zeros = [0 for i in range(num_nodes)]
        _B = array([
            list(chain(*zip(p[0,:], _zeros))),
            list(chain(*zip(_zeros, p[1,:]))),
            list(chain(*zip(p[1,:], p[0,:]))),
        ])
        return _B, J
    
# the 'partials' argument passes a function to compute d(N_i)/d(xi, eta) at (xi, eta)
def compute_stiffness_matrix_quad(nodes, material, thickness, partials, integral_order=2):

    # calculates (4.2.6) at (\xi, \eta)
    def Jacobi(xi, eta):
        coords = row_stack(nodes)
        return dot(partials(xi,eta), nodes)
    
    # computes (dN_i/dx, dN_i/dy) at (\xi, \eta), (4.2.7)
    def partial_xy(xi, eta):
        return dot(inv(Jacobi(xi, eta)), partials(xi, eta))
    
    # computes matrix B, eq (2.2.15), the multiplication of constant D_0 is delayed to final stage
    # the list(chain(*zip(~,~))) operation interleaves two arrays
    def compute_B(xi, eta):
        p = partial_xy(xi, eta)
        num_nodes = len(p[0,:])
        _zeros = [0 for i in range(num_nodes)]
        return array([
            list(chain(*zip(p[0,:], _zeros))),
            list(chain(*zip(_zeros, p[1,:]))),
            list(chain(*zip(p[1,:], p[0,:]))),
        ])
    
    D = get_D(*material)
    
    # integrand of eq (4.4.1)
    def integrand(xi, eta):
        J = Jacobi(xi, eta)
        B = compute_B(xi, eta)
        return dot(dot(transpose(B), D), B) * abs(det(J))
    
    return thickness * dbl_gauss_quad(integrand, integral_order)
    
def compute_stress_quad4(disp, material, nodes):
    def partials(xi, eta):
        return 0.25 * array([
            [eta-1, 1-eta, 1+eta, -1-eta],
            [xi-1,  -1-xi, 1+xi,  1-xi]
        ])
        
    def Jacobi(xi, eta):
        coords = row_stack(nodes)
        return dot(partials(xi,eta), nodes)
    
    def partial_xy(xi, eta):
        return dot(inv(Jacobi(xi, eta)), partials(xi, eta))
    
    def B(xi, eta):
        p = partial_xy(xi, eta)
        return array([
            [ p[0,0],   0,      p[0,1], 0,      p[0,2], 0,      p[0,3], 0       ],
            [ 0,        p[1,0], 0,      p[1,1], 0,      p[1,2], 0,      p[1,3]  ],
            [ p[1,0],   p[0,0], p[1,1], p[0,1], p[1,2], p[0,2], p[1,3], p[0,3]  ],
        ])
    
    strain_at_gaussian_nodes = [dot(B(-1/sqrt(3),-1/sqrt(3)), disp), dot(B(1/sqrt(3),-1/sqrt(3)), disp), dot(B(1/sqrt(3), 1/sqrt(3)), disp), dot(B(-1/sqrt(3), 1/sqrt(3)), disp)]
    
    p, q = 1+1/sqrt(3), 1-1/sqrt(3)
    weight_matrix = 1/4 * array([
        [p*p, p*q, q*q, q*p],
        [p*q, p*p, p*q, q*q],
        [q*q, p*q, p*p, p*q],
        [p*q, q*q, p*q, p*p]
    ])
    
    D = get_D(*material)
    
    stress_at_gaussian_nodes = array([dot(D, e) for e in strain_at_gaussian_nodes])
    return dot(inv(weight_matrix), stress_at_gaussian_nodes)
    
def compute_stress_quad9(disp, material, nodes):
    def partials(xi, eta):
        # better use a large screen to read this function :-)
        return 0.25 * array([
            [ (2*xi-1)*eta*(eta-1), (2*xi+1)*eta*(eta-1), (2*xi+1)*eta*(eta+1), (2*xi-1)*eta*(eta+1), -4*xi*eta*(eta-1),     2*(2*xi+1)*(1-eta**2), -4*xi*eta*(1+eta),     2*(2*xi-1)*(1-eta**2), -8*xi*(1-eta**2) ],
            [ xi*(xi-1)*(2*eta-1),  xi*(xi+1)*(2*eta-1),  xi*(xi+1)*(2*eta+1),  xi*(xi-1)*(2*eta+1),  2*(1-xi**2)*(2*eta-1), -4*xi*(xi+1)*eta,      2*(1-xi**2)*(2*eta+1), -4*xi*(xi-1)*eta,      -8*(1-xi**2)*eta ]
        ])
        
    def Jacobi(xi, eta):
        p = partials(xi, eta)
        coords = row_stack(nodes)
        return dot(p, nodes)
    
    def partial_xy(xi, eta):
        J = Jacobi(xi, eta)
        partial_xe = partials(xi, eta)
        return dot(inv(J), partial_xe)
        
    def B(xi, eta):
        p = partial_xy(xi, eta)
        zeros9 = [0 for i in range(9)]
        return array([
            list(chain(*zip(p[0,:], zeros9))),
            list(chain(*zip(zeros9, p[1,:]))),
            list(chain(*zip(p[1,:], p[0,:]))),
        ])
        
    D = get_D(*material)
    
    # only want 4 nodes with high precision
    strain_at_internal_nodes = [dot(B(0,-1), disp), dot(B(1,0), disp), dot(B(0,1), disp), dot(B(-1,0), disp), dot(B(0,0), disp)]
        
    strain_at_gaussian_nodes = [dot(B(-sqrt(3/5),-sqrt(3/5)), disp), dot(B(sqrt(3/5),-sqrt(3/5)), disp), dot(B(sqrt(3/5), sqrt(3/5)), disp), dot(B(-sqrt(3/5), sqrt(3/5)), disp)]
    
    p, q = 1+1/sqrt(3), 1-1/sqrt(3)
    weight_matrix = 1/4 * array([
        [p*p, p*q, q*q, q*p],
        [p*q, p*p, p*q, q*q],
        [q*q, p*q, p*p, p*q],
        [p*q, q*q, p*q, p*p]
    ])
    
    stress_at_gaussian_nodes = array([dot(D, e) for e in strain_at_gaussian_nodes])
    stress_at_corner_nodes = dot(inv(weight_matrix), stress_at_gaussian_nodes)  # high precision
    stress_at_internal_nodes = [dot(D, e) for e in strain_at_internal_nodes]    # low precision
    return row_stack((stress_at_corner_nodes, stress_at_internal_nodes))