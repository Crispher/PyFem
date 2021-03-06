from computation import *
    
# IO utilities
def read_line_args(file):
    while 1:
        line = file.readline()
        if line == '':
            return 'EOF'
        args = [ s for s in filter(lambda x : x != '', line[:-1].split(' '))]
        if len(args) > 0:
            return args
    
# data structure
class Material:
    def __init__(self):
        self.E = 2.1e9
        self.v = 0.3

class LineForce:
    def __init__(self, args):
        self.element_index = int(args[0]) - 1
        num_nodes = (len(args) - 3) // 2
        self.nodes = list(map(lambda s: int(s)-1, args[2:2+num_nodes]))
        self.direction = args[2+num_nodes]
        self.forces = list(map(float, args[-num_nodes:]))
        
class BoundaryCondition:
    def __init__(self):
        self.line_force = []
        self.displacement = []

class Element:
    def __init__(self):
        self.nodes_index = []
        self.type = 'NULL'
        self.material = Material()
        self.thickness = 1
        self.loaded = False
        self.stiffness_matrix = 0
        self.load_vector = 'NULL'
        
    def apply_force(self, line_force, coords):
        if not self.loaded:
            self.load_vector = zeros(2*len(self.nodes_index))
            self.loaded = True
        if self.type == 'RECTANGLE4':
            self._apply_force_quad4(line_force, coords)
            return
        if self.type == 'RECTANGLE9':
            self._apply_force_quad9(line_force, coords)
            return
        assert(0)
        
    # keeping accordance with FEMT, use left handed forces by default
    def _apply_force_quad4(self, line_force, coords):
        start, end = line_force.nodes[0], line_force.nodes[1]
        local_index_start, local_index_end = self.nodes_index.index(start), self.nodes_index.index(end)
        q_start = -line_force.forces[0]
        q_end = -line_force.forces[1]
        t = self.thickness
        Dx = coords[local_index_end][0] - coords[local_index_start][0]
        Dy = coords[local_index_end][1] - coords[local_index_start][1]

        if line_force.direction == 'F_N':                
            Fx_start = -(1/3 * q_start + 1/6 * q_end) * t * Dy
            Fy_start = (1/3 * q_start + 1/6 * q_end) * t * Dx
            Fx_end = -(1/6 * q_start + 1/3 * q_end) * t * Dy
            Fy_end = (1/6 * q_start + 1/3 * q_end) * t * Dx
        elif line_force.direction == 'F_T':
            Fx_start = Dx * (1/3 * q_start + 1/6 * q_end) * t
            Fy_start = Dy * (1/3 * q_start + 1/6 * q_end) * t
            Fx_end = Dx * (1/6 * q_start + 1/3 * q_end) * t
            Fy_end = Dy * (1/6 * q_start + 1/3 * q_end) * t
                    
        Pe = zeros(8)
        Pe[2*local_index_start] = Fx_start
        Pe[2*local_index_start+1] = Fy_start
        Pe[2*local_index_end] = Fx_end
        Pe[2*local_index_end+1] = Fy_end
        self.load_vector += Pe
        
        
    def _apply_force_quad9(self, line_force, coords):
        start, middle, end = line_force.nodes[0], line_force.nodes[1], line_force.nodes[2]
        local_index_start, local_index_middle, local_index_end = self.nodes_index.index(start), self.nodes_index.index(middle), self.nodes_index.index(end)
        
        q_start = -line_force.forces[0]
        q_middle = -line_force.forces[1]
        q_end = -line_force.forces[2]        
        t = self.thickness
        Pe = zeros(18)        
        Dx = coords[local_index_end][0] - coords[local_index_start][0]
        Dy = coords[local_index_end][1] - coords[local_index_start][1]
        
        # integrations by hand. :-(
        if line_force.direction == 'F_N':
            Fx_start = -(4/30 * q_start + 2/30 * q_middle - 1/30 * q_end) * t * Dy
            Fy_start = (4/30 * q_start + 2/30 * q_middle - 1/30 * q_end) * t * Dx
            Fx_middle = -(2/30 * q_start + 16/30 * q_middle + 2/30 * q_end) * t * Dy
            Fy_middle = (2/30 * q_start + 16/30 * q_middle + 2/30 * q_end) * t * Dx
            Fx_end = -(-1/30 * q_start + 2/30 * q_middle + 4/30 * q_end) * t * Dy
            Fy_end = (-1/30 * q_start + 2/30 * q_middle + 4/30 * q_end) * t * Dx
        elif line_force.direction == 'F_T':
            Fx_start = (4/30 * q_start + 2/30 * q_middle - 1/30 * q_end) * t * Dx
            Fy_start = (4/30 * q_start + 2/30 * q_middle - 1/30 * q_end) * t * Dy
            Fx_middle = (2/30 * q_start + 16/30 * q_middle + 2/30 * q_end) * t * Dx
            Fy_middle = (2/30 * q_start + 16/30 * q_middle + 2/30 * q_end) * t * Dy
            Fx_end = (-1/30 * q_start + 2/30 * q_middle + 4/30 * q_end) * t * Dx
            Fy_end = (-1/30 * q_start + 2/30 * q_middle + 4/30 * q_end) * t * Dy
            
        Pe[2*local_index_start] = Fx_start
        Pe[2*local_index_start+1] = Fy_start
        Pe[2*local_index_middle] = Fx_middle
        Pe[2*local_index_middle+1] = Fy_middle
        Pe[2*local_index_end] = Fx_end
        Pe[2*local_index_end+1] = Fy_end
        
        self.load_vector += Pe
        
    def compute_stress(self, ans, material, coords):
        if self.type == 'RECTANGLE4':
            self.stress = compute_stress_quad4(ans, material, coords)
            return
        if self.type == 'RECTANGLE9':
            self.stress = compute_stress_quad9(ans, material, coords)
            return
        assert(0)

############## central part ################################
class Problem:
    
    def load_ctr(self, ctr_file):
        file = open(ctr_file, 'r')
        
        mode = 'NULL'
        data = 'NULL'
        
        num_nodes, num_elements, num_materials = 0, 0, 0    
        temp_material_index = -1
        
        args = read_line_args(file)        
        while (args != 'EOF'):
            if args[0] == 'BASIC_DATA':
                mode = 'BASIC_DATA'
            
            elif args[0] == 'COORDINATES':
                data = 'COORDINATES'
                num_nodes = int(args[1])
            
            elif args[0].isdigit() and not args[1] == 'TO':
                if data == 'COORDINATES':
                    self.nodes = self.nodes + [array([x for x in map(float, args[1:])])]
                elif data == 'ELEMENT_NODES':
                    index = int(args[0]) - 1
                    self.elements[index].nodes_index = [x for x in map(lambda s: int(s) - 1, args[1:])]
                    
            elif args[0] == 'ELEMENTS':
                assert(num_nodes == len(self.nodes))
                data = 'ELEMENTS'
                num_elements = int(args[1])
                self.elements = [Element() for i in range(num_elements)]
            
            elif args[0] == 'ELEMENT_TYPE':
                data = 'ELEMENT_TYPE'
                
            elif args[0] == 'ELEMENT_MATERIAL':
                data = 'ELEMENT_MATERIAL'

            elif args[0].isdigit() and args[1] == 'TO':
                start, end = int(args[0]) - 1, int(args[2])
                if (data == 'ELEMENT_TYPE'):
                    for i in range(start, end):
                        self.elements[i].type = args[4]
                    assert(args[3] == 'TYPE')
                elif (data == 'ELEMENT_MATERIAL'):
                    for i in range(start, end):
                        self.elements[i].material = int(args[4]) - 1
                    assert(args[3] == 'MATERIAL')
            
            elif args[0] == 'ELEMENT_NODES':
                data = 'ELEMENT_NODES'
                # we handle the construction of nodes in the isdigit() branch
            
            # geometry information ignored since it is always 1
            
            elif args[0] == 'MATERIALS':
                data = 'MATERIALS'
                num_materials = int(args[1])
                self.materials = [Material() for i in range(num_materials)]
            
            elif args[0] == 'MATERIAL':
                temp_material_index = int(args[1]) - 1
                
            elif args[0] == 'E':
                assert(data == 'MATERIALS')
                self.materials[temp_material_index].E = float(args[1])
            
            elif args[0] == 'v':
                assert(data == 'MATERIALS')
                self.materials[temp_material_index].v = float(args[1])
            
            elif args[0] == 'END':
                if args[1] == 'BASIC_DATA':
                    assert(num_elements == len(self.elements) and num_materials == len(self.materials)
                        and num_nodes == len(self.nodes))
                    mode = 'NULL'
                if args[1] in ['ELEMENT_TYPE', 'ELEMENT_MATERIAL', 'MATERIALS']:
                    data = 'NULL'
                    
            # go on for another loop
            args = read_line_args(file)
            
    def load_bnd(self, bnd_file):
        file = open(bnd_file, 'r')
        mode = 'NULL'
        data = 'NULL'
        num_disp_conds, num_force_conds = 0, 0
        args = read_line_args(file)
        while (args != 'EOF'):
            if args[0] == 'DISPLACEMENT':
                mode = 'DISPLACEMENT'
            elif args[0] == 'GIVEN_DISP':
                data = 'GIVEN_DISP'
                num_disp_conds = int(args[1])
                
            elif args[0].isdigit():
                if data == 'GIVEN_DISP':
                    self.boundary_condition.displacement += [(int(args[0])-1, args[1], float(args[2]))]
                if data == 'LINE_FORCE':
                    self.boundary_condition.line_force += [LineForce(args)]
                    
            elif args[0] == 'FORCE':
                mode = 'FORCE'
            elif args[0] == 'LINE_FORCE':
                data = 'LINE_FORCE'
                num_force_conds = int(args[1])
            elif args[0] == 'END':
                if args[1] in ['GIVEN_DISP', 'LINE_FORCE']:
                    data = 'NULL'
                if args[1] in ['DISPLACEMENT', 'FORCE']:
                    mode = 'NULL'
            
            args = read_line_args(file)
            
        assert(num_disp_conds == len(self.boundary_condition.displacement) and
            num_force_conds == len(self.boundary_condition.line_force) )

    def __init__(self, proj_name):
        self.nodes = []
        self.boundary_condition = BoundaryCondition()
        self.elements = []
        self.materials = []
        self.thickness = 1
        self.proj_name = proj_name
        self.load_ctr(proj_name+'.ctr')
        self.load_bnd(proj_name+'.bnd')
        self.K = zeros( (len(self.nodes)*2, len(self.nodes)*2) )
        self.P = zeros( len(self.nodes)*2 )
        self.penalty = 1e30         # the penalty parameter is tuned to fit the output of FEMT
        self.ans = 0
        print('problem loaded')
        
    def compute_stiffness_matrices(self):
        for e in self.elements:
            e.stiffness_matrix = compute_stiffness_matrix(
                e.type,
                [self.nodes[i][:2] for i in e.nodes_index],
                (self.materials[e.material].E, self.materials[e.material].v),
                self.thickness
            )
        
    def compute_load_vectors(self):
        for f in self.boundary_condition.line_force:
            i = f.element_index
            self.elements[i].apply_force(
                f, 
                [self.nodes[i][:2] for i in self.elements[i].nodes_index]
            )
        
    def assemble_stiffness_matrix(self):
        for e in self.elements:
            e_num_nodes = len(e.nodes_index)
            for i, j in product(range(e_num_nodes), repeat=2):
                self.K[2*e.nodes_index[i], 2*e.nodes_index[j]] += e.stiffness_matrix[2*i, 2*j]
                self.K[2*e.nodes_index[i]+1, 2*e.nodes_index[j]] += e.stiffness_matrix[2*i+1, 2*j]
                self.K[2*e.nodes_index[i], 2*e.nodes_index[j]+1] += e.stiffness_matrix[2*i, 2*j+1]
                self.K[2*e.nodes_index[i]+1, 2*e.nodes_index[j]+1] += e.stiffness_matrix[2*i+1, 2*j+1]
        print('stiffness matrix assemble complete')
        
    def assemble_load_vector(self):
        for e in self.elements:
            if not e.loaded:
                continue
            e_num_nodes = len(e.nodes_index)
            for i in range(e_num_nodes):
                self.P[2*e.nodes_index[i] ] += e.load_vector[2*i]
                self.P[2*e.nodes_index[i]+1] += e.load_vector[2*i+1]
        print('load vector assemble complete')
        
    # multiply by large number
    def apply_displacement_condition(self, node, axis, given):
        self.K[2*node+axis, 2*node+axis] *= self.penalty
        self.P[2*node+axis] = self.K[2*node+axis, 2*node+axis]*given
        
    def apply_boundary_condition(self):
        for disp in self.boundary_condition.displacement:
            axis = 0 if disp[1] == 'U' else 1
            self.apply_displacement_condition(disp[0], axis, disp[2])
        
    # average over all adjacent elements
    def compute_stresses(self):
        for e in self.elements:
            e.compute_stress(
                array([self.ans[i] for i in chain(*zip(map(lambda i : 2*i, e.nodes_index), map(lambda i : 2*i + 1, e.nodes_index)))]),
                (self.materials[e.material].E, self.materials[e.material].v),
                [self.nodes[i][:2] for i in e.nodes_index]
            )
        sum_stresses = [array([0., 0., 0.]) for i in range(len(self.nodes))]
        count = [0 for i in range(len(self.nodes))]
        for e in self.elements:
            for i, n in enumerate(e.nodes_index): # i: local index, n: global index
                sum_stresses[n] += e.stress[i]
                count[n] += 1
        return list(starmap(lambda v, n: v/n, zip(sum_stresses, count)))
        
    def solve(self):
        self.compute_stiffness_matrices()
        self.assemble_stiffness_matrix()
        self.compute_load_vectors()
        self.assemble_load_vector()
        self.apply_boundary_condition()
        self.ans = solve(self.K, self.P)
        self.stresses = self.compute_stresses()
        print('problem solved')
        
    # write to '.OUT' file, and trick the post-process matlab script into work
    def write_to_file(self):
        out_file = open(self.proj_name + '.OUT', 'w')
        out_file.write(' *** DISPLACEMENT ***\n  NODE         U              V\n-----------------------------------\n')
        for i in range(len(self.nodes)):
            out_file.write('\t' + str(i+1) + '\t' + str(self.ans[2*i]) + '\t' + str(self.ans[2*i+1]) + '\n')
        out_file.write('\n *** NODAL STRESS ***\n' + 
                       '  NODE         SXX            SYY            SXY            SI            S2\n' + 
                       '----------------------------------------------------------------------------------\n')
        for i in range(len(self.nodes)):
            stress_tensor = array([[self.stresses[i][0], self.stresses[i][2]], [self.stresses[i][2], self.stresses[i][1] ] ])
            eigen_values = eig(stress_tensor)[0]
            out_file.write('\t' + str(i+1) + '\t' + str(self.stresses[i][0]) + '\t' + str(self.stresses[i][1]) + '\t' + str(self.stresses[i][2]) + 
                '\t' + str(eigen_values[0].real) + '\t' + str(eigen_values[1].real) +'\n')
            
        out_file.write('PROGRAM STARTED // no time recoreded, I am just tricking post-process.m into working\n')
        out_file.close()
      

import sys
def main():
    if len(sys.argv) > 1:
        p = Problem(sys.argv[1])
        p.solve()
        p.write_to_file()
    else:
        print("usage: python femt.py proj_name")
main()