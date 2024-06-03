'''
    Abhilash Reddy Malipeddi, December 2015.
    Laplace Beltrami operator on a function sampled at the vertices using the direct 
    discretization via Gauss divergence theorem from the paper Xu 2004[1].
    This scheme is named  $nabla^D_S$ in the paper
    [1]Xu, Guoliang. "Convergent discrete laplace-beltrami operators over
    triangular surfaces." Geometric Modeling and Processing, 2004. Proceedings. IEEE, 2004.
'''
import numpy
import scipy.sparse
import math
import tqdm
import time
import torch
#import openmesh as om
import open3d as o3d
from utils import *

cross = lambda x,y:np.cross(x,y)

def GetTopologyAroundVertex(vertices,faces):
    '''
    Calculates the topology around the vertices:
    Returns
    1.  vertex_surr_faces   :  indices of the faces that surround each vertex.
    2.  vertex_across_edge  :  for vertex 'i' and triangle 'j' gives vertex 'k'
    3.  face_across_edge    :  for vertex 'i' and triangle 'j' gives face 'f' -- refer to fig 
    #
    #                                i
    #                               / \
    #                              / j \
    #                             /_____\
    #                             \     /
    #                              \ f /
    #                               \ / 
    #                                k
    '''
    MaxValence=50
    npt=vertices.shape[0]
    vertex_across_edge  =-numpy.ones([npt,MaxValence],dtype=int)
    face_across_edge    =-numpy.ones([npt,MaxValence],dtype=int)   
    vertex_surr_faces   =-numpy.ones([npt,MaxValence],dtype=int)   # index of the faces that surrounds each vertex
    vertexValence       = numpy.zeros([npt,1],dtype=int)


    FaceNeighbors=GetFaceNeighbors(vertices,faces)

#This loop will error if the valency becomes greater than MaxValence
    for ke,f in enumerate(faces):
        vertexValence[f]+=1 # valence/number of neighbours of f'th point
        # store the index of the elements surrounding each vertex pointed to by f
        vertex_surr_faces[f[0],vertexValence[f[0]]-1]=ke
        vertex_surr_faces[f[1],vertexValence[f[1]]-1]=ke
        vertex_surr_faces[f[2],vertexValence[f[2]]-1]=ke

    #vertex across edge
    for i,vsf in enumerate(vertex_surr_faces):
# vertex_surr_faces is initialized with -1 throughout. The actual count of faces will usually
# be different for different vertices. There is no way to know this count before hand.
# The below line keeps only face indices that are greater than -1, which are what 
# actually exist.
      vsf=vsf[vsf>-1]
      for j,vj in enumerate(vsf):
         for f in FaceNeighbors[vj]:
            if f<0:
                continue
            t=faces[f]
            if t[0]!=i and t[1]!=i and t[2]!=i:
               vertex_across_edge[i,j]=numpy.setdiff1d(t,faces[vj])
               face_across_edge[i,j]=f
    return vertex_surr_faces,vertex_across_edge,face_across_edge

def GetFaceNeighbors(vertices,faces):
    '''
    build neighbouring element information: method 1
    '''
    n2e=scipy.sparse.lil_matrix((vertices.shape[0],faces.shape[0]),dtype=int)
    FaceNeighbor=-numpy.ones(faces.shape,dtype=int)

#build adjcency matrix
    for i,t in enumerate(faces):
        n2e[t,i]=numpy.ones([3,1],dtype=int)

    n2e=n2e.tocsr()
    for i,t in enumerate(faces):
    #   if i == 89:
    #     print(i)
      nb=numpy.intersect1d(numpy.nonzero(n2e[t[1]])[1],numpy.nonzero(n2e[t[2]])[1])
      nb=numpy.setdiff1d(nb,i)
      if nb.shape[0]==1:
          FaceNeighbor[i,0]=nb[0]
      nb=numpy.intersect1d(numpy.nonzero(n2e[t[2]])[1],numpy.nonzero(n2e[t[0]])[1])
      nb=numpy.setdiff1d(nb,i)
      if nb.shape[0]==1:
          FaceNeighbor[i,1]=nb[0]
      nb=numpy.intersect1d(numpy.nonzero(n2e[t[0]])[1],numpy.nonzero(n2e[t[1]])[1])
      nb=numpy.setdiff1d(nb,i)
      if nb.shape[0]==1:
          FaceNeighbor[i,2]=nb[0]
    return FaceNeighbor

#Area = 

def ConventionalDiscreteLB(verts_input, faces_input, inpfunc):
    '''
    1. Turn mesh into half edge data structure.
    2. Figure out mesh topology.
    ----- Steps before this line are topology related with O(n) time complexity using for loops on CPU -----
    3. Calculate the voronoi area of each triangle in the mesh.
    4. For every half edge, compute cotangent alpha.
    5. Directly calculate L using the the LBO formula
    ----- These steps have O(n) time complexity, but can run with CUDA without for loops。
    '''

    # idx = numpy.array(list(range(faces.shape[0])), dtype=numpy.int32)
    # idx = numpy.stack([idx, idx, idx], axis=-1)
    # entries = numpy.ones(idx.shape[0] * 3, dtype=numpy.int32)
    #vertex_surr_faces_mat = scipy.sparse.coo_matrix((entries, (idx.reshape(-1), faces.reshape(-1))))
    
    t0 = time.time()
    
    # 1
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts_input)
    mesh.triangles = o3d.utility.Vector3iVector(faces_input)
    if 1:
        mesh = mesh.remove_duplicated_triangles()
        mesh = mesh.remove_degenerate_triangles()
        #mesh = mesh.remove_duplicated_vertices()
        mesh = mesh.remove_non_manifold_edges()
        #mesh = mesh.remove_unreferenced_vertices()
    verts = np.array(mesh.vertices, dtype=np.float64); faces = np.array(mesh.triangles, dtype=np.int64); 
    # mesh = o3d.geometry.HalfEdgeTriangleMesh.create_from_triangle_mesh(mesh) # faces and vertices kept the same. num_half_edges = 3 * num_faces
    # edges = []
    edges0 = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]], axis=0)
    edges1 = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]], axis=0)
    edges = np.stack([edges0, edges1], axis=-1)
    t1 = time.time(); print('Turn halfedge: ', t1-t0)
    
    # 2
    row = numpy.zeros(edges.shape[0], dtype=np.int64) # HE
    col = numpy.zeros(edges.shape[0], dtype=np.int64) # HE
    #he_idx = numpy.array(list(range(len(mesh.half_edges))))
    tri_by_he = numpy.zeros(edges.shape[0], dtype=np.int64) # HE
    heidx_in_tri = numpy.zeros(edges.shape[0], dtype=np.int64) # HE
    #next_he = numpy.zeros(len(mesh.half_edges), dtype=np.int64)
    #twin_he = numpy.zeros(len(mesh.half_edges), dtype=np.int64)
    he_by_tri = numpy.zeros([faces.shape[0], 3], dtype=np.int64) # F x 3
    he_by_vert = numpy.ones([verts.shape[0], 25], dtype=np.int64) * edges.shape[0] # N x 25

    #half_edge_mat = scipy.sparse.csc_matrix((verts.shape[0], verts.shape[0]), dtype=np.int8)
    half_edge_set = set()
    he_by_vert_flag = numpy.zeros(verts.shape[0], dtype=np.int32) # N
    count_nm = 0
    for i, half_edge in enumerate(edges):
        # v0 = half_edge[0]; v1 = half_edge[1]
        f = i % faces.shape[0]
        if half_edge[0] > half_edge[1]:
            half_edge[0], half_edge[1] = half_edge[1], half_edge[0]
        if half_edge[0] * verts.shape[0] + half_edge[1] not in half_edge_set:
            he = [half_edge[0], half_edge[1]]
        elif half_edge[1] * verts.shape[0] + half_edge[0] not in half_edge_set:
            he = [half_edge[1], half_edge[0]]
        else:
            # print("non-manifold mesh!")
            count_nm += 1
        half_edge_set.add(he[0] * verts.shape[0] + he[1])
            
        # if half_edge_mat[half_edge.vertex_indices[0], half_edge.vertex_indices[1]] == 0:
        #     he = half_edge.vertex_indices
        # else:
        #     he = [half_edge.vertex_indices[1], half_edge.vertex_indices[0]]
        # half_edge_mat[he[0], he[1]] += 1
        row[i] = he[0]
        col[i] = he[1]
        tri_by_he[i] = f
        for j in range(3):
            vert = faces[f][j]
            if vert not in he:
                he_by_tri[f][j] = i
                heidx_in_tri[i] = j
                he_by_vert[vert][he_by_vert_flag[vert]] = i
                he_by_vert_flag[vert] += 1
        #next_he[i] = half_edge.next
        #twin_he[i] = half_edge.twin
    t2 = time.time(); print('Topology: ', t2-t1)
    
    # 3
    verts = torch.from_numpy(verts).cuda()
    faces = torch.from_numpy(faces).cuda()

    all_face_v = verts[faces] # F, 3, 3
    areas = triangle_area_torch(all_face_v[..., 0, :], all_face_v[..., 1, :], all_face_v[..., 2, :]) # F
    #thetas_tri = calculate_angles_torch(all_face_v) # F, 3
    R = circumcircle_R_torch(all_face_v) # F
    cos_thetas_tri = calculate_angles_torch(all_face_v) # F, 3
    # if 0:
    cos_thetas_tri = torch.clamp(cos_thetas_tri, min=-1, max=1)
    #     print(torch.min(thetas_tri) / torch.pi * 180)
    #     print(torch.mean(thetas_tri) / torch.pi * 180)
    #     print(torch.max(thetas_tri) / torch.pi * 180)
    
    #thetas_tri2 = 2 * torch.clamp(thetas_tri, 0, torch.pi / 2) # F, 3
    # voronoi_areas_tri = torch.zeros((thetas2.shape[0] + 1, thetas2.shape[1]), dtype=thetas2.dtype).cuda()
    #sin_2theta = torch.sin(thetas_tri * 2)
    sin_2theta = 2 * cos_thetas_tri * torch.sqrt(1 - cos_thetas_tri * cos_thetas_tri)
    # voronoi_areas_tri = areas.unsqueeze(-1) * torch.ones([1, 3]).float().cuda() / 3
    #voronoi_areas_tri = 0.5 * (areas.unsqueeze(-1) - 0.5 * sin_2theta * R.unsqueeze(-1) * R.unsqueeze(-1))
    voronoi_areas_tri = torch.where(torch.min(sin_2theta, dim=-1, keepdim=True).values < 0,
                                    ((sin_2theta < 0) * 0.25 + 0.25) * areas.unsqueeze(-1),
                                    0.5 * (areas.unsqueeze(-1) - 0.5 * sin_2theta * R.unsqueeze(-1) * R.unsqueeze(-1)))
    #0.5 * (areas.unsqueeze(-1) - sin_2theta * R.unsqueeze(-1)) # F + 1, 3

    tri_by_he_torch = torch.from_numpy(tri_by_he).cuda()
    heidx_in_tri_torch = torch.from_numpy(heidx_in_tri).cuda()

    voronoi_areas = torch.zeros((edges.shape[0] + 1)).float().cuda()
    voronoi_areas[:-1] = torch.gather(voronoi_areas_tri[tri_by_he_torch], 
                                      dim=1, index=heidx_in_tri_torch.unsqueeze(-1)).squeeze(-1) # voronoi_areas_he
    voronoi_areas = voronoi_areas[torch.from_numpy(he_by_vert).cuda()]
    voronoi_areas = torch.sum(voronoi_areas, dim=-1)

    # 4
    #cos_thetas = torch.zeros((len(mesh.half_edges) + 1)).float().cuda()
    cos_thetas_he = torch.gather(cos_thetas_tri[tri_by_he_torch], 
                               dim=1, index=heidx_in_tri_torch.unsqueeze(-1)).squeeze(-1) # cos_thetas_he
    # cos_thetas = cos_thetas[torch.from_numpy(he_by_vert).cuda()]
    # cos_thetas = torch.sum(cos_thetas, dim=-1)
    cos_thetas_he = torch.clamp(cos_thetas_he, min=-1, max=1)
    cot_thetas_he = cos_thetas_he / (torch.sqrt(1 - cos_thetas_he * cos_thetas_he) + 1e-10)
    cot_thetas_he = torch.abs(cot_thetas_he)
    cot_thetas_he = torch.clamp(cot_thetas_he, min=0, max=1e3) 
    # 5
    L = torch.sparse_coo_tensor(torch.from_numpy(numpy.stack([row, col], axis=0)).cuda(),
        cot_thetas_he, (verts.shape[0], verts.shape[0]))
    L = L + L.transpose(0, 1)
    L = L.to_dense() / torch.sqrt(voronoi_areas.unsqueeze(-1) @ voronoi_areas.unsqueeze(-1).T + 1e-10) / 2
    L_diag = torch.diag(torch.max(torch.stack([torch.sum(L, dim=-1), torch.sum(L, dim=0)], dim=0), dim=0)[0])
    L = L - L_diag
    # row = np.concatenate([row, np.array(list(range(len(mesh.vertices))), dtype=np.float32)], axis=-1)
    # col = np.concatenate([col, np.array(list(range(len(mesh.vertices))), dtype=np.float32)], axis=-1)
    # cot_thetas_he = torch.cat([-cot_thetas_he, L_diag], dim=-1)
    # L = torch.sparse_coo_tensor(torch.from_numpy(numpy.stack([row, col], axis=0)).cuda(),
    #     cot_thetas_he, (len(mesh.vertices), len(mesh.vertices)))
    # L = L + L.transpose(0, 1)
    

    
    # L = 0.5 * (L + L.transpose(0, 1)) 

    LB = L @ torch.from_numpy(inpfunc).cuda()
    #L = L.detach().cpu().numpy()
    LB = LB.detach().cpu().numpy()
    L = L.float()
    t3 = time.time(); print('LBO: ', t3-t2)
    
    return LB, L

def DirectDiscreteLB(vertices,faces,inpfunc):

    LB=numpy.zeros(inpfunc.shape)
    L = numpy.zeros((vertices.shape[0],vertices.shape[0]))
    alpha = numpy.zeros((vertices.shape[0],vertices.shape[0], 2))
    Av = numpy.zeros((vertices.shape[0]))
    #beta = numpy.zeros((vertices.shape[0],vertices.shape[0]))
    vertex_surr_faces,vertex_across_edge,face_across_edge=GetTopologyAroundVertex(vertices,faces)

    #calculate the area of each triangle in the mesh
    e1=vertices[faces[:,1]]-vertices[faces[:,0]]
    e2=vertices[faces[:,2]]-vertices[faces[:,0]]
    Area=cross(e1,e2)
    Area=0.5*numpy.sqrt((Area*Area).sum(axis=1))[:,numpy.newaxis]
    # for i in range(Area.shape[0]):
    #     Area[i] = max(Area[i], 1)

    for i,pi in enumerate(vertices):
        fi=inpfunc[i]
        vsf=vertex_surr_faces[i]
        vsf=vsf[vsf>-1]
        Lsum=0.
        AP=0.
        for j,vj in enumerate(vsf):
            #t0 = time.time()
            Aj=Area[vj]
            Adj=Area[face_across_edge[i,j]]

            tj,tjp=numpy.setdiff1d(faces[vj],i)#vertices that are not 'i'
            pj =vertices[tj]
            pjp=vertices[tjp]

            fj =inpfunc[tj]
            fjp=inpfunc[tjp]

            normp  = numpy.sqrt(numpy.sum((pj-pjp)**2)) + 1e-10
            normpj = numpy.sqrt(numpy.sum((pi-pjp)**2)) + 1e-10
            normpjp = numpy.sqrt(numpy.sum((pi-pj)**2)) + 1e-10
            if alpha[i, tj, 0] == 0:
                alpha[i, tj, 0] = numpy.dot(pjp-pj,pjp-pi) / normp / normpj # cos
            else:
                alpha[i, tj, 1] = numpy.dot(pjp-pj,pjp-pi) / normp / normpj # cos
            if alpha[i, tjp, 0] == 0:
                alpha[i, tjp, 0] = numpy.dot(pj-pi,pj-pjp) / normp / normpjp # cos
            else:
                alpha[i, tjp, 1] = numpy.dot(pj-pi,pj-pjp) / normp / normpjp # cos

            cos_theta = numpy.dot(pjp-pi,pj-pi) / normpjp / normpj
            cot_theta = cos_theta / (numpy.sqrt(numpy.abs(1.0 - cos_theta**2)) + 1e-10)
            Av[i] += max(0.5* (Aj[0] - normp*normp * cot_theta / 4), 0.5 * Aj[0] )
            #t1 = time.time()
            #print(t1-t0)

        # for j in range(vertices.shape[0]):
        #     for k in range(2):
        #         alpha[i, j, k] = alpha[i, j, k] / numpy.sqrt(1 - alpha[i, j, k]**2) # cot
    alpha = alpha / (numpy.sqrt(numpy.abs(1.0 - alpha ** 2)) + 1e-10)

    #test = numpy.sum(alpha, axis=2) - numpy.sum(alpha, axis=2).transpose()

    L = numpy.sum(alpha, axis=2) / (numpy.sqrt(Av.reshape(-1, 1) @ Av.reshape(1, -1)) + 1e-10) / 2
    L = L - numpy.diag(numpy.sum(L, axis=0))
    
    
    # for i in range(vertices.shape[0]):
    #     for j in range(vertices.shape[0]):
    #         if 0:
    #             A = Av[i]
    #         else:
    #             A = numpy.sqrt(Av[i] * Av[j])
    #         L[i, j] = numpy.sum(alpha[i, j]) / 2 / A
    # for i in range(vertices.shape[0]):
    #     L[i, i] = -numpy.sum(L[i])
    return LB, L
    


def OriginDiscreteLB(vertices,faces,inpfunc):
    '''
    Laplace Beltrami operator of a function sampled at the vertices using the direct 
    discretization via gauss formula from "Convergent Discrete Laplace-Beltrami 
    Operators over Triangular Surfaces" by Guoliang Xu
    INPUTS:
    - vertices           :  coordinates of the vertices of the meshj
    - faces              :  vertex number of each face
    - inpfunc            :  a function that is sampled at the vertices of the mesh
    - vertex_surr_faces  :  indices of the faces that surround each vertex.
    - face_across_edge   :  for vertex 'i' and triangle 'j' gives face 'f' -- refer to fig 
    - vertex_across_edge :  for vertex 'i' and triangle 'j' gives vertex 'k'
    #                                i
    #                               / \
    #                              / j \
    #                             /_____\
    #                             \     /
    #                              \ f /
    #                               \ / 
    #                                k
    '''
    LB=numpy.zeros(inpfunc.shape)
    L = numpy.zeros((vertices.shape[0],vertices.shape[0]))
    alpha = numpy.ones((vertices.shape[0],vertices.shape[0], 2)) * 2
    #beta = numpy.zeros((vertices.shape[0],vertices.shape[0]))
    vertex_surr_faces,vertex_across_edge,face_across_edge=GetTopologyAroundVertex(vertices,faces)

    #calculate the area of each triangle in the mesh
    e1=vertices[faces[:,1]]-vertices[faces[:,0]]
    e2=vertices[faces[:,2]]-vertices[faces[:,0]]
    Area=numpy.cross(e1,e2)
    Area=0.5*numpy.sqrt((Area*Area).sum(axis=1))[:,numpy.newaxis]
    for i in range(Area.shape[0]):
        Area[i] = max(Area[i], 1)

    for i,pi in enumerate(vertices):
      fi=inpfunc[i]
      vsf=vertex_surr_faces[i]
      vsf=vsf[vsf>-1]
      Lsum=0.
      AP=0.
      for j,vj in enumerate(vsf):
        Aj=Area[vj]
        # if Aj < 1:
        #     Aj = 1
        Adj=Area[face_across_edge[i,j]]
        # if Adj < 1:
        #     Adj = 1

        tj,tjp=numpy.setdiff1d(faces[vj],i)#vertices that are not 'i'
        pj =vertices[tj]
        pjp=vertices[tjp]
        pdj=vertices[vertex_across_edge[i,j]]

        fj =inpfunc[tj]
        fjp=inpfunc[tjp]
        fdj=inpfunc[vertex_across_edge[i,j]]

        nj     = -(0.5/(Aj+1e-10) )*( numpy.dot(pi-pj,pj-pjp) *(pjp-pi)  + numpy.dot(pi-pjp,pjp-pj) *(pj-pi)  )
        ndj    = -(0.5/(Adj+1e-10))*( numpy.dot(pdj-pj,pj-pjp)*(pjp-pdj) + numpy.dot(pdj-pjp,pjp-pj)*(pj-pdj) )
        fbarj  = -(0.5/(Aj+1e-10) )*( numpy.dot(pi-pj,pj-pjp) *(fjp-fi)  + numpy.dot(pi-pjp,pjp-pj) *(fj-fi)  )
        fbardj = -(0.5/(Adj+1e-10))*( numpy.dot(pdj-pj,pj-pjp)*(fjp-fdj) + numpy.dot(pdj-pjp,pjp-pj)*(fj-fdj) )
        normp  = numpy.sqrt(numpy.sum((pj-pjp)**2)) + 1e-10
        normn  = numpy.sqrt(numpy.sum((nj-ndj)**2))
        
        Lsum  += (fbarj-fbardj)*normp/(normn+1e-10)
        
        # if i == 110 and j == 0:
        #     print(i)
        L[i, tjp] += (-0.5/(Aj+1e-10) * numpy.dot(pi-pj,pj-pjp) + (0.5/(Adj+1e-10)) * numpy.dot(pdj-pj,pj-pjp)) *normp/(normn+1e-10)
        L[i, tj] += (-0.5/(Aj+1e-10) * numpy.dot(pi-pjp,pjp-pj) + (0.5/(Adj+1e-10)) * numpy.dot(pdj-pjp,pjp-pj)) *normp/(normn+1e-10)
        L[i, i] += (0.5/(Aj+1e-10) * numpy.dot(pi-pj,pj-pjp) + (0.5/(Aj+1e-10)) * numpy.dot(pi-pjp,pjp-pj)) *normp/(normn+1e-10)
        L[i, vertex_across_edge[i,j]] += (-0.5/(Adj+1e-10) * numpy.dot(pdj-pj,pj-pjp) + (-0.5/(Adj+1e-10)) * numpy.dot(pdj-pjp,pjp-pj)) *normp/(normn+1e-10)

        AP    += Aj
      L[i]=L[i]/(AP+1e-10)
      LB[i]=Lsum/(AP+1e-10)
    #   Lsum_test = L[i:i+1] @ inpfunc
    #   print('')
    return LB, L