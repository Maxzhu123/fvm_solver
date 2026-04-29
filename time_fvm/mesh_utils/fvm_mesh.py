import torch
from cprint import c_print
import time


def build_sparse_gradient_matrix(combined_neigh, G_mat, dim, n_cells, n_boundaries):
    """
    Build a sparse gradient matrix for one spatial dimension using both cell neighbors and boundary facets.

    Args:
        combined_neigh Tensor: For each cell i, a 1D tensor of neighbor cell indices.
        G_mat (Tensor): For each cell i, a 2D tensor of shape [n_dims, num_total_neighbors_i].
                              The first columns correspond to cell neighbors and the remaining columns to facets.
        dim (int): The spatial dimension (0 for x, 1 for y) to build the gradient matrix.
        n_cells (int): Total number of cells.
        n_boundaries (int): Total number of boundary facets.

    Returns:
        A (torch.sparse.FloatTensor): A sparse matrix of shape [n_cells, n_cells+n_boundaries] that computes
                                      the gradient along the given dimension.
    """
    rows, cols, vals = [], [], []
    for i in range(n_cells):
        diag_val = 0.0
        # Loop over the combined neighbors.
        for k in range(combined_neigh[i].shape[0]):
            neighbor_idx = int(combined_neigh[i, k].item())
            # Use the appropriate weight from G_mat[i] (first cell_neigh.shape[0] entries correspond to cells)
            g_val = G_mat[i][dim, k]
            rows.append(i)
            cols.append(neighbor_idx)
            vals.append(g_val)
            diag_val += g_val

        # Diagonal entry: subtract the sum of all off-diagonals.
        rows.append(i)
        cols.append(i)
        vals.append(-diag_val)

    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.tensor(vals, dtype=G_mat[0].dtype)
    A = torch.sparse_coo_tensor(indices, values, (n_cells, n_cells + n_boundaries))
    return A


class FVMMesh:
    n_cells: int
    n_facets: int
    n_bc_facet: int
    # Local only for saving / plotting
    facets: torch.Tensor  # shape = (n_facet, 2)
    vertices: torch.Tensor  # shape = (n_vertices, 2)
    triangles: torch.Tensor  # shape = (n_cells, 3)
    # Used for FVM calculations
    bc_facet_mask: torch.Tensor  # shape = (n_facet)
    areas: torch.Tensor  # shape = (n_cells)
    normals: torch.Tensor  # shape = (n_facet, 2)
    lengths: torch.Tensor  # shape = (n_facet)
    centroids: torch.Tensor  # shape = (n_cells, 2)
    midpoints: torch.Tensor  # shape = (n_facet, 2)
    tri_to_facet: torch.Tensor  # shape = (n_cells, 3)
    tri_facet_signs: torch.Tensor  # shape = (n_cells, 3)
    facet_to_tri: dict[int, torch.Tensor]  # shape = {n_facet}[2]        # Mapping facet to triangle indices. Ordered [antiparallel, parallel] to facet normal.

    # Only for interior facet
    normals_main: torch.Tensor  # shape = (n_facete_main, 2)
    cell_grad_stuff: tuple # Stuff needed to calculate gradient on a cell
    facet_to_tri_main: torch.Tensor # shape = (n_facet_main, 2)              # Mapping facet to triangle indices for non boundary facet

    # Only for boundary facets
    facet_to_tri_bc: torch.Tensor # shape = (n_facet_bc, 1)              # Mapping facet to triangle indices for boundary facet
    normals_bc: torch.Tensor  # shape = (n_facet_bc, 2)

    def __init__(self, vertices, cells, facets, bc_facet_mask, device="cuda"):
        self.vertices = vertices
        self.triangles = cells
        self.facets = facets
        self.bc_facet_mask = bc_facet_mask
        self.device = device

        self.n_cells = cells.shape[0]
        self.n_facets = facets.shape[0]
        self.n_bc_facet = bc_facet_mask.sum().item()
        assert facets.shape[0] == bc_facet_mask.shape[0], f'Different number of facets from bc facet mask {facets.shape = }, {bc_facet_mask.shape = }'

        c_print(f'Computing mesh properties', color="bright_magenta")
        self._compute_facet_props(vertices, cells, facets)

    def _grad_weighting(self, tri_to_facet, facet_to_tri_ord, centroids, midpoints, normals):
        """ Use least squares formula to compute gradient weighting.
            grad(u) = A^-1 * b
            A = sum_i (d_i d_i^T)
            b = sum_i d_i (u_i - u_c)
        """

        bound_facet_idxs = torch.nonzero(self.bc_facet_mask, as_tuple=False).flatten()
        global_to_local = {int(global_idx): local_idx for local_idx, global_idx in enumerate(bound_facet_idxs)}
        facet_to_tri_comb = []
        for e, cell in facet_to_tri_ord.items():
            if len(cell) == 2:
                facet_to_tri_comb.append(cell)
            else:
                bc_facet_id = global_to_local[e] + self.n_cells
                bc_facet_id = torch.tensor([bc_facet_id, bc_facet_id])

                facet_to_tri_comb.append(bc_facet_id)
        facet_to_tri_comb = torch.stack(facet_to_tri_comb)

        combined_neigh, neigh_cents, combined_bc = [], [], []
        for cell_id, facets in enumerate(tri_to_facet):  # Must keep this order. Neighbor id: torch.cat([Us, Us_bc_facet])
            # Get neighboring cells
            neighbors, centers, is_bc = [], [], []
            for e in facets:
                e = e.item()
                if len(facet_to_tri_ord[e]) == 2:
                    """ Interior facet"""
                    is_bc.append(False)
                    tris = facet_to_tri_ord[e]
                    neigh_cell = tris[tris != cell_id]
                    centers.append(centroids[neigh_cell])  # [1, 2]
                    neighbors.append(neigh_cell.item())
                else:
                    """ Boundary facet """
                    is_bc.append(True)
                    midpoint = midpoints[e].unsqueeze(0)
                    centers.append(midpoint)
                    glob_facet_idx = global_to_local[e]
                    neighbors.append(glob_facet_idx + self.n_cells)

            combined_neigh.append(torch.tensor(neighbors))
            neigh_cents.append(torch.cat(centers))  # [3, 2]
            combined_bc.append(torch.tensor(is_bc)) # [3]
        combined_neigh = torch.stack(combined_neigh).int()
        neigh_cents = torch.stack(neigh_cents)  # [n_cells, 3, 2]
        combined_bc = torch.stack(combined_bc).bool()  # [n_cells, 3]

        # --- Compute gradient vectors in batch ---
        # For each cell, compute the displacement vectors d_i = (neighbor center - cell center)
        center_expanded = centroids.unsqueeze(1)  # shape: [n_cells, 1, 2]
        d = neigh_cents - center_expanded  # shape: [n_cells, 3, 2]
        # Compute weights per neighbor: w_i = 1 / norm(d_i) ** k
        w = 1 / torch.norm(d.double(), dim=2) ** 0.25 # shape: [n_cells, 3]
        w[combined_bc] *= 7.5
        w2 = w ** 2  # shape: [n_cells, 3]
        # Compute A = dᵀ @ diag(w²) @ d for each cell.
        # dᵀ has shape [n_cells, 2, 3] and d * w2.unsqueeze(-1) scales each 2D neighbor vector.
        dT = d.transpose(1, 2).double()  # shape: [n_cells, 2, 3]
        A = torch.bmm(dT, d * w2.unsqueeze(-1))  # shape: [n_cells, 2, 2]
        # Invert A for each cell.
        A_inv = torch.inverse(A.double())  # shape: [n_cells, 2, 2]
        # Finally, compute the gradient matrix as A_inv @ dᵀ @ diag(w²)
        # Multiply dᵀ by w2 along the neighbor dimension:
        A_inv_di_T = torch.bmm(A_inv, dT * w2.unsqueeze(1)).float()  # shape: [n_cells, 2, 3]

        G_mats = []
        for i in range(2):
            G_mat = build_sparse_gradient_matrix(combined_neigh, A_inv_di_T, i, self.n_cells, self.n_bc_facet)
            G_mats.append(G_mat)

        # Get displacement between cells with facet indexing. In direction of right to left
        cell_disps, facet_dist_bc = [], []
        for e, cells in facet_to_tri_ord.items():
            if cells.shape[0] == 1:
                # BC cell / facet: Distance from centroid to facet.
                n_hat = normals[e] / torch.norm(normals[e], dim=-1, keepdim=True)
                f = midpoints[e]
                p = centroids[cells[0]]
                disp = n_hat * torch.dot(f - p, n_hat)
                sign = torch.sign(torch.dot(f - p, n_hat))
                dist = torch.norm(disp) * sign
                facet_dist_bc.append(dist)
            else:
                # Main cell / facet: Distance between centroids
                d = centroids[cells[1]] - centroids[cells[0]]
                cell_disps.append(d)
        cell_disps = torch.stack(cell_disps)
        facet_dist_bc = torch.stack(facet_dist_bc)

        return cell_disps, facet_dist_bc, G_mats, combined_neigh, facet_to_tri_comb

    def _compute_facet_props(self, vertices, triangles, facets):
        # Compute facet normals and lengths
        facet_vertex = vertices[facets]
        facet_vectors = facet_vertex[:, 1] - facet_vertex[:, 0]        # Ordering is used as facet index from here.
        normals = torch.stack([facet_vectors[:, 1], -facet_vectors[:, 0]], dim=1)
        midpoints = torch.mean(facet_vertex, dim=1)      # shape = [n_facet, 2]
        self.facet_vertex = facet_vertex                  # shape = [n_facet, 2, 2]
        self.normals = normals                          # shape = [n_facet, 2]
        self.midpoints = midpoints                      # shape = [n_facet, 2]

        # Triangle area and centroid
        tri_points = vertices[triangles]
        self.areas = self._tri_area(tri_points)
        self.centroids = torch.mean(tri_points, dim=1)  # shape = [n_cells, 2]

        # Compute mapping of facet to triangles
        tri_to_facet = self._get_tri_facets(triangles, facets) # shape = [n_cells, 3]
        self.tri_to_facet = tri_to_facet
        unique_facets, _ = torch.unique(tri_to_facet, sorted=True, return_inverse=True)
        _facet_to_tri, tri_facet_idxs = {}, {}
        for facet in unique_facets:
            pos = (facet == tri_to_facet).nonzero()
            _facet_to_tri[facet.item()] = pos[:, 0]
            tri_facet_idxs[facet.item()] = pos[:, 1]

        # Sort triangle in order of facet signed direction. ORDER: [-, +], so cell on right comes first.
        self.tri_facet_signs, facet_to_tri, cent_to_facet_disp = self._tri_facet_sign(self.centroids, midpoints, tri_to_facet, self.normals, _facet_to_tri, tri_facet_idxs)
        self.facet_to_tri = facet_to_tri

        # Split tensors into facet and main
        normals_main, facet_to_tri_main = [], []
        facet_to_tri_bc, normals_bc = [], []
        for e_idx, e_bc in enumerate(self.bc_facet_mask):
            if e_bc:
                # Precompute tensors for boundary facets
                facet_to_tri_bc.append(facet_to_tri[e_idx])
                normals_bc.append(normals[e_idx])
            else:
                # Precompute tensors for interior facets
                normals_main.append(normals[e_idx])
                facet_to_tri_main.append(facet_to_tri[e_idx])

        self.normals_main = torch.stack(normals_main)
        self.facet_to_tri_main = torch.stack(facet_to_tri_main)
        self.cent_to_facet_disp = cent_to_facet_disp
        self.facet_to_tri_bc = torch.stack(facet_to_tri_bc).squeeze()
        self.normals_bc = torch.stack(normals_bc)

        # Compute grad weighting
        self.cell_grad_stuff = self._grad_weighting(tri_to_facet, facet_to_tri, self.centroids, midpoints, normals)

    def _tri_area(self, vertices):
        """ vertices.shape = (n_cells, 3, 2) """
        a, b, c = vertices[:, 0], vertices[:, 1], vertices[:, 2]
        # Compute the vectors for each triangle
        ab = b - a  # shape [n, 2]
        ac = c - a  # shape [n, 2]
        # Compute the 2D cross product (determinant) for each triangle
        cross = ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]  # shape [n]
        # Triangle area is half the absolute value of the cross product
        area = 0.5 * torch.abs(cross)

        return area

    def _tri_facet_sign(self, centroids, midpoints, tri_to_facet, normals, facet_to_tri, tri_facet_idxs):
        """ Compute which triangle is on the left and right of each facet.
            For ordering, cell on Left comes first, then right
            Signs: 1 if on left, -1 if on right.
        """
        # signs = []
        # cent_to_facet_disp = []
        # for tri_idx, (facet, center) in enumerate(zip(tri_to_facet, centroids)):
        #     midpoint = midpoints[facet]      # shape = [3, 2]
        #     normal = normals[facet]          # shape = [3, 2]
        #
        #     p_diff = midpoint - center      # shape = [3, 2]
        #
        #     # Or normals dot (midpt-center)
        #     norm_hat = normal / torch.norm(normal, dim=-1, keepdim=True)
        #     dist_dot = torch.sum(norm_hat * p_diff, dim=-1)
        #
        #     sign_dot = torch.sign(dist_dot)
        #     signs.append(sign_dot)
        #
        #     # Compute shortest vector from centroid to line.
        #     r = p_diff
        #     cent_to_facet_disp.append(r)
        #
        # signs = torch.stack(signs).long()       # shape = [n_cells, 3]
        # cent_to_facet_disp = torch.stack(cent_to_facet_disp)  # shape = [n_cells, 3, 2]

        midpoints_tri = midpoints[tri_to_facet]  # shape: [n_cells, 3, 2]
        normals_tri = normals[tri_to_facet]  # shape: [n_cells, 3, 2]
        # Compute the difference between each facet midpoint and the centroid.
        p_diff = midpoints_tri - centroids.unsqueeze(1)  # shape: [n_cells, 3, 2]
        # Normalize the normals along the last dimension.
        norms = torch.norm(normals_tri, dim=-1, keepdim=True)  # shape: [n_cells, 3, 1]
        norm_hat = normals_tri / norms  # shape: [n_cells, 3, 2]
        # Compute the dot product and then its sign.
        dist_dot = torch.sum(norm_hat * p_diff, dim=-1)  # shape: [n_cells, 3]
        signs = torch.sign(dist_dot).long()  # shape: [n_cells, 3]
        # The displacement vectors are simply p_diff.
        cent_to_facet_disp = p_diff  # shape: [n_cells, 3, 2]

        facet_to_tri_ordered = {}
        p_m, m_p = torch.tensor([1, -1]), torch.tensor([-1, 1])
        for facet in sorted(facet_to_tri.keys()):
            tri_idx = facet_to_tri[facet]
            tri_facet = tri_facet_idxs[facet]

            order = signs[tri_idx, tri_facet]

            # Boundary facets only have 1 triangle
            if order.shape[0] == 1:
                assert self.bc_facet_mask[facet] == True, "Inconsistent boundary bug"
            else:
                if torch.all(order == m_p):
                    tri_idx = torch.flip(tri_idx, dims=[0])
            facet_to_tri_ordered[facet] = tri_idx

        return signs, facet_to_tri_ordered, cent_to_facet_disp

    def _get_tri_facets(self, triangles, facets):
        """
            Compute which facets belong to each triangle
            triangles.shape = (n_cells, 3)
            facets.shape = (n_facet, 2)
        """
        # 1) Normalize each facet (sort nodes in ascending order).
        # -------------------------------------------------------
        # facets_sorted will be shape [m, 2] with each row sorted.
        facets_sorted, _ = facets.sort(dim=1)

        # 2) Build a lookup: (nodeA, nodeB) -> facet_index
        # -----------------------------------------------
        facet_dict = {}
        for idx, e in enumerate(facets_sorted):
            # Make a tuple key (nodeA, nodeB)
            key = (e[0].item(), e[1].item())
            facet_dict[key] = idx

        # 3) For each triangle, find the 3 facets
        # --------------------------------------
        # We'll create a result tensor of shape [num_triangles, 3],
        # each row will store the indices of the 3 facets of that triangle.

        tri_to_facet = []
        for tri in triangles:
            # Extract triangle nodes (v0, v1, v2)
            v0 = tri[0].item()
            v1 = tri[1].item()
            v2 = tri[2].item()

            # Sort each pair so we can look it up in the facet_dict
            e1 = tuple(sorted((v0, v1)))
            e2 = tuple(sorted((v1, v2)))
            e3 = tuple(sorted((v2, v0)))

            # Get the facet indices
            facet_indices = [facet_dict[e1], facet_dict[e2], facet_dict[e3]]
            tri_to_facet.append(facet_indices)

        tri_to_facet = torch.tensor(tri_to_facet)
        return tri_to_facet
