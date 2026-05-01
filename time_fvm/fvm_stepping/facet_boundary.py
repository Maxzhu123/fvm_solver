from typing import TYPE_CHECKING
import torch

from time_fvm.config_fvm import ConfigFVM
from time_fvm.sparse_utils import to_csr
from time_fvm.fvm_stepping.boundary_process import BC
if TYPE_CHECKING:
    from torch import Tensor
    from time_fvm.fvm_stepping.facet_process import FacetFlux
    from time_fvm.fvm_equation import PhysicalSetup


class BoundarySetter:
    """ Non-orthogonal correction for Neumann BCs."""
    n_comp: int
    n_edges_bc: int

    grad_comps: torch.Tensor          # shape = [n_neum_edges, 2, 1]
    where_neum: tuple[torch.Tensor]      # shape = [2][n_neum_edges, 2]

    # Matrices for general face values
    A_bc: torch.Tensor
    b_bc: torch.Tensor

    # Farfield boundary condition
    use_farfield: bool
    farfield_calc: BC
    exit_cell2edge: torch.Tensor    # shape = (n_cells, 2)  # Exit edge for each cell

    # Inlet boundary condition
    use_inlet: bool
    inlet_calc: BC
    inlet_cell2edge: torch.Tensor    # shape = (n_cells, 2)  # Inlet edge for each cell

    def __init__(self, E_props: FacetFlux, phy_setup: PhysicalSetup):
        self.phy_setup = phy_setup

        self.use_farfield, self.use_inlet = False, False

        self.n_comp = E_props.n_comp
        self.n_edges_bc = E_props.n_facets_bc
        mesh = E_props.mesh
        cell_to_facet = mesh.cell_to_facet.view(-1, 3)

        # Flatten out all Neumann BCs and index according to order where_neum_all[0]
        neum_mask_all = torch.zeros_like(mesh.bc_facet_mask)
        neum_mask_all = neum_mask_all.unsqueeze(-1).repeat(1, 4)
        neum_mask_all[mesh.bc_facet_mask] = E_props.neumann_mask
        where_neum_all = torch.where(neum_mask_all)
        where_neum = {'edge': where_neum_all[0], 'comp': where_neum_all[1]}  # shape = [n_neum_edges, 2]
        # Mapping from boundary id to boundary edge id
        self.where_neum = torch.where(E_props.neumann_mask)

        # Mapping from boundary edge to cell
        bc_edge_to_tri = torch.zeros_like(mesh.bc_facet_mask).long()
        bc_edge_to_tri[mesh.bc_facet_mask] = mesh.facet_to_cell_bc

        # Cells corresponding to Neumann BC
        self.neum_cells = bc_edge_to_tri[where_neum_all[0]]  # shape = [n_neum_edges], which cells have neuman BCs
        where_neum['cells'] = self.neum_cells
        # Edge within cell corresponding to Neumann BC
        tri_edge_num = (where_neum['edge'].unsqueeze(-1).repeat(1, 3) == cell_to_facet[where_neum['cells']])
        tri_edge_id = torch.where(tri_edge_num)[1]
        where_neum['tri_edge_id'] = tri_edge_id

        # Which component of gradient is needed for Neumann BC
        self.grad_comps = where_neum['comp'].unsqueeze(1).repeat(1, 2).unsqueeze(2)     # shape = [n_neum_edges, 2, 1]
        # Normal vector of edges
        n_hats = mesh.normals_hat.squeeze()[where_neum['edge']]  # shape = [n_neum_edges, 2]
        # Displacement from centroid to edge
        cent_to_edge = mesh.cent_to_facet_disp[where_neum['cells']].squeeze()  # shape = [n_neum_edge, 3, 2]
        r = cent_to_edge[torch.arange(cent_to_edge.shape[0]), where_neum['tri_edge_id']]  # shape = [n_neum_edge, 2]
        # Normal component of r
        d = n_hats * (r * n_hats).sum(dim=1, keepdim=True)  # shape = [n_neum_edge, 2]
        # Parallel component of r
        self.l = r - d

        A_bc, b_bc = self._build_spm_face_vals(E_props)
        self.A_bc, self.b_bc = A_bc, b_bc

    def set_face_values(self, Us, cell_grads=None, dt=None):
        """Compute and return boundary face values from cell values.

        Uses the precomputed sparse operator to map flattened cell values to
        boundary face values, then applies non-orthogonal correction and
        farfield adjustments if enabled.
        """
        Us_flat = Us.flatten()
        # Final U_face in flattened form.a
        U_face_flat = torch.mv(self.A_bc, Us_flat) + self.b_bc      # shape = [n_edges_bc * n_comp]
        # Reshape back to (n_edges_bc, n_comp)
        U_face = U_face_flat.view(self.n_edges_bc, self.n_comp)

        if cell_grads is not None:
            self._non_orthogonal_correction(U_face, cell_grads)

        if self.use_farfield:
            self.farfield_calc.set_bc_U_face(U_face, Us[self.exit_cell2edge], dt)

        if self.use_inlet:
            self.inlet_calc.set_bc_U_face(U_face, Us[self.inlet_cell2edge], dt)

        return U_face

    def _non_orthogonal_correction(self, U_face, cell_grads):
        """
        Use previous gradient for non-orthogonal correction.

        U_face.shape = [n_bc_faces, n_comp]
        cell_grads.shape = [n_cells, 2, n_comp]

        r = centroid to midpoint.
        d = normal component of r
        U_f = U_0 + d * dUdn + (r-d) grad(U)
        """
        grads = torch.gather(cell_grads[self.neum_cells], 2, self.grad_comps).squeeze()  # shape = [n_neum_edge, 2]

        dU = (grads * self.l).sum(dim=1)  # shape = [n_neum_edge]
        U_face[self.where_neum[0], self.where_neum[1]] += dU

    def _build_spm_face_vals(self, E_props: FacetFlux):
        """ Compute bc edge values using sparse matrix multiplication. """

        device = E_props.device
        n_bc = E_props.n_facets_bc  # number of boundary edges
        n_comp = E_props.n_comp
        n_cells = E_props.n_cells

        # Total number of flattened BC rows.
        N = n_bc * n_comp

        # Create flattened indices for the boundary rows and the corresponding component.
        # Each boundary edge gives rise to n_comp rows.
        bc_rows = torch.arange(n_bc, device=device).unsqueeze(1).expand(n_bc, n_comp).reshape(-1)
        comp_idx = torch.arange(n_comp, device=device).unsqueeze(0).expand(n_bc, n_comp).reshape(-1)

        # Reshape the condition masks to a flat vector of length N.
        dirich_mask = E_props.dirich_mask.reshape(-1)  # For Dirichlet conditions.
        neum_mask = E_props.neumann_mask.reshape(-1)  # For Neumann conditions.

        # --- Build sparse matrix A ---
        # For Neumann entries, we want to extract the cell value from Us.
        # For each Neumann row, the corresponding column in Us (flattened) is given by:
        #   col = self.edge_to_tri_bc[ edge_index ] * n_comp + component
        neum_indices = torch.nonzero(neum_mask, as_tuple=False).squeeze(1)  # indices where Neumann is True.
        A_rows = neum_indices
        # bc_rows[neum_indices] gives the corresponding boundary edge for each flattened row.
        A_cols = E_props.mesh.facet_to_cell_bc[bc_rows[neum_indices]] * n_comp + comp_idx[neum_indices]
        A_vals = torch.ones_like(A_rows, dtype=torch.float32, device=device)

        size_A = (N, n_cells * n_comp)
        A_bc = torch.sparse_coo_tensor(torch.stack([A_rows, A_cols], dim=0), A_vals, size=size_A)# .coalesce().to_sparse_csr()
        A_bc = to_csr(A_bc, device=device)
        # Build the offset vector b.
        b_bc = torch.empty(N, device=device, dtype=torch.float32)
        # For Dirichlet entries, the prescribed value should override any extracted value.
        b_bc[dirich_mask] = E_props.dirich_val
        # For Neumann entries, add the offset computed from the edge distance.
        # Here, we select the proper component value from self.neumann_val using comp_idx.
        b_bc[neum_mask] = E_props.neumann_val[comp_idx[neum_mask]] * E_props.mesh.facet_dists_bc.flatten()[neum_mask]

        return A_bc, b_bc

    def init_farfield(self, cfg: ConfigFVM, farfield_mask, exit_cell2edge, farfield_normals):
        self.use_farfield = True
        self.farfield_calc = BC(self.phy_setup, cfg, cfg.exit_cfg, farfield_mask, farfield_normals)
        self.exit_cell2edge = exit_cell2edge

    def init_inlet(self, cfg: ConfigFVM, inlet_mask, inlet_cell2edge, inlet_normals):
        self.use_inlet = True
        self.inlet_calc = BC(self.phy_setup, cfg, cfg.inlet_cfg, inlet_mask, inlet_normals)
        self.inlet_cell2edge = inlet_cell2edge







