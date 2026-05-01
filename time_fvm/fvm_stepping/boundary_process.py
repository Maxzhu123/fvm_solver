from typing import TYPE_CHECKING
import torch

from time_fvm.config_fvm import ConfigFVM, BCMode, ConfigBC
if TYPE_CHECKING:
    from torch import Tensor
    from time_fvm.fvm_equation import PhysicalSetup


class BC:
    bc_normals: Tensor     # shape = [n_inlet_edges, 2]
    bc_tangents: Tensor

    def __init__(self, phy_setup: PhysicalSetup, cfg: ConfigFVM, bc_cfg: ConfigBC, bc_mask: Tensor, bc_normals: Tensor):
        """
        Generalised boundary condition.
        args:
            phy_setup: PhysicalSetup, contains EOS and other physics-specific functions.
            cfg: ConfigFVM, contains general configuration parameters.
            bc_cfg: ConfigBC, contains boundary condition specific parameters and mode.
            bc_mask: Tensor, shape [n_bc], which edges are set for this BC.
            bc_normals: Tensor, shape [n_bc], Outward pointing normals for boundary edges.
        """

        self.cfg = cfg
        self.device = cfg.device
        self.phy_setup = phy_setup

        # Boundary geometry
        self.bc_mask = bc_mask
        self.bc_normals = bc_normals
        self.bc_tangents = torch.stack((-bc_normals[:, 1], bc_normals[:, 0]), dim=1)
        self.bc_idx = torch.where(bc_mask)[0]

        # Shared boundary parameters
        self.T_inf = torch.tensor(bc_cfg.T_inf, device=self.device)
        self.rho_inf = torch.tensor(bc_cfg.rho_inf, device=self.device)
        self.v_n_inf = - torch.tensor(bc_cfg.v_n_inf, device=self.device)  # Since inlet edge points outwards
        self.v_t_inf = torch.tensor(bc_cfg.v_t_inf, device=self.device)
        p_inf = self.phy_setup.eos_P(self.rho_inf, self.T_inf)
        a_inf = self.phy_setup.eos_c(self.rho_inf, self.T_inf)

        # Set stagnation condition to be similar to natural parameters.
        match bc_cfg.mode:
            # Characteristic boundary condition
            case BCMode.Characteristic:
                self.set_bc_U_face = self.BC_characteristic

                self.p_inf = p_inf
                # We use this a lot
                self.ones = torch.ones(self.bc_mask.sum(), device=cfg.device)  # shape = [n_inlet]

            # Isentropic boundary condition
            case BCMode.Isentropic:
                self.set_bc_U_face = self.BC_isentropic

                self.gamma = phy_setup.gamma
                # Get stagnation conditions
                M_inf = self.v_n_inf / a_inf
                self.p_0 = p_inf * (1 + (self.gamma - 1) / 2 * M_inf ** 2) ** (self.gamma / (self.gamma - 1))
                self.T_0 = self.T_inf * (1 + (self.gamma - 1) / 2 * M_inf ** 2)

            # Farfield boundary condition
            case BCMode.Farfield:
                self.set_bc_U_face = self.BC_farfield

                self.R = phy_setup.R
                self.gamma = phy_setup.gamma
                self.R_far = self.v_n_inf - 2 * a_inf / (self.gamma - 1)

            # Farfield blended boundary condition
            case BCMode.FarfieldBlended:
                self.set_bc_U_face = self.BC_farfield_blended

                self.R = phy_setup.R
                self.gamma = phy_setup.gamma
                self.R_m_far = self.v_n_inf - 2 * a_inf / (self.gamma - 1)
                self.R_p_far = self.v_n_inf + 2 * a_inf / (self.gamma - 1)
                self.S_far = p_inf / (self.rho_inf ** self.gamma)

            case _ :
                raise NotImplementedError(f"Unknown Inlet Mode {bc_cfg.mode = }")

    def _construct_R(self, c: Tensor, rho: Tensor, ones=None):
        """ Construct R and R^-1 for the characteristic decomposition.
            A = [   [u, rho,        0       ],
                    [0, u,          1/rho   ],
                    [0, rho*c**2,   u       ]]
            Decompose A = R D R^-1

            c: shape = [n_bc]
            rho: shape = [n_bc]
            R: shape = [n_bc, 3, 3]
            R^-1: shape = [n_bc, 3, 3]
        """
        # Use cached ones matrix if needed.
        if ones is None:
            ones = self.ones

        zeros = torch.zeros_like(ones)
        c_sq = c ** 2

        R = torch.stack([
            torch.stack([ones, ones, ones], dim=1),
            torch.stack([-c / rho, zeros, c / rho], dim=1),
            torch.stack([c_sq, zeros, c_sq], dim=1),
        ], dim=1)  # shape = [n_inlet, 3, 3]

        R_inv = torch.stack([
            torch.stack([zeros, -rho / (2 * c), 1 / (2 * c_sq)], dim=1),
            torch.stack([ones, zeros, -1 / c_sq], dim=1),
            torch.stack([zeros, rho / (2 * c), 1 / (2 * c_sq)], dim=1),
        ], dim=1)

        return R, R_inv

    def _split_Vs(self, Vs):
        """ Split velocity into normal and tangential components.
            Vs: shape = [n_bc, 2]
            Return V_n, V_t: shape = [n_bc], [n_bc]
        """
        n = self.bc_normals
        t = self.bc_tangents

        v_n = (Vs * n).sum(dim=1)
        v_t = (Vs * t).sum(dim=1)

        return v_n, v_t

    def _recombine_Vs(self, v_n, v_t):
        """ Combine normal and tangential component of Vs back into x-y components."
            V_n: shape = [n_bc]
            V_t: shape = [n_bc]
            Return V_x, v_y: shape = [n_bc]
        """
        n = self.bc_normals
        t = self.bc_tangents

        V = v_n.unsqueeze(-1) * n + v_t.unsqueeze(-1) * t
        v_x, v_y = V[:, 0], V[:, 1]

        return v_x, v_y, V

    def _gating(self, v_n_int, c_int):
        """ Gating for forward and backward characteristics.
            v_n_int.shape = [n_bc]
            return.shape = [n_bc, 3]
        """
        lambda_vals = torch.stack([v_n_int - c_int, v_n_int, v_n_int + c_int], dim=-1)  # shape = [n_bc, 3]
        c = c_int.mean()
        lambda_scaled = (10/c) * lambda_vals                # shape = [n_bc, 3]
        gating = 0.5 * (1 - torch.tanh(lambda_scaled))

        return gating

    def set_bc_U_face(self, U_face, Us_bc_cells, dt):
        """ Set U_face.
            U_face: shape = [n_edges, n_comp], all boundary edges. Set value in place, given by mask.
            Us_bc_cells: shape = [n_bc_edges, n_comp], cell values at boundary edges.
        """
        raise NotImplementedError

    # ------------------------------- Specific BC implementations -------------------------------
    def BC_characteristic(self, U_face, Us_bc_cells, dt):
        """ Characteristic BC.
            Use W = (rho, v_n, p) -> dW/dt + div(f(W)) = 0
            Linearize using W = W_int + delta W
            Diagonalise equations
            Then solve for delta W_b by continuing characteristics from the left and right side.
            Tangential velocity is interpolated as well.

            U_face.shape = [n_bc_edges, n_comp], all boundary edges. Set value in place, given by mask.
            Us_bc_cells.shape = [n_inlet, n_comp], cell values at boundary edges.
        """
        # 1) Interior properties
        v_n_int, v_t_int = self._split_Vs(Us_bc_cells[:, :2])           # shape = [n_inlet]
        rho_int = Us_bc_cells[:, 2]  # shape = [n_inlet]
        T_int = Us_bc_cells[:, 3]  # shape = [n_inlet]
        # Convert basis to W = (rho, v_n, p)
        p_int = self.phy_setup.eos_P(rho_int, T_int)
        W_int = torch.stack([rho_int, v_n_int, p_int], dim=-1)  # shape = [n_inlet, 3]
        # Interior speed of sound
        c_int = self.phy_setup.eos_c(rho_int, T_int)

        # 2) Exterior properties
        rho_inf = self.rho_inf
        v_n_inf = self.v_n_inf
        p_inf = self.p_inf
        W_inf = torch.stack([rho_inf, v_n_inf, p_inf]) # shape = [3]

        # 3) Get transformation from dW basis to orthogonal dChi basis, decompose A = R D R^-1,
        R, R_inv = self._construct_R(c_int, rho_int)

        # 4) Compute dW and project into dChi space:
        dW_inf = W_inf - W_int
        # dChi = R^-1 dW to diagonalise the system into forward and backward components
        dChi_inf = (R_inv @ dW_inf.unsqueeze(-1)).squeeze(-1)           # shape = [n_inlet, 3]

        # 5) Filter incoming and outgoing components. Transition smoothly on scale O(c)
        gating = self._gating(v_n_int, c_int)
        dChi_b = gating * dChi_inf        # shape = [n_inlet, 3]
        # 5.1) Tangential velocity is also interpolated using the gating
        v_t_int = gating[:, 1] * v_t_int + (1-gating[:, 1]) * self.v_t_inf

        # 6) Convert back to dW = R dChi,
        dW_b = (R @ dChi_b.unsqueeze(-1)).squeeze()                         # shape = [n_inlet, 3]
        W_b = W_int + dW_b

        # 7) Convert back to primatives
        rho_b = W_b[:, 0]
        # Keep tangential velocity from interior.
        v_n_b = W_b[:, 1]
        v_x_b, v_y_b, _ = self._recombine_Vs(v_n_b, v_t_int)
        # Convert pressure back into temperature
        p_b = W_b[:, 2]
        T_b = self.phy_setup.eos_T(rho_b, p_b)

        # Inplace update for U_face.
        U_face[self.bc_mask] = torch.stack([v_x_b, v_y_b, rho_b, T_b], dim=-1)

    def BC_isentropic(self, U_face, Us_bc_cells, dt):
        """ Subsonic inlet boundary conditions based on isentropic flow relations (Adiabatic).
            Ideal gas only.
        Given stagnation (total) conditions p_0 and T_0, we compute the boundary state by combining interior flow.
        1. Pressure ratio (isentropic relation):
           p_0/p = (1 + (γ-1)/2 * M²)^(γ/(γ-1))
        2. Temperature ratio (isentropic relation):
           T_0/T = 1 + (γ-1)/2 * M²
        3. Speed of sound:
           a = √(γ * R * T)
        4. Mach number definition:
           M = V/a

        - p and T are computed from interior cell values, p=p_int, T=T_int
        - Backflow prevention: p_ratio is clamped to be ≥ 1 + ε to ensure M_bc ≥ 0
        - The tangential velocity component is preserved from the interior flow

        Parameters:
        U_face : torch.Tensor, shape [n_edges_bc, n_comp]
            Boundary face values to be modified in place
        Us_bc_cells : torch.Tensor, shape [n_inlet_edges, n_comp]
            Interior cell values adjacent to inlet edges [V_x, V_y, ρ, T]
        dt : float
            Time step (unused, kept for interface compatibility)
        """

        V_int = Us_bc_cells[:, [0, 1]]  # shape = [n_inlet_edge, 2]
        rho_int = Us_bc_cells[:, 2]  # shape = [n_inlet_edge]
        T_int = Us_bc_cells[:, 3]  # shape = [n_inlet_edge]

        # Helper basis uses outward normals; inlet formulas below use inward normals.
        _, v_t_int = self._split_Vs(V_int)

        # Interior values
        p_int = self.phy_setup.eos_P(rho_int, T_int)
        p_ratio = self.p_0 / p_int
        p_ratio.clamp_(min=1 + 1e-7)  # Ensure no backflow.

        # Boundary values
        M_bc = torch.sqrt((2/(self.gamma - 1)) * (p_ratio ** ((self.gamma - 1)/self.gamma) - 1))
        T_bc = self.T_0 / (1 + (self.gamma - 1)/2 * M_bc ** 2)
        c_bc = self.phy_setup.eos_c(rho_int, T_bc)

        # Inlet velocity
        v_n_bc = - M_bc * c_bc
        v_x, v_y, _ = self._recombine_Vs(v_n_bc, v_t_int)

        rho_bc = self.phy_setup.eos_rho(p_int, T_bc)

        U_face_farfield = torch.stack([v_x, v_y, rho_bc, T_bc], dim=-1)
        U_face[self.bc_mask] = U_face_farfield

    def BC_farfield(self, U_face, Us_bc_cells, dt):
        """ Compressible farfield:
                R+ = u + 2a/(gamma - 1)
                R- = u - 2a/(gamma - 1)
                S = P / rho^gamma
            On exit, we have:
                R+ = R+_int
                R- = R-_inf
                S = S_int
        """
        gm1 = self.gamma - 1

        V = Us_bc_cells[:, [0, 1]]                                    # shape = [n_ff_edge, 2]
        rho_int = Us_bc_cells[:, 2]                                   # shape = [n_ff_edge]
        T_int = Us_bc_cells[:, 3]                                     # shape = [n_ff_edge]

        # Parallel and tangential velocity
        V_t, V_n = self._split_Vs(V)

        # Incoming farfield:
        R_m = self.R_far #self.v_far - 2 * self.a_far / gm1

        # Outgoing (extrapolate from internal) :
        a_int = torch.sqrt(self.gamma * self.R * T_int)
        R_p = V_n + 2 * a_int / gm1       #  R+ = R+_int = V_n + 2 * a_in / (gamma-1)
        S_p = self.R * T_int * rho_int ** (-gm1)

        # Boundary values: a_bc = (gamma-1)/4 * (R+ - R-)
        V_n_bc = 1/2 * (R_p + R_m)
        a_bc_2 = (gm1/4 * (R_p - R_m)) ** 2
        rho_bc = (a_bc_2 / (self.gamma * S_p)) ** (1 / gm1)
        T_bc = a_bc_2 / (self.gamma * self.R)

        _, _, V_bc = self._recombine_Vs(V_n_bc, V_t)

        U_face_farfield = torch.cat([V_bc, rho_bc.unsqueeze(-1), T_bc.unsqueeze(-1)], dim=-1)
        U_face[self.bc_mask] = U_face_farfield

    def BC_farfield_blended(self, U_face, Us_bc_cells, dt):
        """Set farfield boundary conditions using blended characteristic approach. Good if flow changes direction.

        This function implements a smooth, blended farfield boundary condition that
        automatically transitions between inflow and outflow using Riemann invariants
        and entropy.

        Theory - Riemann Invariants:
        ----------------------------
        1. Left-running Riemann invariant (characteristic speed: u - a):
           R⁻ = u - 2a/(γ-1)
        2. Entropy invariant (characteristic speed: u):
           S = p/ρ^γ = RT/ρ^(γ-1)
           constant along streamlines for isentropic flow
        3. Right-running Riemann invariant (characteristic speed: u + a):
           R⁺ = u + 2a/(γ-1)

        Blending Strategy:
        ------------------
        The method smoothly interpolates each invariant based on the sign and magnitude  of λ = u ± a:
        - If λᵢ >> 0 (strongly outgoing): use interior value (information flows outward)
        - If λᵢ << 0 (strongly incoming): use farfield value (information flows inward)
        - If λᵢ ≈ 0 (near-sonic): blend smoothly between interior and farfield
        Blending function:
            αᵢ = 0.5 * (1 - tanh(λᵢ/c))
        Blended invariants:
            R⁻_bc = α₁ * R⁻_far + (1-α₁) * R⁻_int
            S_bc  = α₂ * S_far  + (1-α₂) * S_int
            R⁺_bc = α₃ * R⁺_far + (1-α₃) * R⁺_int

        Reconstruction:
        ---------------
        Once the blended invariants are computed, the primitive variables are
        reconstructed:

        1. Normal velocity:
           u_bc = (R⁺_bc + R⁻_bc) / 2
        2. Speed of sound:
           a_bc = (γ-1)/4 * (R⁺_bc - R⁻_bc)
        3. Density (from entropy and speed of sound):
           ρ_bc = (a²_bc / (γ * S_bc))^(1/(γ-1))
        4. Temperature (from equation of state):
           T_bc = a²_bc / (γ * R)

        Parameters:
        -----------
        U_face : torch.Tensor, shape [n_edges_bc, n_comp]
            Boundary face values to be modified in place
        Us_bc_cells : torch.Tensor, shape [n_farfield_edges, n_comp]
            Interior cell values adjacent to farfield edges [V_x, V_y, ρ, T]
        dt : float, optional
            Time step (unused, kept for interface compatibility)

        Notes:
        ------
        - The tangential velocity component is always preserved from interior
        - The blending scale is O(a), making the transition region sonic-scale
        - Farfield values R⁻_far, R⁺_far, S_far are precomputed from exit conditions
        - This approach is stable for both subsonic and supersonic flows
        """
        gm1 = self.gamma - 1

        # Interior properties
        V_int = Us_bc_cells[:, [0, 1]]                                    # shape = [n_ff_edge, 2]
        rho_int = Us_bc_cells[:, 2]                                   # shape = [n_ff_edge]
        T_int = Us_bc_cells[:, 3]                                     # shape = [n_ff_edge]
        # Parallel and tangential velocity
        V_n, V_t = self._split_Vs(V_int)

        # Interior invariants:
        a_int = self.phy_setup.eos_c(rho_int, T_int)
        R_m_int = V_n - 2 * a_int / gm1                     # 1
        S_int = self.R * T_int * rho_int ** (-gm1)          # 2
        R_p_int = V_n + 2 * a_int / gm1                     # 3

        # Interpolation values (assuming c = a_int). Transition smoothly on scale O(c)
        gating = self._gating(V_n, a_int)
        alpha1, alpha2, alpha3 = gating[:, 0], gating[:, 1], gating[:, 2]

        # Set boundary invariants
        R_m_bc = alpha1 * self.R_m_far + (1 - alpha1) * R_m_int
        S_bc = alpha2 * self.S_far + (1 - alpha2) * S_int
        R_p_bc = alpha3 * self.R_p_far + (1 - alpha3) * R_p_int

        # Reconstruct primatives
        V_n_bc = 1/2 * (R_m_bc + R_p_bc)
        a_bc_2 = (gm1/4 * (R_p_bc - R_m_bc)) ** 2
        rho_bc = (a_bc_2 / (self.gamma * S_bc)) ** (1 / gm1)
        T_bc = a_bc_2 / (self.gamma * self.R)

        # Add onto tangential component
        _, _, V_bc = self._recombine_Vs(V_n_bc, V_t)

        U_face_farfield = torch.cat([V_bc, rho_bc.unsqueeze(-1), T_bc.unsqueeze(-1)], dim=-1)
        U_face[self.bc_mask] = U_face_farfield
