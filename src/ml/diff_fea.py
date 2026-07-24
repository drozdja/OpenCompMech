"""Differentiable 2D SIMP FEA in torch — the engine for physics-GUIDED generation.

Assembles the global stiffness from a density field, solves under given fixed
DOFs and a probe load, and returns compliance. Because the whole path is torch
autograd, dC/drho is free — so a directional-stiffness objective (minimize
compliance under a unit load in direction d = maximize stiffness along d) can
steer a diffusion/flow sampler toward stiff-in-d designs the training data never
contained. That is the mechanism for producing NOVEL, functionally-specialised
structures rather than remixing the dataset.

Element stiffness + edof are reused from the numpy solver (src/solvers/linear)
so guidance is numerically consistent with the dataset's FEA labels. Element
stiffness is linear in E, so ke(E)=E·ke(1); per-element E follows SIMP:
E_e = Emin + rho_e^penal (E0 - Emin).

Dense solve: ndof = 2(res+1)^2 (8450 @64², 2178 @32²). Fine for guidance at
<=64² on the R9700; drop to 32² for cheap per-step guidance.
"""

import numpy as np
import torch

from src.solvers.linear import get_element_stiffness_cached, get_cached_edof


class DiffFEA:
    def __init__(self, res, nu=0.3, E0=1.0, Emin=1e-6, penal=3.0,
                 device="cuda", dtype=torch.float64):
        self.res = res
        self.penal = penal
        self.E0 = E0
        self.Emin = Emin
        self.device = device
        self.dtype = dtype
        self.ndof = 2 * (res + 1) * (res + 1)

        ke = get_element_stiffness_cached(1.0, nu)          # (8,8) for E=1
        edof = get_cached_edof(res, res).astype(np.int64)   # (n_elem, 8)
        self.ke = torch.as_tensor(ke, dtype=dtype, device=device)
        self.edof = torch.as_tensor(edof, dtype=torch.long, device=device)
        self.n_elem = self.edof.shape[0]

        # precompute flattened (row*ndof+col) assembly indices, once
        e = self.edof
        rows = e[:, :, None].expand(-1, 8, 8)
        cols = e[:, None, :].expand(-1, 8, 8)
        self.flat_idx = (rows * self.ndof + cols).reshape(-1)   # (n_elem*64,)
        self.ke_flat = self.ke.reshape(-1)                      # (64,)

    def assemble(self, rho):
        """rho (res,res) in [0,1] -> dense global K (ndof,ndof), differentiable."""
        Ee = self.Emin + torch.clamp(rho, 0, 1).reshape(-1) ** self.penal \
            * (self.E0 - self.Emin)                             # (n_elem,)
        vals = (Ee[:, None] * self.ke_flat[None]).reshape(-1)   # (n_elem*64,)
        K = torch.zeros(self.ndof * self.ndof, dtype=self.dtype, device=self.device)
        K = K.index_add(0, self.flat_idx, vals).reshape(self.ndof, self.ndof)
        return K

    def compliance(self, rho, fixed_dofs, f):
        """Solve K u = f on the free DOFs; return (compliance = fᵀu, u)."""
        K = self.assemble(rho)
        free = torch.ones(self.ndof, dtype=torch.bool, device=self.device)
        free[fixed_dofs] = False
        idx = torch.where(free)[0]
        Kff = K.index_select(0, idx).index_select(1, idx)
        uf = torch.linalg.solve(Kff, f[idx])
        u = torch.zeros(self.ndof, dtype=self.dtype, device=self.device)
        u = u.index_copy(0, idx, uf)
        return torch.dot(f, u), u

    def load_vector(self, node_dirs):
        """node_dirs: list of (node, fx, fy) -> global force vector."""
        f = torch.zeros(self.ndof, dtype=self.dtype, device=self.device)
        for node, fx, fy in node_dirs:
            f[2 * node] = fx
            f[2 * node + 1] = fy
        return f

    def directional_compliance(self, rho, fixed_dofs, probe_node, direction):
        """Compliance under a UNIT load at probe_node along `direction` (dx,dy).
        Lower = stiffer along that direction. Differentiable in rho."""
        d = torch.as_tensor(direction, dtype=self.dtype, device=self.device)
        d = d / (torch.linalg.norm(d) + 1e-12)
        f = self.load_vector([(probe_node, float(d[0]), float(d[1]))])
        c, _ = self.compliance(rho, fixed_dofs, f)
        return c


def bcs_to_fixed_dofs(bcs, device="cuda"):
    """Problem BCs (node_indices + directions: 0=x,1=y,2=both) -> fixed DOF idx."""
    dofs = []
    for bc in bcs:
        dirs = np.atleast_1d(bc.directions)
        if dirs.shape[0] == 1:
            dirs = np.full(len(bc.node_indices), int(dirs[0]))
        for n, dd in zip(bc.node_indices, dirs):
            if dd == 0:
                dofs.append(2 * int(n))
            elif dd == 1:
                dofs.append(2 * int(n) + 1)
            else:
                dofs += [2 * int(n), 2 * int(n) + 1]
    return torch.as_tensor(sorted(set(dofs)), dtype=torch.long, device=device)
