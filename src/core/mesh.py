"""
Mesh generation for 2D Q4 finite element analysis.

Provides structured quadrilateral mesh generation with:
- Node coordinate generation
- Element connectivity (4-node quads)
- DOF numbering (2 DOFs per node: x, y)
- Graph adjacency for GNN training
"""

import numpy as np
from typing import Tuple, NamedTuple
from dataclasses import dataclass


@dataclass
class Mesh2D:
    """
    Structured 2D quadrilateral mesh.
    
    Attributes:
        nelx: Number of elements in X direction
        nely: Number of elements in Y direction
        nodes: Node coordinates (n_nodes, 2)
        elements: Element connectivity (n_elements, 4) - node indices per element
        n_nodes: Total number of nodes
        n_elements: Total number of elements
        n_dofs: Total degrees of freedom (2 per node)
    """
    nelx: int
    nely: int
    nodes: np.ndarray      # (n_nodes, 2)
    elements: np.ndarray   # (n_elements, 4)
    
    @property
    def n_nodes(self) -> int:
        return (self.nelx + 1) * (self.nely + 1)
    
    @property
    def n_elements(self) -> int:
        return self.nelx * self.nely
    
    @property
    def n_dofs(self) -> int:
        return 2 * self.n_nodes
    
    def node_to_dofs(self, node_idx: int) -> Tuple[int, int]:
        """Convert node index to (dof_x, dof_y)."""
        return (2 * node_idx, 2 * node_idx + 1)
    
    def element_dofs(self, elem_idx: int) -> np.ndarray:
        """Get 8 DOF indices for a Q4 element."""
        nodes = self.elements[elem_idx]
        dofs = np.zeros(8, dtype=np.int32)
        for i, node in enumerate(nodes):
            dofs[2*i] = 2 * node      # x DOF
            dofs[2*i + 1] = 2 * node + 1  # y DOF
        return dofs
    
    def get_node_index(self, ix: int, iy: int) -> int:
        """Get node index from grid coordinates."""
        return iy * (self.nelx + 1) + ix
    
    def get_element_index(self, ix: int, iy: int) -> int:
        """Get element index from grid coordinates."""
        return iy * self.nelx + ix


def create_mesh(nelx: int, nely: int, lx: float = 1.0, ly: float = 1.0) -> Mesh2D:
    """
    Create a structured 2D quadrilateral mesh.
    
    Args:
        nelx: Number of elements in X direction
        nely: Number of elements in Y direction
        lx: Domain length in X (default 1.0 for normalized coords)
        ly: Domain length in Y (default 1.0 for normalized coords)
    
    Returns:
        Mesh2D object with nodes and element connectivity
    
    Node numbering (example 3x2):
        6---7---8---9
        |   |   |   |
        3---4---5---6      <- Error in comment, should be:
        |   |   |   |
        0---1---2---3      (nely+1) rows, (nelx+1) columns
        
    Actually for 3x2 (nelx=3, nely=2):
        9--10--11--12
        |   |   |   |
        5---6---7---8
        |   |   |   |
        0---1---2---3
    
    Element numbering (row-major from bottom-left):
        3---4---5
        0---1---2
    """
    # Node coordinates
    n_nodes = (nelx + 1) * (nely + 1)
    nodes = np.zeros((n_nodes, 2), dtype=np.float64)
    
    dx = lx / nelx
    dy = ly / nely
    
    for iy in range(nely + 1):
        for ix in range(nelx + 1):
            node_idx = iy * (nelx + 1) + ix
            nodes[node_idx, 0] = ix * dx
            nodes[node_idx, 1] = iy * dy
    
    # Element connectivity (4 nodes per element, counter-clockwise)
    n_elements = nelx * nely
    elements = np.zeros((n_elements, 4), dtype=np.int32)
    
    for ey in range(nely):
        for ex in range(nelx):
            elem_idx = ey * nelx + ex
            # Bottom-left, bottom-right, top-right, top-left (CCW)
            n0 = ey * (nelx + 1) + ex          # bottom-left
            n1 = ey * (nelx + 1) + ex + 1      # bottom-right
            n2 = (ey + 1) * (nelx + 1) + ex + 1  # top-right
            n3 = (ey + 1) * (nelx + 1) + ex    # top-left
            elements[elem_idx] = [n0, n1, n2, n3]
    
    return Mesh2D(nelx=nelx, nely=nely, nodes=nodes, elements=elements)


def create_graph_adjacency(nelx: int, nely: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create graph adjacency for element-based graph (for GNNs).
    
    Each element is a node in the graph. Edges connect adjacent elements
    (4-connectivity: left, right, top, bottom).
    
    Args:
        nelx: Number of elements in X direction
        nely: Number of elements in Y direction
    
    Returns:
        edges: (2, n_edges) array of edge indices [src, dst]
        pos: (n_nodes, 2) normalized element center positions [0, 1]
    """
    n_elements = nelx * nely
    edges = []
    
    for ey in range(nely):
        for ex in range(nelx):
            elem_idx = ey * nelx + ex
            
            # Right neighbor
            if ex < nelx - 1:
                neighbor = ey * nelx + (ex + 1)
                edges.append([elem_idx, neighbor])
                edges.append([neighbor, elem_idx])  # Undirected
            
            # Top neighbor
            if ey < nely - 1:
                neighbor = (ey + 1) * nelx + ex
                edges.append([elem_idx, neighbor])
                edges.append([neighbor, elem_idx])  # Undirected
    
    edges = np.array(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)
    
    # Element center positions (normalized to [0, 1])
    pos = np.zeros((n_elements, 2), dtype=np.float32)
    for ey in range(nely):
        for ex in range(nelx):
            elem_idx = ey * nelx + ex
            pos[elem_idx, 0] = (ex + 0.5) / nelx
            pos[elem_idx, 1] = (ey + 0.5) / nely
    
    return edges, pos


def generate_grid_assets(resolutions: list = [32, 64, 128], 
                         output_dir: str = "data/assets") -> None:
    """
    Pre-generate graph structures for all resolutions.
    
    Creates:
        grid_{res}_edges.npy - Edge indices (2, n_edges)
        grid_{res}_pos.npy   - Node positions (n_elements, 2)
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for res in resolutions:
        edges, pos = create_graph_adjacency(res, res)
        
        np.save(f"{output_dir}/grid_{res}_edges.npy", edges)
        np.save(f"{output_dir}/grid_{res}_pos.npy", pos)
        
        print(f"Generated grid_{res}: {edges.shape[1]} edges, {pos.shape[0]} nodes")


if __name__ == "__main__":
    # Generate assets when run directly
    generate_grid_assets()
