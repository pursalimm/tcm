import sys
import os
import warnings

warnings.filterwarnings('ignore')

import builtins
_original_print = builtins.print

def silent_print(*args, **kwargs):
    pass

builtins.print = silent_print

import numpy as np
import pandas as pd

builtins.print = _original_print

def setup_patches():
    for attr in ['long', 'ulong', 'alltrue', 'sometrue', 'product']:
        if not hasattr(np, attr):
            if attr == 'long':
                setattr(np, attr, np.int64)
            elif attr == 'ulong':
                setattr(np, attr, np.uint64)
            elif attr == 'alltrue':
                setattr(np, attr, np.all)
            elif attr == 'sometrue':
                setattr(np, attr, np.any)
            elif attr == 'product':
                setattr(np, attr, np.prod)
    
    import collections
    import collections.abc
    for attr_name in ['Sequence', 'Iterable', 'Mapping', 'MutableMapping', 'Set']:
        if hasattr(collections.abc, attr_name) and not hasattr(collections, attr_name):
            setattr(collections, attr_name, getattr(collections.abc, attr_name))
    
    try:
        import louvain
    except ImportError:
        import community as louvain
        sys.modules['louvain'] = louvain

def run_spaotsc(sc_data, sc_labels, st_data, st_coords, cell_types=None):
    builtins.print = silent_print
    
    try:
        setup_patches()
        
        import spaotsc.SpaOTsc as spaotsc_module
        spatial_sc = spaotsc_module.spatial_sc
        
        n_genes = sc_data.shape[0]
        n_cells = sc_data.shape[1]
        n_spots = st_data.shape[1]
        
        if n_cells > n_spots:
            np.random.seed(42)
            cell_indices = np.random.choice(n_cells, n_spots, replace=False)
            sc_data = sc_data[:, cell_indices]
            sc_labels = [sc_labels[i] for i in cell_indices]
            n_cells = n_spots
        
        sc_data_df = pd.DataFrame(
            sc_data.T,
            columns=[f'gene_{i}' for i in range(n_genes)]
        )
        
        from scipy.spatial.distance import cdist, pdist, squareform
        
        is_dmat = squareform(pdist(st_coords))
        
        sc_dmat = squareform(pdist(sc_data_df.values))
        
        st_data_df = pd.DataFrame(
            st_data.T,
            columns=[f'gene_{i}' for i in range(n_genes)]
        )
        cost_matrix = cdist(st_data_df.values, sc_data_df.values, metric='euclidean')
        
        spsc = spatial_sc(
            sc_data=sc_data_df,
            is_dmat=is_dmat,
            sc_dmat=sc_dmat
        )
        
        # transport plan
        spsc.transport_plan(cost_matrix)
        
        # gamma_mapping
        transport_matrix = spsc.gamma_mapping
        
        if transport_matrix is None:
            raise ValueError("gamma_mapping is None")
        
        if hasattr(transport_matrix, 'toarray'):
            transport_matrix = transport_matrix.toarray()
        transport_matrix = np.array(transport_matrix)
        
        unique_cts = list(set(sc_labels))
        proportions = np.zeros((n_spots, len(unique_cts)))
        
        for i, ct in enumerate(unique_cts):
            cell_indices = [j for j, label in enumerate(sc_labels) if label == ct]
            if len(cell_indices) > 0:
                proportions[:, i] = transport_matrix[:, cell_indices].sum(axis=1)
        
        proportions = proportions / np.maximum(proportions.sum(axis=1, keepdims=True), 1e-10)
        
        return proportions, unique_cts
    
    finally:
        builtins.print = _original_print

if __name__ == "__main__":
    print("SpaOTsc module ready")