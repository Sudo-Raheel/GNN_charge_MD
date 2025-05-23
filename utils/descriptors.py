from itertools import combinations, product

import numpy as np

from utils.elements import en_ghosh


def get_dist_features(cif_struct, symbols, n_atoms):
    # obtain properties list
    property_en = np.array([en_ghosh[symbol] for symbol in symbols])

    # determine the number of mirror images required in the supercell
    max_frac = cif_struct.lattice.get_fractional_coords([8.0, 8.0, 8.0])
    super_scale = np.ceil(max_frac).astype(int)
    list_options = list(product(*[range(2 * i + 1) for i in super_scale]))
    super_options = np.array(list_options, dtype=float) - super_scale

    # create the supercell in both fractional and cartesian coordinates
    frac_unit = np.array([atom.frac_coords for atom in cif_struct])
    frac_tiles = np.tile(frac_unit, (len(super_options), 1, 1))
    super_tiles = np.repeat(super_options, n_atoms, axis=0).reshape(frac_tiles.shape)
    frac_super = frac_tiles - super_tiles
    cart_super = cif_struct.lattice.get_cartesian_coords(frac_super)
    unit_cell_idx = len(super_options) // 2
    cart_unit = cart_super[unit_cell_idx]

    # parameters for descriptors
    r_0 = 1.0
    rad_cut = 8.0
    ang_cut = 6.0
    rdf_bin_size = 0.25
    rdf_alpha = 60.0
    acsf_rad_bin_size = 0.5
    acsf_rad_eta = 6.0
    adf_bin_size_degree = 10
    adf_beta = 60.0
    acsf_ang_n_eta = 8

    # inits for atomic property weighted radial distribution functions (wRDF)
    rdf_n_bins = int((rad_cut - r_0) / rdf_bin_size)
    rdf_edges = np.linspace(r_0 + rdf_bin_size, rad_cut, num=rdf_n_bins)

    # inits for radial ACSF
    acsf_rad_n_bins = int((rad_cut - r_0) / acsf_rad_bin_size)
    acsf_rad_edges = np.linspace(r_0 + acsf_rad_bin_size, rad_cut, num=acsf_rad_n_bins)

    # inits for atomic property weighted angular distribution functions (wADF)
    adf_bin_size = np.deg2rad(adf_bin_size_degree)
    adf_n_bins = int(np.pi / adf_bin_size)
    adf_edges = np.linspace(adf_bin_size, np.pi, num=adf_n_bins)

    # inits for angular ACSF
    acsf_ang_edges = np.linspace(r_0, ang_cut, num=acsf_ang_n_eta)
    acsf_ang_eta = 1 / (2 * (acsf_ang_edges**2))
    acsf_ang_lambda = [-1, 1]
    acsf_ang_n_bins = acsf_ang_n_eta * 2
    acsf_ang_lambda_bins, acsf_ang_eta_bins = np.array(
        list(product(acsf_ang_lambda, acsf_ang_eta))
    ).T

    # generate descriptors for one atom at a time to save memory
    features_list = []
    for cart_i in cart_unit:
        # compute distances between atom i to all other atoms in the supercell
        dist_to_i = np.linalg.norm(cart_super - cart_i, axis=-1)

        # obtain distances within the cutoffs for radial descriptors
        rad_bool = (dist_to_i >= r_0) & (dist_to_i <= rad_cut)
        rad_p = property_en[np.nonzero(rad_bool)[1]]
        rad_dist = dist_to_i[rad_bool]

        # compute wRDF
        rdf_diff_bins = (
            np.repeat(rad_dist, rdf_n_bins).reshape(-1, rdf_n_bins) - rdf_edges
        )
        rdf_gauss_bins = np.exp(-rdf_alpha * (rdf_diff_bins**2))
        rdf = (rdf_gauss_bins.T * rad_p).sum(axis=1)
        #rad_p is the 

        # compute RWAAP
        wap_rad_btm = rdf_gauss_bins.sum(axis=0)
        wap_rad = np.divide(
            rdf, wap_rad_btm, out=np.zeros(rdf_n_bins), where=(wap_rad_btm != 0)
        )

        # compute wRACSF
        acsf_rad_fcut = (np.cos(rad_dist * np.pi / rad_cut) + 1) * 0.5
        acsf_rad_diff_bins = (
            np.repeat(rad_dist, acsf_rad_n_bins).reshape(-1, acsf_rad_n_bins)
            - acsf_rad_edges
        )
        acsf_rad_gauss_bins = np.exp(-acsf_rad_eta * (acsf_rad_diff_bins**2))
        acsf_rad = (rad_p * acsf_rad_fcut * acsf_rad_gauss_bins.T).sum(axis=1)

        # obtain distances within the cutoffs for angular descriptors
        ang_bool = (dist_to_i >= r_0) & (dist_to_i <= ang_cut)
        ang_p = property_en[np.nonzero(ang_bool)[1]]
        ang_dist = dist_to_i[ang_bool]

        if len(ang_dist) > 1:
            # compute angles for all triplets (i,j,k) within the supercell
            ang_sphere_cart = cart_super[ang_bool]
            atoms_j, atoms_k = np.array(list(combinations(range(len(ang_dist)), 2))).T
            property_jk = ang_p[atoms_j] * ang_p[atoms_k]
            d_ijk = np.vstack(
                [
                    ang_dist[atoms_j],
                    ang_dist[atoms_k],
                    np.linalg.norm(
                        ang_sphere_cart[atoms_j] - ang_sphere_cart[atoms_k], axis=1
                    ),
                ]
            )
            d_ijk_sq = d_ijk**2
            cos_theta_ijk = (d_ijk_sq[0] + d_ijk_sq[1] - d_ijk_sq[2]) / (
                2 * d_ijk[0] * d_ijk[1]
            )
            cos_theta_ijk[cos_theta_ijk < -1.0] = -1.0
            cos_theta_ijk[cos_theta_ijk > 1.0] = 1.0
            theta_ijk = np.arccos(cos_theta_ijk)

            # compute wADF
            adf_diff_bins = np.tile(theta_ijk, (adf_n_bins, 1)).T - adf_edges
            adf_gauss_bins = np.exp(-adf_beta * (adf_diff_bins**2))
            adf = (property_jk * adf_gauss_bins.T).sum(axis=1)

            # compute AWAAP
            wap_ang_btm = adf_gauss_bins.sum(axis=0)
            wap_ang = np.divide(
                adf, wap_ang_btm, out=np.zeros(adf_n_bins), where=(wap_ang_btm != 0)
            )

            # compute wAACSF
            acsf_ang_fcut = ((np.cos(d_ijk * np.pi / ang_cut) + 1) * 0.5).prod(axis=0)
            acsf_ang_pre = np.tile(cos_theta_ijk, (acsf_ang_n_bins, 1)).T
            acsf_ang_cos_bins = 1 + acsf_ang_lambda_bins * acsf_ang_pre
            acsf_ang_dist_bins = np.tile(d_ijk_sq.sum(axis=0), (acsf_ang_n_bins, 1)).T
            acsf_ang_gauss_bins = np.exp(-acsf_ang_eta_bins * acsf_ang_dist_bins)
            acsf_ang_bins = (acsf_ang_cos_bins * acsf_ang_gauss_bins).T
            acsf_ang_bins *= property_jk * acsf_ang_fcut
            acsf_ang = acsf_ang_bins.sum(axis=1)
        else:
            # no angular descriptors computed if there are not enough neighbours
            adf, wap_ang = np.zeros((2, adf_n_bins))
            acsf_ang = np.zeros(acsf_ang_n_bins)

        features_list.append(
            np.hstack(
                [
                    rdf,
                    adf,
                    acsf_rad,
                    acsf_ang,
                    wap_rad,
                    wap_ang,
                ]
            ),
        )

    return np.vstack(features_list)


def get_shell_features(bonds_full, property_matrix, n_atoms):
    # convert adj_mat to adj_list
    # the list is a 2D list in which the frst index represents i 
    # adj_list[0] contains all the neighbour of the 0th atoms ..it itself is a list 
    # for ex it can be [2,0,0,0],[2,-1,-1,-1]... where frst index is jth atom while following are the miller indices
    adj_list = [bonds_full[bonds_full[:, 0] == i][:, 1:] for i in range(n_atoms)] 
    #extended form below  
    # adj_list = []
    # for i in range(n_atoms):
        # bonds_from_i = bonds_full[bonds_full[:, 0] == i]  # select rows where source == i
        # neighbors_info = bonds_from_i[:, 1:]              # get [neighbor, dx, dy, dz]
        # adj_list.append(neighbors_info)

    # init for shell descriptors
    n_shells = 4
    n_properties = 8
    shell_avg = np.zeros((n_atoms, n_properties * n_shells))
    shell_std = np.zeros((n_atoms, n_properties * n_shells))
    shell_diff = np.zeros((n_atoms, n_properties * n_shells))

    for source in range(n_atoms):  # runs over all the nodes creating trees for each node 
        # skip atom if there's no neighbours
        if len(adj_list[source]) == 0:
            continue

        # create a tree that represent coordination shells of the source atom
        tree = [np.array([[source, 0, 0, 0]]), adj_list[source]] # already contains the 0th node and frst neighbour
        for i in range(2, n_shells + 1):
            if len(tree[-1]) < 1:  
                tree.pop()
                break
            nodes = []
            for bond in tree[-1]:
                neighbours = adj_list[bond[0]].copy()
                neighbours[:, 1:] += bond[1:] # since h,k,l are computed locally i.e j wrt to i not with origin 
                # we need to sum the local h,k,l with the previous neighbour to track the global h,k,l 
                nodes.append(neighbours)
            nodes = np.unique(np.vstack(nodes), axis=0)  #np.unique eliminates duplicate neighbors that are identical in both index and offset.
            tree.append(nodes)
            full_levels, idx = np.unique(np.vstack(tree), axis=0, return_index=True) # finds the first apperance of duplicates in the tree
            tree[i] = full_levels[idx >= sum([len(tree[lv]) for lv in range(i)])]   #removes the duplicates 

        # compute descriptors for each shell
        self_properties = property_matrix[source]
        for pos, level in enumerate(tree[1:]):
            level_properties = property_matrix[level[:, 0]]
            if len(level) < 1:
                break
            elif len(level) == 1:
                lv_avg = level_properties
                lv_std = 0.0
                lv_diff = abs(level_properties - self_properties)
            else:
                lv_avg = level_properties.mean(axis=0)
                lv_std = level_properties.std(axis=0)
                lv_diff = abs(level_properties - self_properties).mean(axis=0)
            loc_head = pos * n_properties
            loc_tail = loc_head + n_properties  
            # all the statistical aggregations are performed for each shell. 
            shell_avg[source, loc_head:loc_tail] = lv_avg   
            shell_std[source, loc_head:loc_tail] = lv_std
            shell_diff[source, loc_head:loc_tail] = lv_diff

    return np.hstack([shell_avg, shell_std, shell_diff])
