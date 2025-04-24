for bond in tree[-1]:
    neighbours = adj_list[bond[0]].copy()
    neighbours[:, 1:] += bond[1:] # since h,k,l are computed locally i.e j wrt to i not with origin 
    # we need to sum the local h,k,l with the previous neighbour to track the global h,k,l 
$$$$$$Expalnation$$$$$$$$$$

# Suppose:

# Atom 0 has a neighbor: Atom 1 in the unit cell just to the right → offset [1, 0, 0]

# Then Atom 1 has a neighbor: Atom 2 in the unit cell above it → offset [0, 1, 0] from Atom 1

# So from Atom 0, Atom 2 is located at:

Atom 1: [1, 0, 0]

Atom 2: [0, 1, 0] relative to Atom 1

So: [1, 0, 0] + [0, 1, 0] = [1, 1, 0] relative to Atom 0

# If you don’t add these offsets as you go deeper, you’ll think Atom 2 is still in the base cell or only offset by [0, 1, 0], and your descriptors (especially distance or angle-based ones) will be wrong.


#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX






#how these two lines work 
full_levels, idx = np.unique(np.vstack(tree), axis=0, return_index=True) # finds the first apperance of duplicates in the tree
tree[i] = full_levels[idx >= sum([len(tree[lv]) for lv in range(i)])]   #removes the duplicate

$$$$$$Expalnation$$$$$$$$$$

#Let's say we're at shell level i = 2 (i.e., shell 2).


tree = [
    [[0, 0, 0, 0]],                 # level 0: Atom A
    [[1, 0, 0, 0], [2, 0, 0, 0]],   # level 1: Atom B, C
    [[2, 0, 0, 0], [3, 0, 0, 0], [3, 0, 0, 0]]  # level 2 before filtering: C (again), D, D
]

# Step 1: Flatten all tree levels into one big array
vstacked = np.vstack(tree)
# vstacked = [
#   [0, 0, 0, 0],   # A
#   [1, 0, 0, 0],   # B
#   [2, 0, 0, 0],   # C
#   [2, 0, 0, 0],   # C again
#   [3, 0, 0, 0],   # D
#   [3, 0, 0, 0]    # D again
# ]

# Step 2: Unique rows + first occurrence indices
full_levels, idx = np.unique(vstacked, axis=0, return_index=True)
# full_levels = [
#   [0, 0, 0, 0],   # A
#   [1, 0, 0, 0],   # B
#   [2, 0, 0, 0],   # C
#   [3, 0, 0, 0]    # D
# ]
# idx = [0, 1, 2, 4]  (positions in vstacked where each unique row first appeared)

# Step 3: Calculate how many atoms were already in previous shells
# tree[0] has 1 atom, tree[1] has 2 atoms => total = 3
sum_len_prev = len(tree[0]) + len(tree[1])  # = 3

# Step 4: Find unique atoms that appeared **only in shell 2** or later
keep_mask = idx >= sum_len_prev
# keep_mask = [0 >= 3, 1 >= 3, 2 >= 3, 4 >= 3] => [False, False, False, True]

# Step 5: Keep only those new atoms (i.e., not in earlier shells)
tree[2] = full_levels[keep_mask]
# tree[2] = [[3, 0, 0, 0]]   # only D is retained in shell 2
