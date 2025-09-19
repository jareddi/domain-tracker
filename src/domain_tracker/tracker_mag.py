from tracker_utils import *

"""
%%%%%%%%%%%%%%%%%%%%%%%
-----------------------
Magnetization Functions
-----------------------
%%%%%%%%%%%%%%%%%%%%%%%
"""

def periodic_gradient(arr, spacing=1.0, axes=(0, 1, 2), periodic_axes=(0, 1)):
    grads = []
    for axis in axes:
        if axis in periodic_axes:
            grad = (np.roll(arr, -1, axis=axis) - np.roll(arr, 1, axis=axis)) / (2 * spacing)
        else:
            grad = np.gradient(arr, spacing, axis=axis)
        grads.append(grad)
    return grads

def find_theta_gamma(mx, my, mz, degrees=False):
    """
    Computes:
      - the in-plane magnetization angle (mx, my)
      - the angle between magnetization and the domain wall normal (from ∇mz)
    """
    # Compute in-plane magnetization angle
    theta = np.arctan2(my, mx)

    # Compute gradient of mz and get the negative x-y gradient (i.e., from +mz to -mz)
    grad = periodic_gradient(mz)
    dy, dx = [-grad[1], -grad[0]]  # y = axis 1, x = axis 0 for shape (nx, ny, nz)

    theta_wall = np.arctan2(dy, dx)

    # Angle difference, wrapped to (-pi, pi)
    gamma = (theta - theta_wall + np.pi) % (2 * np.pi) - np.pi

    if degrees:
        theta = np.degrees(theta)
        gamma = np.degrees(gamma)

    return theta, gamma

def winding_number(thetas,):
    """
    Calculate winding number from in-plane magnetization angles along a closed contour.
    """
    unwrapped = np.unwrap(thetas)  # unwrap angle discontinuities
    dtheta = np.diff(unwrapped)
    total_change = np.sum(dtheta)
    w = int(np.round(total_change / (2*np.pi)))
    return w

def discard_false_swaps(swaps):
    """
    Removes swaps which do not actually correspond to a topological structure.
    """
    if len(swaps) < 3:
        return swaps

    swaps = list(swaps)
    keep_mask = [True] * len(swaps)

    # Group indices by flip type
    type_to_indices = defaultdict(list)
    for idx, (_, _, ftype) in enumerate(swaps):
        type_to_indices[ftype].append(idx)

    for ftype, group in type_to_indices.items():
        if len(group) < 3:
            continue

        to_remove = set()
        # Use sliding window of size 3
        for i0, i1, i2 in zip(group[:-2], group[1:-1], group[2:]):
            x0, x1, x2 = swaps[i0][0], swaps[i1][0], swaps[i2][0]

            # Distances between swap positions
            d01 = abs(x0 - x1)
            d12 = abs(x1 - x2)
            d02 = abs(x0 - x2)

            # Mark the closest pair for removal
            if d01 <= d12 and d01 <= d02:
                to_remove.update([i0, i1])
            elif d12 <= d01 and d12 <= d02:
                to_remove.update([i1, i2])
            else:
                to_remove.update([i0, i2])

        # Apply removal
        for idx in to_remove:
            keep_mask[idx] = False

    return [swap for keep, swap in zip(keep_mask, swaps) if keep]

def detect_chirality_swaps(gammas, angle_threshold=np.pi/3):
    """
    Efficiently detect chirality swaps in a 1D angle array (range [-π, π]).

    Returns:
        List of tuples: (center_index, width, chirality_type)
    """
    N = len(gammas)

    # Precompute signs and differences
    signs = np.sign(gammas)
    sign_change = signs[:-1] != signs[1:]
    angle_diff = np.abs(gammas[1:] - gammas[:-1])
    flip_type = np.where(angle_diff > np.pi, 'in', 'out')

    # Precompute transition zone masks
    pi_flip_zone = (gammas > (np.pi - angle_threshold)) | (gammas < (-np.pi + angle_threshold))
    zero_flip_zone = np.abs(gammas) > angle_threshold

    swaps = []

    # Loop only over actual sign changes
    flip_indices = np.nonzero(sign_change)[0] + 1  # +1 since diff is shifted left

    for idx in flip_indices:
        center = idx
        flip_kind = flip_type[idx - 1]
        if flip_kind == 'in':
            zone = pi_flip_zone
        else:
            zone = zero_flip_zone
        # Walk left and right efficiently
        l = center - 1
        while l > 0 and zone[l]:
            l -= 1
        r = center
        while r < N - 1 and zone[r]:
            r += 1
        width = r - l
        swaps.append((center, width, flip_kind))

    return discard_false_swaps(swaps)

def segment_thetas_gammas_bps(segment,theta_arr,gamma_arr,layer,image_shape):
    """
    Extract a single segment's domain wall angle information including Bloch Points.
    """
    thetas = get_values_along_contour(theta_arr,segment,layer)
    gammas = get_values_along_contour(gamma_arr,segment,layer)

    gammas_unwrapped = np.unwrap(gammas)
    dg = int(np.round((gammas_unwrapped[0] - gammas_unwrapped[-1])/(2*np.pi)))
    if dg == 0:
        bps = []
    else:
        bps_raw = detect_chirality_swaps(gammas)
        bps = []
        for bp_index, width, chirality_type in bps_raw:
            xy_coord = tuple([int(segment[bp_index,0,0]%image_shape[0]),int(segment[bp_index,0,1]%image_shape[1]),layer])  # OpenCV contour indexing
            bps.append((xy_coord, int(width), str(chirality_type)))
    
    return thetas,gammas,bps

def segment_gammas_bps(segment,gamma_arr,layer,image_shape):
    """
    Extract a single segment's domain wall angle information including Bloch Points.
    """
    gammas = get_values_along_contour(gamma_arr,segment,layer)
    gammas_unwrapped = np.unwrap(gammas)
    dg = int(np.round((gammas_unwrapped[0] - gammas_unwrapped[-1])/(2*np.pi)))
    if dg == 0:
        bps = []
    else:
        bps_raw = detect_chirality_swaps(gammas)
        bps = []
        for bp_index, width, chirality_type in bps_raw:
            xy_coord = tuple([int(segment[bp_index,0,0]%image_shape[0]),int(segment[bp_index,0,1]%image_shape[1]),layer])  # OpenCV contour indexing
            bps.append((xy_coord, int(width), str(chirality_type)))
    
    return gammas,bps

def segment_bps(segment,gamma_arr,layer,image_shape):
    """
    Extract a single segment's domain wall angle information including Bloch Points.
    """
    gammas = get_values_along_contour(gamma_arr,segment,layer)
    gammas_unwrapped = np.unwrap(gammas)
    dg = int(np.round((gammas_unwrapped[0] - gammas_unwrapped[-1])/(2*np.pi)))
    if dg == 0:
        bps = []
    else:
        bps_raw = detect_chirality_swaps(gammas)
        bps = []
        for bp_index, width, chirality_type in bps_raw:
            xy_coord = tuple([int(segment[bp_index,0,0]%image_shape[0]),int(segment[bp_index,0,1]%image_shape[1]),layer])  # OpenCV contour indexing
            bps.append((xy_coord, int(width), str(chirality_type)))
    
    return bps

def find_bls(bps, distance_threshold=5.0):
    """
    Group BPs (Bloch points) into BLs (Bloch lines) by proximity and type.

    Parameters:
        bps: List of (x, y, z, type) tuples
        distance_threshold: max 3D distance for BPs to be considered connected

    Returns:
        Dictionary of BL dictionaries:
    """
    if not bps:
        return {}

    # Extract components from bps
    positions = [bp[0] for bp in bps]
    widths = [bp[1] for bp in bps]
    types = [bp[2] for bp in bps]

    visited = [False] * len(bps)
    bls = {}
    bl_id = 0

    for i in range(len(bps)):
        if visited[i]:
            continue

        visited[i] = True
        queue = [i]
        current_bl = [i]

        while queue:
            current = queue.pop()
            x0, y0, z0 = positions[current]
            type0 = types[current]

            for j in range(len(bps)):
                if visited[j] or types[j] != type0:
                    continue

                x1, y1, z1 = positions[j]
                if abs(round(z1) - round(z0)) > 1:
                    continue  # Only connect to adjacent z-layers

                dist_xy = np.hypot(x1 - x0, y1 - y0)
                if dist_xy <= distance_threshold or distance_threshold <= 0:
                    visited[j] = True
                    queue.append(j)
                    current_bl.append(j)

        # Compute BL properties
        bl_positions = [positions[idx] for idx in current_bl]
        bl_widths = [widths[idx] for idx in current_bl]
        x_avg = np.mean([p[0] for p in bl_positions])
        y_avg = np.mean([p[1] for p in bl_positions])
        z_vals = [p[2] for p in bl_positions]
        z_extent = (min(z_vals), max(z_vals))

        bls[bl_id] = {
            'bp_indices': current_bl,
            'type': types[i],
            'position': (x_avg, y_avg),
            'z_extent': z_extent,
            'widths': bl_widths
        }
        bl_id += 1
    return bls