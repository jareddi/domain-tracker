from tracker_utils import *
from scipy.ndimage import distance_transform_edt

def dilate_domain_pbc(domains_z, mz, max_dist = None, mz_thresh = None):
    """
    Updates domains_z in-place by adding a new 'mask' key to each entry.
    The new 'mask' is a dilated, non-overlapping, PBC-aware version of the 'core' mask.
    Dilation is limited by `max_dist` pixels.

    Parameters:
        domains_z: dict[int, dict], where each value contains a 'core' 2D mask
        max_dist: int, maximum pixel distance for dilation
    """
    if not domains_z:
        return  # nothing to do if domains_z is empty
    if not max_dist:
        max_dist = np.inf
    if not mz_thresh:
        mz_thresh = -0.5

    keys = list(domains_z.keys())
    h, w = next(iter(domains_z.values()))['core'].shape

    # Step 1: build initial labeled image
    label_img = np.zeros((h, w), dtype=np.int32)
    for i, key in enumerate(keys, start=1):
        mask = domains_z[key]['core'].astype(bool)
        label_img[mask] = i

    # Step 2: tile 3x3 for PBC
    label_tile = np.tile(label_img, (3, 3))
    H, W = label_tile.shape

    # Step 3: compute distance transform on background
    background = (label_tile == 0)
    distances, nearest = distance_transform_edt(background, return_indices=True)  # type: ignore
    i0, j0 = nearest

    # Step 4: assign nearest label only if within max_dist
    nearest_labels = label_tile[i0, j0]
    limited_labels = np.where(distances <= max_dist, nearest_labels, 0)

    # Step 5: extract center tile and update each domain's mask
    center_y, center_x = h, w
    final_labels = limited_labels[center_y:center_y + h, center_x:center_x + w]

    for i, key in enumerate(keys, start=1):
        domains_z[key]['mask'] = ((final_labels == i) & (mz > mz_thresh))
        area_dilation = domains_z[key]['mask'].sum()
        domains_z[key]['properties']['area_dilated'] = area_dilation