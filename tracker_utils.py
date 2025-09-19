from pathlib import Path
import os
import re
import numpy as np
import ovf2io as ovf
import cv2 as cv
from collections import Counter, defaultdict, deque

"""
%%%%%%%%%%%%%%%%%%%%%%%
-----------------------
   Simple Functions    
-----------------------
%%%%%%%%%%%%%%%%%%%%%%%
"""

def get_data(data):
    """
    get component-wise data from ovf file or dict
    """
    if isinstance(data, Path):
        title = f"{os.path.basename(data)[:-4]}"
        data = ovf.read_ovf(data)
    elif isinstance(data, dict):
        title = data['title']
        data = data
    return data, title # type: ignore

def check_edge_point(points, image_shape):
    width, height = image_shape

    # If the input is a single point (tuple)
    if isinstance(points, list):
        x, y = points
        return x == 0 or y == 0 or x == width - 1 or y == height - 1

    # If the input is a list of points (contour)
    elif isinstance(points, np.ndarray):
        points = points.reshape(-1, 2)
        return [1 if (point[0] == 0 or point[1] == 0 or point[0] == width - 1 or point[1] == height - 1) else 0 for point in points]
    else:
        raise ValueError("Input must be a tuple or a list of tuples")
    
def check_duplicate_point(contour):
    """
    Returns a list of 0s and 1s indicating whether each point in the contour
    has at least one duplicate elsewhere.
    """
    points = contour[:, 0, :]  # shape (N, 2)
    point_tuples = [tuple(pt) for pt in points]

    counts = Counter(point_tuples)
    return [1 if counts[pt] > 1 else 0 for pt in point_tuples]

def pbc_min_dist(a1, a2, shape):
    """
    Compute the minimum distance between two 3D points under periodic
    boundary conditions in x and y only.
    """
    delta = a1 - a2
    # Apply PBC in x (axis 0) and y (axis 1)
    for i in [0, 1]:
        L = shape[i]
        delta[i] = delta[i] - L * np.round(delta[i] / L)
    # No PBC in z (axis 2), so leave as-is
    return np.linalg.norm(delta)

def pack_mask(mask):
    """Pack a 3D boolean array into the smallest 1D array of bytes."""
    if mask.dtype != np.bool_:
        raise ValueError("Array must be boolean.")
    return np.packbits(mask.ravel())

def unpack_mask(packed_mask, original_shape):
    """Unpack from bit-packed array back to the original boolean shape."""
    unpacked = np.unpackbits(packed_mask)
    return unpacked.reshape(original_shape).astype(bool)

def contours_mask(contours, image_shape, closed=True, fill=True, pack=False):
    mask = np.zeros(image_shape)
    if closed:
        if fill:
            cv.drawContours(mask, contours, contourIdx=-1, color=True, thickness=cv.FILLED)
        else:
            cv.drawContours(mask, contours, contourIdx=-1, color=True, thickness=1)
    else:
        cv.polylines(mask, contours, isClosed=False, color=True, thickness=1) # type: ignore
    if pack:
        return pack_mask(mask.astype(bool))
    else:
        return mask.astype(bool)

def shift_contour(contour, offset):
    return contour + offset.reshape(1, 1, 2)

def shift_segment(prev_end, segment, image_shape):
    """
    Shifts a segment so that it connects smoothly with prev_end.
    """
    w, h = image_shape
    start = segment[0, 0]

    dx = start[0] - prev_end[0]
    dy = start[1] - prev_end[1]

    # Resolve wrapping if needed
    if dx > w // 2: dx -= w
    if dx < -w // 2: dx += w
    if dy > h // 2: dy -= h
    if dy < -h // 2: dy += h

    shift = np.array(prev_end) - np.array(start) + np.array([dx,dy])
    shifted = segment[:, 0, :] + shift
    return shifted.reshape(-1, 1, 2)

def get_values_along_contour(data, contour, z_index):
    """
    Extract values from a 3D array exactly at the contour points, preserving order.
    """
    w,h = data.shape[:2]
    # Extract points (x, y)
    points = contour[:, 0, :]  # shape (N, 2)
    x = points[:, 0] % w
    y = points[:, 1] % h
    return data[x, y, z_index]

def cumulative_rotation(angles, wrapping_tolerance = 0.9):
    angles = np.asarray(angles)
    valid_mask = ~np.isnan(angles)
    angles = angles[valid_mask]

    if len(angles) == 0:
        return np.array([])

    out = [0]
    for i in range(1, len(angles)):
        diff = angles[i] - angles[i - 1]
        # Correct for discontinuity at ±π/2 (equivalent to a jump of ±π)
        if diff > wrapping_tolerance*np.pi:
            diff -= np.pi
        elif diff < -wrapping_tolerance*np.pi:
            diff += np.pi
        out.append(out[-1] + diff)
    return np.array(out)

"""
%%%%%%%%%%%%%%%%%%%%%%%
-----------------------
  Cleaning Functions   
-----------------------
%%%%%%%%%%%%%%%%%%%%%%%
"""
def wrap_contours(contours, image_shape, threshold=0.5):
    """
    Wraps contours using periodic boundary conditions and breaks them into segments
    where wrapping occurs (i.e., large jumps).
    """
    w, h = image_shape
    wrapped_segments = []

    for contour in contours:
        # Extract points and wrap using PBC
        pts = contour[:, 0, :]
        wrapped_pts = np.mod(pts, [w, h])

        # Compute deltas between consecutive points
        deltas = np.diff(wrapped_pts, axis=0)
        jumps = np.logical_or(
            np.abs(deltas[:, 0]) > threshold * w,
            np.abs(deltas[:, 1]) > threshold * h
        )

        # Find indices where the contour should be split
        split_indices = np.where(jumps)[0] + 1
        splits = np.split(wrapped_pts, split_indices)

        # Reformat into OpenCV contour format
        for s in splits:
            if len(s) >= 2:  # Skip degenerate segments
                wrapped_segments.append(s.reshape(-1, 1, 2).astype(np.int32))

    return wrapped_segments

def remove_duplicate_points(contour):
    seen = set()
    unique_pts = []
    for pt in contour:
        # Flatten pt[0] to a tuple (x, y)
        tup = tuple(int(coord) for coord in pt[0])
        if tup not in seen:
            seen.add(tup)
            unique_pts.append([[tup[0], tup[1]]])  # Re-wrap to [[x, y]] for OpenCV

    return np.array(unique_pts, dtype=np.int32)

def find_duplicate_indices(contour):
    """
    Returns the indices of duplicate points in a contour.
    """
    seen = set()
    duplicates = []
    for i, pt in enumerate(contour):
        tup = tuple(int(coord) for coord in pt[0])  # Flatten pt[0] to (x, y)
        if tup in seen:
            duplicates.append(i)
        else:
            seen.add(tup)
    return duplicates

def clean_double_backs(n, edge_flags, dup_flags):
    """
    For a contour with edge points and duplicate points, remove edge segment without breaking continuity.
    """
    # Identify doubleback points (as a set for fast lookup)
    double_backs = {
        i for i in range(n)
        if edge_flags[i] and not dup_flags[i]
        and dup_flags[(i - 1) % n] and dup_flags[(i + 1) % n]
    }

    removed = set()
    visited = set()

    for idx in sorted(double_backs):
        if idx in visited:
            continue
        visited.add(idx)

        # Forward walk
        f_indices = []
        i = (idx + 1) % n
        while i != idx and edge_flags[i]:
            if i in double_backs:
                visited.add(i)
                break
            f_indices.append(i)
            i = (i + 1) % n

        # Backward walk
        b_indices = []
        i = (idx - 1) % n
        while i != idx and edge_flags[i]:
            if i in double_backs:
                visited.add(i)
                break
            b_indices.append(i)
            i = (i - 1) % n

        # Determine which side to remove
        if len(f_indices) <= len(b_indices):
            remove = b_indices
        else:
            remove = f_indices

        if remove:
            last = remove[-1]
            next_f = (last + 1) % n
            next_b = (last - 1) % n
            if not edge_flags[next_f] or not edge_flags[next_b]:
                # Don't include this point; it's likely part of the return path
                remove.pop()
            """
            if next_f in double_backs:
                # It's a valid edge and leads to another doubleback; include it
                remove.append(next_f)
            if next_b in double_backs:
                # It's a valid edge and leads to another doubleback; include it
                remove.append(next_b)
            """

        removed.update(remove)

    # Compute preserved edge indices: edge points not in removed set
    preserved = [i for i in range(n) if edge_flags[i] and i not in removed]

    return list(removed), preserved

def clean_contour(contour, shared_indices, image_shape):
    """
    Removes shared indices unless they're isolated boundary points.
    Skips false segments caused by shared-point pruning.
    """
    # Get contour info
    n = len(contour)
    keep_mask = np.ones(n, dtype=bool)
    edge_flags = np.asarray(check_edge_point(contour,image_shape)).flatten()
    if sum(edge_flags) == n:
        all_edge = True
    else:
        all_edge = False
    dup_flags = np.asarray(check_duplicate_point(contour)).flatten()

    # Prune unnecessary shared indices (those that are deeply embedded)
    if not all_edge:
        for idx in shared_indices:
            left = (idx - 1) % len(contour)
            right = (idx + 1) % len(contour)
            shared_neighbors = sum(n in shared_indices for n in (left, right))
            if shared_neighbors == 2:
                keep_mask[idx] = False
    else:
        keep_mask[find_duplicate_indices(contour)] = False

    # Remove duplicate segments which affect our segmentation process
    if sum(dup_flags) != 0:
        indices_to_remove, indices_to_preserve = clean_double_backs(n, edge_flags, dup_flags)
        keep_mask[indices_to_remove] = False
        keep_mask[indices_to_preserve] = True

    # Get mapping from old index to new index
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(np.flatnonzero(keep_mask))}
    # Filter and remap shared_indices
    updated_shared_indices = [old_to_new[i] for i in shared_indices if keep_mask[i]]

    return contour[keep_mask], updated_shared_indices

"""
%%%%%%%%%%%%%%%%%%%%%%%
-----------------------
     Segmentation   
-----------------------
%%%%%%%%%%%%%%%%%%%%%%%
"""
def find_shared_indices(values1, values2, indices1):
    """
    Return indices from indices1 corresponding to shared values between values1 and values2.
    """
    shared = np.intersect1d(values1, values2)
    if len(shared) == 0:
        return []

    # Return indices from indices1 where values match shared
    return [idx for val, idx in zip(values1, indices1) if val in shared]

def find_unsegmented_pairs(indices, contour_length):
    indices = np.asarray(indices)
    sorted_idx = np.argsort(indices)
    sorted_indices = indices[sorted_idx]
    diffs = (np.roll(sorted_indices, -1) - sorted_indices) % (contour_length)
    is_adjacent_next = diffs == 1
    is_adjacent_prev = np.roll(diffs, 1) == 1
    in_cluster = is_adjacent_prev ^ is_adjacent_next

    first_members = []
    i = 0
    N = len(in_cluster)
    while i < N:
        next_i = (i + 1) % N
        if in_cluster[i] and in_cluster[next_i]:
            # Found size-2 cluster → store first index (in original order)
            first_members.append(sorted_indices[i])
            i += 2  # skip next, already paired
        else:
            i += 1
    return np.array(first_members, dtype=int)

def contours_touch(c1, c2, image_shape, axis):
    """
    Check if two contours touch each other through PBC on a given axis.
    Return (touches: bool, shared_indices: list of indices in c1 that are touching c2).
    """
    h, w = image_shape
    p1 = c1[:, 0, :]
    p2 = c2[:, 0, :]

    shared_indices = []

    if axis == 'x':
        x1, y1 = p1[:, 0], p1[:, 1]
        x2, y2 = p2[:, 0], p2[:, 1]

        # Right1 (x == w-1) ↔ Left2 (x == 0)
        r1_idx = np.where(x1 == w - 1)[0]
        l2_y = p2[x2 == 0][:, 1]
        shared_indices += find_shared_indices(y1[r1_idx], l2_y, r1_idx)

        # Left1 (x == 0) ↔ Right2 (x == w-1)
        l1_idx = np.where(x1 == 0)[0]
        r2_y = p2[x2 == w - 1][:, 1]
        shared_indices += find_shared_indices(y1[l1_idx], r2_y, l1_idx)

    elif axis == 'y':
        x1, y1 = p1[:, 0], p1[:, 1]
        x2, y2 = p2[:, 0], p2[:, 1]

        # Bottom1 (y == h-1) ↔ Top2 (y == 0)
        b1_idx = np.where(y1 == h - 1)[0]
        t2_x = p2[y2 == 0][:, 0]
        shared_indices += find_shared_indices(x1[b1_idx], t2_x, b1_idx)

        # Top1 (y == 0) ↔ Bottom2 (y == h-1)
        t1_idx = np.where(y1 == 0)[0]
        b2_x = p2[y2 == h - 1][:, 0]
        shared_indices += find_shared_indices(x1[t1_idx], b2_x, t1_idx)

    return (len(shared_indices) > 0), sorted(set(shared_indices))

def pbc_close(p1, p2, w, h):
    dx = (p1[0] - p2[0]) % w
    dy = (p1[1] - p2[1]) % h
    if dx > w // 2: dx -= w
    if dy > h // 2: dy -= h
    return abs(dx) <= 1 and abs(dy) <= 1
    
def segments_touch(seg1, seg2, image_shape, forbidden_points=None):
    """
    Returns:
    - touch_any: True if any end of seg1 touches any end of seg2 (under PBC)
    - touch_oriented: True if end of seg1 touches start of seg2 (under PBC)
    """
    w, h = image_shape

    # Oriented touch: end of seg1 to start of seg2
    touch_oriented = pbc_close(seg1[-1, 0], seg2[0, 0], w, h)

    # If forbidden, override touch_oriented
    if forbidden_points is not None and (tuple(seg1[-1, 0]), tuple(seg2[0, 0])) in forbidden_points:
        touch_oriented = False

    # General touch: any end of seg1 to any end of seg2
    touch_any = touch_oriented or any(
        pbc_close(a, b, w, h)
        for a in (seg1[0, 0], seg1[-1, 0])
        for b in (seg2[0, 0], seg2[-1, 0])
    )

    return touch_any, touch_oriented

"""
%%%%%%%%%%%%%%%%%%%%%%%
-----------------------
    File Processing   
-----------------------
%%%%%%%%%%%%%%%%%%%%%%%
"""
def build_prefix_index_dict(fnames):
    """
    Splits .ovf files into two dictionaries:
    - m_index_map: {index: Path} for m*.ovf
    - other_index_map: {prefix: {index: Path}} for all other prefixes
    """
    pattern = re.compile(r"^(.*?)(\d{6})\.ovf$")
    m_index_map = {}
    other_index_map = defaultdict(dict)

    for f in fnames:
        m = pattern.match(f.name)
        if m:
            prefix, num = m.groups()
            index = int(num)
            if prefix == 'm':
                m_index_map[index] = f
            else:
                other_index_map[prefix][index] = f

    return m_index_map, other_index_map