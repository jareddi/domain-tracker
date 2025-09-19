from .tracker_properties import *
from .tracker_mag import *
from .tracker_dilate import *

"""
%%%%%%%%%%%%%%%%%%%%%%%
-----------------------
   Contour Functions
-----------------------
%%%%%%%%%%%%%%%%%%%%%%%
"""

def find_contours(arr, layer=None):
    """
    Process a single image to extract contour information near zero.
    """
    # Threshold above 0, handling for layer selection
    if layer == None:
        arr_bool = np.any((arr) > 0,axis=2).T
    else:
        arr_bool = ((arr[:,:,layer]) > 0).T

    binary = (255*arr_bool).astype(np.uint8)
    _, th = cv.threshold(binary,127,255,0,)
    contours, hierarchy = cv.findContours(th, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    #print(f"# of contours: {len(contours)}")

    return contours, hierarchy

def find_contour_networks(contours, image_shape):
    """
    Group contours into PBC-connected networks.
    Also return shared ranges per connected pair, using local group indices.
    """
    n = len(contours)
    graph = defaultdict(set)
    shared_indices_graph = defaultdict(list)  # (i, j) -> list of indices in contour i that touch j

    # Step 1: Build graph and shared range lookup using global indices
    for i in range(n):
        for j in range(i, n):
            for axis in ['x', 'y']:
                touches, indices_i = contours_touch(contours[i], contours[j], image_shape, axis)
                if touches:
                    shared_indices_graph[(i, j)] += indices_i
                    graph[i].add(j)
                    if i != j:
                        # Get reverse indices (in j) too
                        _, indices_j = contours_touch(contours[j], contours[i], image_shape, axis)
                        shared_indices_graph[(j, i)] += indices_j
                        # add nodes to graph
                        graph[j].add(i)

    # Step 2: Find connected components (groups of contours)
    visited = set()
    groups = []
    for i in range(n):
        if i not in visited:
            queue = deque([i])
            group = []
            while queue:
                node = queue.popleft()
                if node not in visited:
                    visited.add(node)
                    group.append(node)
                    queue.extend(graph[node])
            groups.append(group)

    # Step 3: Build grouped contours
    grouped_contours = [[contours[i] for i in group] for group in groups]

    # Step 4: Build per-group shared indices using local indices
    grouped_shared_indices = []
    for group in groups:
        group_map = {}
        for idx_in_group, global_idx in enumerate(group):
            all_indices = []
            for other in group:
                all_indices += shared_indices_graph.get((global_idx, other), [])
            # Remove duplicates and sort
            group_map[idx_in_group] = sorted(set(all_indices))
        grouped_shared_indices.append(group_map)

    return grouped_contours, grouped_shared_indices

"""
%%%%%%%%%%%%%%%%%%%%%%%
-----------------------
   Segment Functions
-----------------------
%%%%%%%%%%%%%%%%%%%%%%%
"""

def find_contour_segments(contour, shared_indices, image_shape):
    """
    Split contour into segments bounded by gaps.
    """
    if len(contour) < 2:
        return [], shared_indices
    if len(contour) == 2:
        return [contour], shared_indices

    # clean duplicate points and remove shared edge indices
    contour, shared_indices = clean_contour(contour,shared_indices,image_shape)

    contour = np.squeeze(contour)  # Shape: (N, 2)
    N = len(contour)

    # Compute difference between each pair of consecutive points (with wraparound)
    diffs = np.abs(contour - np.roll(contour, -1, axis=0))  # Shape: (N, 2)
    breaks = np.any(diffs > 1, axis=1)  # True where gap exists between i and i+1

    # Find indices where breaks occur
    break_indices = np.where(breaks)[0]
    # Find pairs of shared indices which need to be separated
    cut_indices = find_unsegmented_pairs(shared_indices, N)
    # Mark these pairs as not to be reconnected
    forbidden_pairs = set()
    for i in cut_indices:
        a = tuple(contour[i])
        b = tuple(contour[(i + 1) % N])
        forbidden_pairs.add((a, b))
        forbidden_pairs.add((b, a))
    # Merge these indices
    segment_cuts = np.sort(np.concatenate([break_indices, cut_indices]))

    # Build segments using split points
    segments = []
    start = 0
    for idx in segment_cuts:
        end = idx + 1
        if end - start > 1:
            segments.append(contour[start:end])
        start = end

    # Final segment: wraparound
    if start < N:
        tail = np.concatenate((contour[start:], contour[:1]), axis=0)
        # Check if this final segment connects to the first point
        if np.all(np.abs(tail[-1] - tail[-2]) <= 1):
            segments.append(tail)
        elif len(contour[start:]) > 1:
            segments.append(contour[start:])

    return [seg[:, np.newaxis, :] for seg in segments], forbidden_pairs

def find_segment_networks(segments, image_shape):
    n = len(segments)
    graph = defaultdict(set)

    for i in range(n):
        for j in range(i + 1, n):
            if segments_touch(segments[i], segments[j], image_shape)[0]:
                graph[i].add(j)
                graph[j].add(i)

    visited = set()
    groups = []

    for i in range(n):
        if i not in visited:
            queue = deque([i])
            group = []
            while queue:
                node = queue.popleft()
                if node not in visited:
                    visited.add(node)
                    group.append(node)
                    queue.extend(graph[node])
            groups.append([segments[i] for i in group])
    return groups

def stitch_segment_group(segments, image_shape, forbidden_points=None):
    used = set()
    contour = segments[0]
    used.add(0)

    while len(used) < len(segments):
        last_pt = contour[-1, 0]
        found = False
        for i, seg in enumerate(segments):
            if i in used:
                continue
            touch_any, touch_oriented_0 = segments_touch(contour, seg, image_shape, forbidden_points)
            if touch_any:
                if not touch_oriented_0:
                    seg = np.flip(seg, axis=0)
                    _, touch_oriented_1 = segments_touch(contour, seg, image_shape, forbidden_points)
                    if not touch_oriented_1:
                        contour = np.flip(contour, axis=0)
                        last_pt = contour[-1,0]
                        _, touch_oriented_2 = segments_touch(contour, seg, image_shape, forbidden_points)
                        if not touch_oriented_2:
                            seg = np.flip(seg, axis=0)
                            _, touch_oriented_3 = segments_touch(contour, seg, image_shape, forbidden_points)
                            if not touch_oriented_3:
                                continue
                shifted = shift_segment(last_pt, seg, image_shape)
                contour = np.concatenate([contour, shifted], axis=0)
                used.add(i)
                found = True
                break
        if not found:
            break  # stitched loop completed

    return contour

def process_segments_from_contours(contours, shared_indices, image_shape):
    segments = []
    all_forbidden_pairs = set()
    for i,contour in enumerate(contours):
        segs, forbidden_pairs = find_contour_segments(contour, shared_indices[i], image_shape)
        segments.extend(segs)
        all_forbidden_pairs.update(forbidden_pairs)

    networks = find_segment_networks(segments, image_shape)

    stitched_contours = []
    for net in networks:
        stitched = stitch_segment_group(net, image_shape)
        stitched_contours.append(stitched)

    return stitched_contours

"""
%%%%%%%%%%%%%%%%%%%%%%%
-----------------------
      All-in-one
-----------------------
%%%%%%%%%%%%%%%%%%%%%%%
"""

def process_contours(mz,layer,image_shape,theta_arr=None,gamma_arr=None,dil_dist=100,dil_mz_thresh=-0.5,polarity=1,debug=False):
    """
    Creates a dictionary of domains using mz. This version stores minimal information for storage and processing efficiency.
    """
    contours, _ = find_contours(polarity*mz, layer=layer)
    groups, groups_shared_indices = find_contour_networks(contours, image_shape)
    domains_z = {}
    n_bps = 0
    for g,group in enumerate(groups):
        domains_z[g] = {}
        segments = process_segments_from_contours(group,groups_shared_indices[g],image_shape)
        if layer != None:
            domains_z[g]['bps'] = []
            for segment in segments:
                bp = segment_bps(segment,gamma_arr,layer,image_shape)
                domains_z[g]['bps'] += bp
            n_bps += len(domains_z[g]['bps'])
        if len(group) > 1:
            contour = remove_duplicate_points(np.concatenate(segments, axis=0))
        else:
            contour = group[0]
        domains_z[g]['core'] = contours_mask(group,image_shape)
        domains_z[g]['shell'] = contours_mask(wrap_contours(segments,image_shape),image_shape,closed=False)
        domains_z[g]['properties'] = get_contour_properties(contour)
        if layer != None:
            domains_z[g]['gammas'] = get_values_along_contour(gamma_arr, contour, layer)
            domains_z[g]['w'] = winding_number(get_values_along_contour(theta_arr, contour, layer))

    # dilate core mask for assigning empty space to domains
    if layer != None:
        dilate_domain_pbc(domains_z, polarity*mz[:,:,layer], max_dist=dil_dist, mz_thresh=dil_mz_thresh)
    
    if layer != None and debug:
        print(f"Layer {layer}: {len(groups)} domains, {n_bps} bloch points")
    return domains_z