# -*- encoding: utf-8 -*-
# striped version of maptrace.py from:
#  https://github.com/mzucker/maptrace
#

import sane
import cv2
import ezdxf
import math
import sys, re, os, argparse, heapq
from datetime import datetime
from collections import namedtuple, defaultdict
import numpy as np
from PIL import Image
from scipy import ndimage

######################################################################

DIR_RIGHT = 0
DIR_DOWN = 1
DIR_LEFT = 2
DIR_UP = 3
NEIGHBOR_OFFSET = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])
TURN_RIGHT = np.array([DIR_DOWN, DIR_LEFT, DIR_UP, DIR_RIGHT])
TURN_LEFT = np.array([DIR_UP, DIR_RIGHT, DIR_DOWN, DIR_LEFT])
VMAP_OFFSET = np.array([[-1, 0, 0], [0, 0, 1], [0, 0, 0], [0, -1, 1]])
DIAG_OFFSET = NEIGHBOR_OFFSET + NEIGHBOR_OFFSET[TURN_LEFT]
OPP_OFFSET = NEIGHBOR_OFFSET[TURN_LEFT]
CROSS_ELEMENT = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
BOX_ELEMENT = np.ones((3, 3), dtype=bool)

EdgeInfo = namedtuple("EdgeInfo", ["node0", "node1", "label0", "label1"])
EdgeRef = namedtuple("EdgeRef", ["edge_index", "opp_label", "step"])


class BoundaryRepresentation(object):
    def __init__(self):
        # list of nodes (points) or None for deleted
        self.node_list = []
        # list of sets of edge indices
        self.node_edges = []
        # list of point arrays (or empty for deleted edges)
        self.edge_list = []
        # list of EdgeInfo (or None for deleted edges)
        self.edge_infolist = []
        # map from point to node index
        self.node_lookup = dict()
        # map from EdgeInfo to edge index
        self.edge_lookup = dict()
        # map from label to list of list of EdgeRef
        self.label_lookup = defaultdict(list)

    def lookup_node(self, point, insert=False):
        key = tuple(map(float, point))
        if insert and key not in self.node_lookup:
            node_idx = len(self.node_list)
            self.node_list.append(point.copy())
            self.node_edges.append(set())
            self.node_lookup[key] = node_idx
        else:
            node_idx = self.node_lookup[key]
        return node_idx

    def add_edges(self, cur_label, contour_edges):
        edge_refs = []
        for opp_label, edge in contour_edges:
            assert cur_label != opp_label
            assert cur_label != 0
            label0 = min(cur_label, opp_label)
            label1 = max(cur_label, opp_label)
            if label0 == cur_label:
                step = 1
            else:
                step = -1
            edge_to_add = edge[::step]
            node0 = self.lookup_node(edge_to_add[0], insert=True)
            node1 = self.lookup_node(edge_to_add[-1], insert=True)
            edge_info = EdgeInfo(node0, node1, label0, label1)
            if edge_info in self.edge_lookup:
                edge_idx = self.edge_lookup[edge_info]
                stored_edge = self.edge_list[edge_idx]
                assert self.edge_infolist[edge_idx] == edge_info
                assert np.all(stored_edge == edge_to_add)
                assert edge_idx in self.node_edges[node0]
                assert edge_idx in self.node_edges[node1]
            else:
                edge_idx = len(self.edge_list)
                self.edge_list.append(edge_to_add)
                self.edge_infolist.append(edge_info)
                self.edge_lookup[edge_info] = edge_idx
                self.node_edges[node0].add(edge_idx)
                self.node_edges[node1].add(edge_idx)
            edge_refs.append(EdgeRef(edge_idx, opp_label, step))
        self.label_lookup[cur_label].append(edge_refs)


def angle_of_line(p_1, p_2):
    """gets the angle of a single line."""
    return math.atan2(p_2[1] - p_1[1], p_2[0] - p_1[0])


def calc_distance(p_1, p_2):
    """gets the distance between two points in 2D."""
    return math.hypot(p_1[0] - p_2[0], p_1[1] - p_2[1])


def rotate(origin, point, angle):
    """Rotate a point counterclockwise by a given angle around a given origin."""
    ox, oy = origin
    px, py = point
    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


def pack_rgb(rgb):
    orig_shape = None
    if isinstance(rgb, np.ndarray):
        assert rgb.shape[-1] == 3
        orig_shape = rgb.shape[:-1]
    else:
        assert len(rgb) == 3
        rgb = np.array(rgb)
    rgb = rgb.astype(int).reshape((-1, 3))
    packed = rgb[:, 0] << 16 | rgb[:, 1] << 8 | rgb[:, 2]
    if orig_shape is None:
        return packed
    else:
        return packed.reshape(orig_shape)


def get_mask(input_image):
    rgb = input_image
    alpha = None
    if rgb.mode == "LA" or (rgb.mode == "P" and "transparency" in rgb.info):
        rgb = rgb.convert("RGBA")
    if rgb.mode == "RGBA":
        alpha = np.array(rgb.split()[-1])
    rgb = rgb.convert("RGB")
    rgb = np.array(rgb)
    gray = rgb.max(axis=2)
    mask = gray > 64
    if alpha is not None:
        mask = mask | (alpha < 127)
    return mask


def get_labels_and_colors_outlined(mask):
    structure = CROSS_ELEMENT
    labels, num_labels = ndimage.label(mask, structure=structure)
    print("found {} labels".format(num_labels))
    unlabeled = ~mask
    print("computing areas... ")
    start = datetime.now()
    areas, bins = np.histogram(
        labels.flatten(), bins=num_labels, range=(1, num_labels + 1)
    )
    elapsed = (datetime.now() - start).total_seconds()
    print("finished computing areas in {} seconds.".format(elapsed))
    idx = np.hstack(([0], np.argsort(-areas) + 1))
    replace = np.zeros_like(idx)
    replace[idx] = range(len(idx))
    labels = replace[labels]
    areas = areas[idx[1:] - 1]
    colors = 255 * np.ones((num_labels + 1, 3), dtype=np.uint8)
    print("running DT... ")
    start = datetime.now()
    result = ndimage.distance_transform_edt(
        unlabeled, return_distances=False, return_indices=True
    )
    idx = result
    elapsed = (datetime.now() - start).total_seconds()
    print("ran DT in {} seconds".format(elapsed))
    labels = labels[tuple(idx)]
    assert not np.any(labels == 0)
    labels_big = np.zeros(
        (labels.shape[0] + 2, labels.shape[1] + 2), dtype=labels.dtype
    )
    labels_big[1:-1, 1:-1] = labels
    start = datetime.now()
    print("finding objects... ")
    slices = ndimage.find_objects(labels, num_labels)
    elapsed = (datetime.now() - start).total_seconds()
    print("found all objects in {} seconds".format(elapsed))
    slices_big = []
    for spair in slices:
        spair_big = []
        for s in spair:
            spair_big.append(slice(s.start, s.stop + 2))
        slices_big.append(tuple(spair_big))
    assert labels_big.min() == 0 and labels_big.max() == num_labels
    assert len(slices) == num_labels
    return num_labels, labels_big, slices_big, colors


def follow_contour(l_subrect, cur_label, startpoints, pos):
    start = pos
    cur_dir = DIR_RIGHT
    contour_info = []
    while True:
        ooffs = OPP_OFFSET[cur_dir]
        noffs = NEIGHBOR_OFFSET[cur_dir]
        doffs = DIAG_OFFSET[cur_dir]
        neighbor = tuple(pos + noffs)
        diag = tuple(pos + doffs)
        opp = tuple(pos + ooffs)
        assert l_subrect[pos] == cur_label
        assert l_subrect[opp] != cur_label
        contour_info.append(pos + (cur_dir, l_subrect[opp]))
        startpoints[pos] = False
        if l_subrect[neighbor] != cur_label:
            cur_dir = TURN_RIGHT[cur_dir]
        elif l_subrect[diag] == cur_label:
            pos = diag
            cur_dir = TURN_LEFT[cur_dir]
        else:
            pos = neighbor
        if pos == start and cur_dir == DIR_RIGHT:
            break
    n = len(contour_info)
    contour_info = np.array(contour_info)
    clabels = contour_info[:, 3]
    # set of unique labels for this contour
    opp_label_set = set(clabels)
    assert cur_label not in opp_label_set
    # if multiple labels and one wraps around, correct this
    if len(opp_label_set) > 1 and clabels[0] == clabels[-1]:
        idx = np.nonzero(clabels != clabels[0])[0][0]
        perm = np.hstack((np.arange(idx, n), np.arange(idx)))
        contour_info = contour_info[perm]
        clabels = contour_info[:, 3]
    # make sure no wraparound
    assert len(opp_label_set) == 1 or clabels[0] != clabels[-1]
    # apply offset to get contour points
    cpoints = contour_info[:, :2].astype(np.float32)
    cdirs = contour_info[:, 2]
    cpoints += 0.5 * (OPP_OFFSET[cdirs] - NEIGHBOR_OFFSET[cdirs] + 1)
    # put points in xy format
    cpoints = cpoints[:, ::-1]
    if len(opp_label_set) == 1:
        idx = np.arange(len(cpoints))
        xyi = zip(cpoints[:, 0], cpoints[:, 1], idx)
        imin = min(xyi)
        i = imin[-1]
        cpoints = np.vstack((cpoints[i:], cpoints[:i]))
        assert np.all(clabels == clabels[0])
    return cpoints, clabels


def split_contour(cpoints, clabels):
    edges = []
    shifted = np.hstack(([-1], clabels[:-1]))
    istart = np.nonzero(clabels - shifted)[0]
    iend = np.hstack((istart[1:], len(clabels)))
    for start, end in zip(istart, iend):
        assert start == 0 or clabels[start] != clabels[start - 1]
        assert clabels[end - 1] == clabels[start]
        opp_label = clabels[start]
        if end < len(cpoints):
            edge = cpoints[start : end + 1]
        else:
            edge = np.vstack((cpoints[start:end], cpoints[0]))
        edges.append((opp_label, edge))
        start = end
    return edges


def _simplify_r(p0, edge, output_list):
    assert np.all(output_list[-1][-1] == p0)
    assert not np.all(edge[0] == p0)
    p1 = edge[-1]
    if len(edge) == 1:
        output_list.append(edge)
        return
    l = np.cross([p0[0], p0[1], 1], [p1[0], p1[1], 1])
    n = l[:2]
    w = np.linalg.norm(n)
    if w == 0:
        dist = np.linalg.norm(edge - p0, axis=1)
        idx = dist.argmax()
        dmax = np.inf
    else:
        l /= w
        dist = np.abs(np.dot(edge, l[:2]) + l[2])
        idx = dist.argmax()
        dmax = dist[idx]
    if dmax < 1.42:
        output_list.append(np.array([p1]))
    elif len(edge) > 3:
        _simplify_r(p0, edge[: idx + 1], output_list)
        _simplify_r(edge[idx], edge[idx + 1 :], output_list)
    else:
        output_list.append(edge)


def simplify(edge):
    if not len(edge):
        return edge
    p0 = edge[0]
    output_list = [edge[[0]]]
    _simplify_r(p0, edge[1:], output_list)
    return np.vstack(tuple(output_list))


def build_brep(num_labels, labels, slices, colors):
    brep = BoundaryRepresentation()
    label_range = range(1, num_labels + 1)
    print("building boundary representation...")
    # for each object
    for cur_label, (yslc, xslc) in zip(label_range, slices):
        p0 = (xslc.start - 1, yslc.start - 1)
        # extract sub-rectangle for this label
        l_subrect = labels[yslc, xslc]
        # get binary map of potential start points for contour in
        # rightward direction
        mask_subrect = l_subrect == cur_label
        mask_shifted_down = np.vstack(
            (np.zeros_like(mask_subrect[0].reshape(1, -1)), mask_subrect[:-1])
        )
        startpoints = mask_subrect & ~mask_shifted_down
        print(
            "  processing label {}/{} with area {}".format(
                cur_label, num_labels, (l_subrect == cur_label).sum()
            )
        )
        # while there are candidate locations to start at
        while np.any(startpoints):
            # get the first one
            i, j = np.nonzero(startpoints)
            pos = (i[0], j[0])
            # extract points and adjacent labels along contour,
            # this modifies startpoints
            cpoints, clabels = follow_contour(l_subrect, cur_label, startpoints, pos)
            cpoints += p0
            # split contour into (opp_label, points) pairs
            contour_edges = split_contour(cpoints, clabels)
            # add them to our boundary representation
            brep.add_edges(cur_label, contour_edges)
    simplified = False
    print("simplifying edges...")
    brep.edge_list = [simplify(edge) for edge in brep.edge_list]
    simplified = True
    return brep


def output_dxf(orig_shape, brep, colors, output_file):
    cpacked = pack_rgb(colors.astype(int))
    cset = set(cpacked)
    lsets = []
    for c in cset:
        idx = np.nonzero(cpacked == c)[0]
        if 1 in idx:
            lsets.insert(0, idx)
        else:
            lsets.append(idx)

    assert 1 in lsets[0]
    longest_line = [0, 0, (0, 0), (0, 0)]
    lines = []
    for lset in lsets:
        for cur_label in lset:
            if cur_label not in brep.label_lookup:
                continue
            contours = brep.label_lookup[cur_label]
            for i, contour in enumerate(contours):
                if i == 0:
                    continue
                last = (0, 0)
                for j, (edge_idx, _, step) in enumerate(contour):
                    edge = brep.edge_list[edge_idx][::step]
                    iedge = edge.astype(int)
                    if np.all(edge == iedge):
                        pprev = iedge[0]
                        for pt in iedge[1:]:
                            dist = calc_distance(pprev, pt)
                            if dist > 50 and dist > longest_line[0]:
                                angle = abs(angle_of_line(pt, pprev) * 180 / math.pi)
                                while angle > 90:
                                    angle -= 90
                                if angle < 20:
                                    longest_line = [dist, angle, pprev, pt]
                            lines.append((pprev / (12, -12), pt / (12, -12)))
                            pprev = pt
                    else:
                        if j == 0:
                            last = pprev
                        for pt in edge[1:]:
                            lines.append((last / (12, -12), pt / (12, -12)))
                            last = pt

    # rotate
    if longest_line[0] > 0:
        angle = -angle_of_line(longest_line[2], longest_line[3]) + math.pi / 2
        angle = -longest_line[1] / 180 * math.pi
        print(f"rotating image: {angle * 180 / math.pi}")
        for line_n, line in enumerate(lines):
            start = rotate((0, 0), line[0], angle)
            end = rotate((0, 0), line[1], angle)
            lines[line_n] = (start, end)

    # size
    min_x = lines[0][0][0]
    min_y = lines[0][0][1]
    max_x = lines[0][0][0]
    max_y = lines[0][0][1]
    for line in lines:
        start = (line[0][0], line[0][1])
        end = (line[1][0], line[1][1])
        min_x = min(min_x, min(start[0], end[0]))
        min_y = min(min_y, min(start[1], end[1]))
        max_x = max(max_x, max(start[0], end[0]))
        max_y = max(max_y, max(start[1], end[1]))

    # move
    for line_n, line in enumerate(lines):
        start = (line[0][0] - min_x, line[0][1] - min_y)
        end = (line[1][0] - min_x, line[1][1] - min_y)
        lines[line_n] = (start, end)

    # write to dxf
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    doc.units = ezdxf.units.MM
    for line in lines:
        msp.add_line(line[0], line[1])

    print("wrote", output_file)
    for vport in doc.viewports.get_config("*Active"):
        vport.dxf.grid_on = True
        vport.dxf.center = (200, 200)
    doc.saveas(output_file)


def list_scanners():
    ver = sane.init()
    print("SANE version:", ver)
    print("searching for scanner devices, please wait...")
    devices = sane.get_devices()
    print("scanner devices:")
    for device in devices:
        print(f"  '{device[0]}' : {' - '.join(device[1:])}")


def scan_image(scanner_name):
    ver = sane.init()
    print("SANE version:", ver)
    # print('device:', devices[-1][0])
    # dev = sane.open(devices[-1][0])
    dev = sane.open(scanner_name)
    params = dev.get_parameters()
    try:
        dev.depth = 8
    except:
        print("Cannot set depth, defaulting to %d" % params[3])
    try:
        dev.mode = "color"
    except:
        print("Cannot set mode, defaulting to %s" % params[0])
    try:
        dev.br_x = 320.0
        dev.br_y = 240.0
    except:
        print("Cannot set scan area, using default")
    params = dev.get_parameters()
    print("Device parameters:", params, "\n Resolutions %d " % (dev.resolution))
    dev.start()
    im = dev.snap()
    #im.save("scann.png")
    return im


def edge_detection(imagedata=None, filename=None):
    if imagedata is not None:
        img = np.array(imagedata)
    else:
        img = cv2.imread(filename)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (13, 13), 0)
    ret, thresh = cv2.threshold(img_blur, 150, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(image=thresh, threshold1=100, threshold2=200)
    return Image.fromarray(cv2.cvtColor(~edges, cv2.COLOR_BGR2RGB))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--image", help="image input file", type=str, default=None
    )
    parser.add_argument(
        "-o", "--output", help="dxf output file", type=str, default="scann.dxf"
    )
    parser.add_argument("-s", "--scanner", help="scanner name", type=str, default=None)
    parser.add_argument(
        "-l", "--list", help="list scanner devices", action="store_true"
    )
    args = parser.parse_args()

    if args.list:
        list_scanners()
        exit(0)

    if args.scanner:
        imagedata = scan_image(args.scanner)
        input_image = edge_detection(imagedata=imagedata)
    elif args.image:
        input_image = edge_detection(filename=args.image)
    else:
        print("ERROR: please select an image or scanner")
        exit(1)

    mask = get_mask(input_image)
    num_labels, labels, slices, colors = get_labels_and_colors_outlined(mask)
    brep = build_brep(num_labels, labels, slices, colors)
    output_dxf(labels.shape, brep, colors, args.output)


if __name__ == "__main__":
    main()
