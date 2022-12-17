"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import collections
import contextlib
import os
import random
from typing import List, Tuple

import numpy as np
import cv2
import math
from scipy.ndimage import label

from .utils import get_intersection_bbox, Rect

""" auxiliary functions """
# unwarp corodinates
def warpCoord(Minv, pt):
    out = np.matmul(Minv, (pt[0], pt[1], 1))
    return np.array([out[0]/out[2], out[1]/out[2]])
""" end of auxiliary functions """


def getDetBoxes_core(
    textmap, linkmap, text_threshold, link_threshold, low_text, 
    estimate_num_chars=False, vertical=False
    ):
    DEBUG_FOLDER = f"{os.getcwd().replace('/src', '')}/debug"

    # prepare data
    linkmap = linkmap.copy()
    textmap = textmap.copy()
    img_h, img_w = textmap.shape

    """ labeling method """
    ret, text_score = cv2.threshold(textmap, low_text, 1, 0)
    ret, link_score = cv2.threshold(linkmap, link_threshold, 1, 0)

    if vertical:
        vertical_filter = VerticalScoreFilter(DEBUG_FOLDER)
        text_score, link_score = vertical_filter.filter_score_images(text_score, link_score)

    text_score_comb = np.clip(text_score + link_score, 0, 1)
    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(text_score_comb.astype(np.uint8), connectivity=4)

    det = []
    mapper = []
    for k in range(1,nLabels):
        # size filtering
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10: continue

        # thresholding
        if np.max(textmap[labels==k]) < text_threshold: continue

        # make segmentation map
        segmap = np.zeros(textmap.shape, dtype=np.uint8)
        segmap[labels==k] = 255
        if estimate_num_chars:
            _, character_locs = cv2.threshold((textmap - linkmap) * segmap /255., text_threshold, 1, 0)
            _, n_chars = label(character_locs)
            mapper.append(n_chars)
        else:
            mapper.append(k)
        segmap[np.logical_and(link_score==1, text_score==0)] = 0   # remove link area
        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
        sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
        # boundary check
        if sx < 0 : sx = 0
        if sy < 0 : sy = 0
        if ex >= img_w: ex = img_w
        if ey >= img_h: ey = img_h
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1 + niter, 1 + niter))
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)

        # make box
        np_contours = np.roll(np.array(np.where(segmap!=0)),1,axis=0).transpose().reshape(-1,2)
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)

        # align diamond-shape
        w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        box_ratio = max(w, h) / (min(w, h) + 1e-5)
        if abs(1 - box_ratio) <= 0.1:
            l, r = min(np_contours[:,0]), max(np_contours[:,0])
            t, b = min(np_contours[:,1]), max(np_contours[:,1])
            box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

        # make clock-wise order
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4-startidx, 0)
        box = np.array(box)

        det.append(box)

    return det, labels, mapper

def getPoly_core(boxes, labels, mapper, linkmap):
    # configs
    num_cp = 5
    max_len_ratio = 0.7
    expand_ratio = 1.45
    max_r = 2.0
    step_r = 0.2

    polys = []  
    for k, box in enumerate(boxes):
        # size filter for small instance
        w, h = int(np.linalg.norm(box[0] - box[1]) + 1), int(np.linalg.norm(box[1] - box[2]) + 1)
        if w < 10 or h < 10:
            polys.append(None); continue

        # warp image
        tar = np.float32([[0,0],[w,0],[w,h],[0,h]])
        M = cv2.getPerspectiveTransform(box, tar)
        word_label = cv2.warpPerspective(labels, M, (w, h), flags=cv2.INTER_NEAREST)
        try:
            Minv = np.linalg.inv(M)
        except:
            polys.append(None); continue

        # binarization for selected label
        cur_label = mapper[k]
        word_label[word_label != cur_label] = 0
        word_label[word_label > 0] = 1

        """ Polygon generation """
        # find top/bottom contours
        cp = []
        max_len = -1
        for i in range(w):
            region = np.where(word_label[:,i] != 0)[0]
            if len(region) < 2 : continue
            cp.append((i, region[0], region[-1]))
            length = region[-1] - region[0] + 1
            if length > max_len: max_len = length

        # pass if max_len is similar to h
        if h * max_len_ratio < max_len:
            polys.append(None); continue

        # get pivot points with fixed length
        tot_seg = num_cp * 2 + 1
        seg_w = w / tot_seg     # segment width
        pp = [None] * num_cp    # init pivot points
        cp_section = [[0, 0]] * tot_seg
        seg_height = [0] * num_cp
        seg_num = 0
        num_sec = 0
        prev_h = -1
        for i in range(0,len(cp)):
            (x, sy, ey) = cp[i]
            if (seg_num + 1) * seg_w <= x and seg_num <= tot_seg:
                # average previous segment
                if num_sec == 0: break
                cp_section[seg_num] = [cp_section[seg_num][0] / num_sec, cp_section[seg_num][1] / num_sec]
                num_sec = 0

                # reset variables
                seg_num += 1
                prev_h = -1

            # accumulate center points
            cy = (sy + ey) * 0.5
            cur_h = ey - sy + 1
            cp_section[seg_num] = [cp_section[seg_num][0] + x, cp_section[seg_num][1] + cy]
            num_sec += 1

            if seg_num % 2 == 0: continue # No polygon area

            if prev_h < cur_h:
                pp[int((seg_num - 1)/2)] = (x, cy)
                seg_height[int((seg_num - 1)/2)] = cur_h
                prev_h = cur_h

        # processing last segment
        if num_sec != 0:
            cp_section[-1] = [cp_section[-1][0] / num_sec, cp_section[-1][1] / num_sec]

        # pass if num of pivots is not sufficient or segment width is smaller than character height 
        if None in pp or seg_w < np.max(seg_height) * 0.25:
            polys.append(None); continue

        # calc median maximum of pivot points
        half_char_h = np.median(seg_height) * expand_ratio / 2

        # calc gradiant and apply to make horizontal pivots
        new_pp = []
        for i, (x, cy) in enumerate(pp):
            dx = cp_section[i * 2 + 2][0] - cp_section[i * 2][0]
            dy = cp_section[i * 2 + 2][1] - cp_section[i * 2][1]
            if dx == 0:     # gradient if zero
                new_pp.append([x, cy - half_char_h, x, cy + half_char_h])
                continue
            rad = - math.atan2(dy, dx)
            c, s = half_char_h * math.cos(rad), half_char_h * math.sin(rad)
            new_pp.append([x - s, cy - c, x + s, cy + c])

        # get edge points to cover character heatmaps
        isSppFound, isEppFound = False, False
        grad_s = (pp[1][1] - pp[0][1]) / (pp[1][0] - pp[0][0]) + (pp[2][1] - pp[1][1]) / (pp[2][0] - pp[1][0])
        grad_e = (pp[-2][1] - pp[-1][1]) / (pp[-2][0] - pp[-1][0]) + (pp[-3][1] - pp[-2][1]) / (pp[-3][0] - pp[-2][0])
        for r in np.arange(0.5, max_r, step_r):
            dx = 2 * half_char_h * r
            if not isSppFound:
                line_img = np.zeros(word_label.shape, dtype=np.uint8)
                dy = grad_s * dx
                p = np.array(new_pp[0]) - np.array([dx, dy, dx, dy])
                cv2.line(line_img, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), 1, thickness=1)
                if np.sum(np.logical_and(word_label, line_img)) == 0 or r + 2 * step_r >= max_r:
                    spp = p
                    isSppFound = True
            if not isEppFound:
                line_img = np.zeros(word_label.shape, dtype=np.uint8)
                dy = grad_e * dx
                p = np.array(new_pp[-1]) + np.array([dx, dy, dx, dy])
                cv2.line(line_img, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), 1, thickness=1)
                if np.sum(np.logical_and(word_label, line_img)) == 0 or r + 2 * step_r >= max_r:
                    epp = p
                    isEppFound = True
            if isSppFound and isEppFound:
                break

        # pass if boundary of polygon is not found
        if not (isSppFound and isEppFound):
            polys.append(None); continue

        # make final polygon
        poly = []
        poly.append(warpCoord(Minv, (spp[0], spp[1])))
        for p in new_pp:
            poly.append(warpCoord(Minv, (p[0], p[1])))
        poly.append(warpCoord(Minv, (epp[0], epp[1])))
        poly.append(warpCoord(Minv, (epp[2], epp[3])))
        for p in reversed(new_pp):
            poly.append(warpCoord(Minv, (p[2], p[3])))
        poly.append(warpCoord(Minv, (spp[2], spp[3])))

        # add to final result
        polys.append(np.array(poly))

    return polys

def getDetBoxes(
    textmap, linkmap, text_threshold, link_threshold, low_text, 
    poly=False, estimate_num_chars=False, vertical=False
    ):
    if poly and estimate_num_chars:
        raise Exception("Estimating the number of characters not currently supported with poly.")
    boxes, labels, mapper = getDetBoxes_core(textmap, linkmap, text_threshold, link_threshold, low_text, estimate_num_chars, vertical)

    if poly:
        polys = getPoly_core(boxes, labels, mapper, linkmap)
    else:
        polys = [None] * len(boxes)

    return boxes, polys, mapper

def adjustResultCoordinates(polys, ratio_w, ratio_h, ratio_net = 2):
    if len(polys) > 0:
        polys = np.array(polys)
        for k in range(len(polys)):
            if polys[k] is not None:
                polys[k] *= (ratio_w * ratio_net, ratio_h * ratio_net)
    return polys


class VerticalScoreFilter(object):

    def __init__(self, debug_dir_path: str):
        self._debug_dir_path = debug_dir_path

    def filter_score_images(self, text_score: np.ndarray, link_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # 1. Size filtering for text_score
        ts_nlabels, ts_labels, ts_stats, ts_centroids = cv2.connectedComponentsWithStats(
            text_score.astype(np.uint8), connectivity=4
        )
        for i in range(1, ts_nlabels):
            x, y, width, height, size = ts_stats[i]
            if size < 50:
                text_score[ts_labels == i, ] = 0

        # 2. Detect external char rectangles
        external_char_rects = []
        for i in range(1, ts_nlabels):
            x, y, width, height, size = ts_stats[i]
            if size < 50:
                continue

            char_component = text_score.astype(np.uint8).copy()
            char_component[ts_labels == i, ] = 255
            char_component[ts_labels != i, ] = 0

            contours, hierarchy = cv2.findContours(char_component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                bx, by, bw, bh = cv2.boundingRect(contour)
                external_char_rects.append(
                    Rect(
                        left=bx,
                        top=by,
                        right=bx + bw,
                        bottom=by + bh
                    )
                )

        # 3. Size and shape filtering for link_score
        lk_nlabels, lk_labels, lk_stats, lk_centroids = cv2.connectedComponentsWithStats(
            link_score.astype(np.uint8), connectivity=4
        )
        for i in range(1, lk_nlabels):
            x, y, width, height, size = lk_stats[i]
            if (
                size < 20 or
                width > 1.5 * height
            ):
                link_score[lk_labels == i, ] = 0

        # 4. Detect external link rectangles
        contours, hierarchy = cv2.findContours(link_score.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        external_link_rects = []
        for i in range(0, len(contours)):
            contour = contours[i]
            if (
                len(contour) == 0 or
                cv2.contourArea(contour) < 20
            ):
                continue

            bx, by, bw, bh = cv2.boundingRect(contour)
            external_link_rects.append(
                Rect(
                    left=bx,
                    top=by,
                    right=bx + bw,
                    bottom=by + bh
                )
            )

        # 5. Get mapping between overlapped text_score and link_score for external rects
        overlapped_mapping = collections.defaultdict(list)
        for li, link_rect in enumerate(external_link_rects):
            for text_rect in external_char_rects:
                iou = Rect.calc_iou(link_rect, text_rect)
                if iou == 0:
                    continue

                link_score_exists_in_landscape_text_score = (
                    text_rect.width / text_rect.height >= 1.5 and
                    link_rect.is_inside_of(text_rect)
                )
                if (
                    not link_score_exists_in_landscape_text_score and
                    text_rect.left <= link_rect.left <= link_rect.right <= text_rect.right
                ):
                    continue

                # * shape filtering
                intersection_bbox = get_intersection_bbox(link_rect, text_rect)
                intersection_rect = Rect(*intersection_bbox)

                overlapped_edges = Rect.get_overlapped_edges(text_rect, link_rect)
                is_horizontal_candidate = False
                if len(overlapped_edges) == 1 and overlapped_edges[0] in ['l', 'r']:
                    is_horizontal_candidate = True

                if (
                    not is_horizontal_candidate and
                    intersection_rect.width >= intersection_rect.height * 2
                ):
                    continue

                # * horizontal occupation filtering
                # tex_line = (text_rect.left, text_rect.right)
                # its_line = (intersection_rect.left, intersection_rect.right)
                # line_iou = get_horizontal_iou_1_dim(tex_line, its_line)
                # if line_iou >= 0.5:
                #     continue

                overlapped_mapping[li].append(text_rect)

        # 6. Detect horizontal candidates for text_score
        horizontal_detector = HorizontalCandidateDetectorForTextScore(self._debug_dir_path)
        horizontal_candidate_char_rects = horizontal_detector.detect(external_char_rects, text_score)

        # --- Debugging ---
        debug_canvas = cv2.cvtColor(
            np.clip(text_score * 255, a_min=0, a_max=255).astype(np.uint8).copy(), cv2.COLOR_GRAY2BGR
        )
        link_color = (0, 255, 0)
        for li, overlapped_text_rects in overlapped_mapping.items():
            link_rect = external_link_rects[li]
            cv2.rectangle(debug_canvas, link_rect.lt, link_rect.rb, link_color, 1)

            color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
            for text_rect in overlapped_text_rects:
                cv2.rectangle(debug_canvas, text_rect.lt, text_rect.rb, color, 1)

        horizontal_candidate_color = (0, 0, 255)
        for hcc_rect in horizontal_candidate_char_rects:
            cv2.rectangle(debug_canvas, hcc_rect.lt, hcc_rect.rb, horizontal_candidate_color, 1)

        cv2.imwrite(f"{self._debug_dir_path}/horizontal_connection_candidates.jpg", debug_canvas)
        # -----------------

        # 7. Delete link regions of horizontal candidates
        lk_nlabels, lk_labels, lk_stats, lk_centroids = cv2.connectedComponentsWithStats(
            link_score.astype(np.uint8), connectivity=4
        )
        for i in range(1, lk_nlabels):
            x, y, width, height, size = lk_stats[i]
            if (
                size < 30 or
                width > 1.5 * height
            ):
                link_score[lk_labels == i, ] = 0

            link_component = link_score.astype(np.uint8).copy()
            link_component[lk_labels == i, ] = 255
            link_component[lk_labels != i, ] = 0

            for li in overlapped_mapping.keys():
                link_rect = external_link_rects[li]
                li_left, li_top, li_right, li_bottom = link_rect.bbox

                white_area = cv2.countNonZero(link_component[li_top: li_bottom, li_left: li_right])
                if white_area > 0:
                    link_score[lk_labels == i, ] = 0
                    break

        # 8. Split text regions of horizontal connected candidates
        for hc_rect in horizontal_candidate_char_rects:
            x_boundary_in_rect = hc_rect.attribute.get('x_boundary_in_rect')
            if not x_boundary_in_rect:
                continue

            x_boundary_left = hc_rect.left + x_boundary_in_rect
            x_boundary_right = x_boundary_left + 1
            text_score[hc_rect.top: hc_rect.bottom, x_boundary_left: x_boundary_right] = 0

        cv2.imwrite(
            f"{self._debug_dir_path}/link_score_for_vertical.jpg",
            np.clip(link_score * 255, a_min=0, a_max=255).astype(np.uint8)
        )
        cv2.imwrite(
            f"{self._debug_dir_path}/text_score_for_vertical.jpg",
            np.clip(text_score * 255, a_min=0, a_max=255).astype(np.uint8)
        )
        return text_score, link_score


class HorizontalCandidateDetectorForTextScore(object):

    def __init__(self, debug_dir_path: str):
        self._debug_dir_path = debug_dir_path

    def detect(
        self,
        external_char_rects: List[Rect],
        text_score: np.ndarray
    ) -> List[Rect]:
        horizontal_candidates = []
        for e_idx, external_char_rect in enumerate(external_char_rects):
            text_domain = self._create_text_domain(text_score, external_char_rect)
            connected_components = self.get_vertical_histogram_projection(text_domain, subtract_base_projection=True)
            # cv2.imwrite(f"{self._debug_dir_path}/text_domain{e_idx}.jpg", text_domain)
            # cv2.imwrite(f"{self._debug_dir_path}/connected_components{e_idx}.jpg", connected_components)

            number_of_labels, markers, stats, centroids = cv2.connectedComponentsWithStats(connected_components)
            interested_stats = []
            for nl_idx in range(1, number_of_labels):
                stat = stats[nl_idx].tolist()
                _x, _y, width, height, size = stat
                # Remove small stat
                if width <= 2 or height <= 2 or size <= 5:
                    continue

                interested_stats.append(stat)
                # target_component = self._get_target_component(connected_components, markers, nl_idx)
                # cv2.imwrite(f"{self._debug_dir_path}/target_component{e_idx}_{nl_idx}.jpg", target_component)

            # Shape filter (Remove landscape "hyoutan" shape image)
            if len(interested_stats) == 2:
                min_width = 999
                max_width = -1
                min_height = 999
                max_height = -1
                sorted_interested_stats = sorted(interested_stats, key=lambda x: x[0])
                for interested_stat in sorted_interested_stats:
                    # stat: x, y, width, height, size

                    # width
                    width = interested_stat[2]
                    if width < min_width:
                        min_width = width

                    if max_width < width:
                        max_width = width

                    # height
                    height = interested_stat[3]
                    if height < min_height:
                        min_height = height

                    if max_height < height:
                        max_height = height

                column_width_ratio = min_width / max_width
                column_height_ratio = min_height / max_height
                if column_width_ratio >= 0.4 and column_height_ratio >= 0.4:
                    boundary = self._get_x_boundary(
                        connected_components,
                        left_stat=tuple(sorted_interested_stats[0]),
                        right_stat=tuple(sorted_interested_stats[1]),
                    )
                    external_char_rect.set_attribute('x_boundary_in_rect', boundary)
                    horizontal_candidates.append(external_char_rect)

        return horizontal_candidates

    @staticmethod
    def _create_text_domain(text_score: np.ndarray, char_rect: Rect) -> np.ndarray:
        cropped_text_score = text_score[char_rect.top:char_rect.bottom, char_rect.left:char_rect.right]
        number_of_labels, markers, stats, centroids = cv2.connectedComponentsWithStats(
            cropped_text_score.astype(np.uint8), connectivity=4
        )

        text_domain = np.zeros(cropped_text_score.shape[0:3])
        for nl_idx in range(1, number_of_labels):
            text_domain[markers == nl_idx] = 255

        # Add margin
        text_domain = cv2.copyMakeBorder(text_domain, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        return text_domain

    @staticmethod
    def get_vertical_histogram_projection(
        image: np.ndarray,
        re_parse: bool = False,
        subtract_base_projection: bool = False,
    ) -> np.ndarray:
        vertical_projection = np.sum(image, 0)

        vp_max = np.max(vertical_projection)
        h, w = image.shape
        result = np.zeros((h, w), np.uint8)

        for c_idx in range(w):
            projection_height = 0
            if vp_max:
                projection_height = int(vertical_projection[c_idx] * h / vp_max)

            cv2.line(result, (c_idx, h), (c_idx, h - projection_height), (255, 255, 255), 1)

        if re_parse:
            min_col_h = h
            for c_idx in range(2, w - 2):
                with contextlib.suppress(IndexError):
                    col = result[:, c_idx]
                    col_max = col.max().astype('int64')
                    if col_max == 255:
                        col_h = h - np.argmax(col)
                        if col_h < min_col_h:
                            min_col_h = col_h

            if min_col_h == h:
                min_col_h = 0
        else:
            min_col_h = 0

        result = result[:h - min_col_h, :]

        if subtract_base_projection:
            base_projection_row_index = -1
            for r_idx, image_row in enumerate(result):
                white_indices = np.where(image_row == 255)[0]
                if len(white_indices) / w > 0.8:
                    base_projection_row_index = r_idx
                    break

            if base_projection_row_index > h / 2:
                result = result[:base_projection_row_index]

        return result

    @staticmethod
    def _get_target_component(
        connected_components_image: np.ndarray,
        markers: np.ndarray,
        nl_idx: int,
    ) -> np.ndarray:
        target_component = connected_components_image.copy()
        target_component[markers == nl_idx] = 255
        return target_component

    @staticmethod
    def _get_x_boundary(
        connected_components: np.ndarray,
        left_stat: Tuple[int, int, int, int, int],
        right_stat: Tuple[int, int, int, int, int],
    ) -> int:
        # stat: x, y, width, height, size
        left_stat_right = left_stat[0] + left_stat[2]
        right_stat_left = right_stat[0]

        min_white_pixel_count = 999
        x_at_min_white_pixel = -1
        for x in range(left_stat_right, right_stat_left + 1):
            vertical_needle = connected_components[:, x]
            white_pixel_count = cv2.countNonZero(vertical_needle)
            if x_at_min_white_pixel < min_white_pixel_count:
                min_white_pixel_count = white_pixel_count
                x_at_min_white_pixel = x

        if x_at_min_white_pixel > 0:
            return x_at_min_white_pixel
        else:
            return int((left_stat_right + right_stat_left) / 2)
