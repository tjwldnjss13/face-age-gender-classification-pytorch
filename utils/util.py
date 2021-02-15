import numpy as np


def crop(in_, out_size):
    in_size = (in_.shape[1], in_.shape[2])
    drop_size = (in_size[0] - out_size[0], in_size[1] - out_size[1])
    drop_size = (drop_size[0] // 2, drop_size[1] // 2)
    out_ = in_[:, drop_size[0]:drop_size[0] + out_size[0], drop_size[1]:drop_size[1] + out_size[1]]

    return out_


def calculate_ious(box1, box2):
    ious = np.zeros((box1.shape[0], box2.shape[0]), dtype=np.float32)

    for i, b1 in enumerate(box1):
        b1 = box1[i]
        for j, b2 in enumerate(box2):
            b2 = box2[j]
            ious[i, j] = calculate_iou(b1, b2)

    return ious


def calculate_iou(box1, box2):
    # Inputs:
    #    box: [y1, x1, y2, x2]
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    y1 = np.maximum(box1[0], box2[0])
    x1 = np.maximum(box1[1], box2[1])
    y2 = np.minimum(box1[2], box2[2])
    x2 = np.minimum(box1[3], box2[3])

    iou = 0
    if y1 < y2 and x1 < x2:
        inter = (y2 - y1) * (x2 - x1)
        union = area1 + area2 - inter
        iou = inter / union

    return iou


def mean_iou_segmentation(output, predict):
    a, b = (output[:, 1, :, :] > 0), (predict > 0)

    a_area = len(a.nonzero())
    b_area = len(b.nonzero())
    union = a_area + b_area
    inter = len((a & b).nonzero())
    iou = inter / (union - inter)

    return iou


def nms(boxes, probs, threshold):
    # Inputs:
    #    boxes: [[y1, x1, y2, x2]]

    nms_list = list()

    box_prob_tuple = list(zip(boxes, probs))

    def prob(t):
        return t[1]

    box_prob_tuple.sort(key=prob, reverse=True)
    base = box_prob_tuple.pop(0)
    box_base, prob_base = base[0], base[1]

    nms_list.append(base)

    for i, t in enumerate(box_prob_tuple):
        box, prob = t[0], t[1]
        iou = calculate_iou(box_base, box)

        if iou < threshold:
            nms_list.append(t)

    return nms_list



def nms_ground_truth(anchor_boxes, ground_truth, score, iou_threshold):
    n_gt = ground_truth.shape[0] if len(ground_truth.shape) > 1 else 1
    anchor_boxes_nms = []

    for i in range(n_gt):
        anchor_boxes_cat = anchor_boxes[i]
        ious_boxes_gts = calculate_ious(anchor_boxes_cat, np.array([ground_truth[i]]))
        argmax_iou_boxes_gt = np.argmax(ious_boxes_gts, axis=0)
        max_iou_box = anchor_boxes_cat[argmax_iou_boxes_gt][0]
        anchor_boxes_nms.append(max_iou_box)
        for j in range(anchor_boxes_cat.shape[0]):
            if j == argmax_iou_boxes_gt:
                continue
            iou_temp = calculate_iou(max_iou_box, anchor_boxes_cat[j])
            if iou_temp >= iou_threshold:
                anchor_boxes_nms.append(anchor_boxes_cat[j])

    return np.array(anchor_boxes_nms)


def time_calculator(sec):
    if sec < 60:
        return 0, 0, sec
    if sec < 3600:
        M = sec // 60
        S = sec % (M * 60)
        return 0, int(M), S
    H = sec // 3600
    sec = sec % 3600
    M = sec // 60
    S = sec % 60
    return int(H), int(M), S

