import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as mpcm

from dataset import dataset_common

def gain_translate_table():
    label2name_table = {}
    for class_name, labels_pair in dataset_common.VOC_LABELS.items():
        label2name_table[labels_pair[0]] = class_name
    return label2name_table

label2name_table = gain_translate_table()

def colors_subselect(colors, num_classes=21):
    dt = len(colors) // num_classes
    sub_colors = []
    for i in range(num_classes):
        color = colors[i*dt]
        if isinstance(color[0], float):
            sub_colors.append([int(c * 255) for c in color])
        else:
            sub_colors.append([c for c in color])
    return sub_colors

colors = colors_subselect(mpcm.plasma.colors, num_classes=21)
colors_tableau = [(255, 255, 255), (31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """Draw a collection of lines on an image.
    """
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    return img

def draw_rectangle(img, p1, p2, color=[255, 0, 0], thickness=2):
    cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)

    return img

def draw_bbox(img, bbox, shape, label, color=[255, 0, 0], thickness=2):
    p1 = (int(bbox[0] * shape[0]), int(bbox[1] * shape[1]))
    p2 = (int(bbox[2] * shape[0]), int(bbox[3] * shape[1]))
    cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)
    p1 = (p1[0]+15, p1[1])
    cv2.putText(img, str(label), p1[::-1], cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)

    return img

# def bboxes_draw_on_img(img, classes, scores, bboxes, thickness=2):
#     shape = img.shape
#     for i in range(bboxes.shape[0]):
#         bbox = bboxes[i]
#         color = colors_tableau[classes[i]]
#         # Draw bounding box...
#         p1 = (int(bbox[0] * shape[0]), int(bbox[1] * shape[1]))
#         p2 = (int(bbox[2] * shape[0]), int(bbox[3] * shape[1]))
#         cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)
#         # Draw text...
#         s = '%s|%.3f' % (label2name_table[classes[i]], scores[i])
#         p1 = (p1[0]-6, p1[1])
#         cv2.putText(img, s, p1[::-1], cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)

#     return img
def bboxes_draw_on_img(img, classes, scores, bboxes, thickness=2):
    shape = img.shape
    scale = 0.4
    text_thickness = 1
    line_type = 8
    #print(bboxes.shape[0], scores.shape[0], classes.shape[0])
    for i in range(bboxes.shape[0]):
        #print(classes[i])
        if classes[i] < 1: continue
        #print(classes[i])
        bbox = bboxes[i]
        color = colors_tableau[classes[i]]
        # Draw bounding box...
        p1 = (int(bbox[0] * shape[0]), int(bbox[1] * shape[1]))
        p2 = (int(bbox[2] * shape[0]), int(bbox[3] * shape[1]))
        #print(p1,p2)
        if (p2[0] - p1[0] < 1) or (p2[1] - p1[1] < 1):
            continue


        cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)
        # Draw text...
        s = '%s/%.1f%%' % (label2name_table[classes[i]], scores[i]*100)
        # text_size is (width, height)
        text_size, baseline = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, text_thickness)
        p1 = (p1[0] - text_size[1], p1[1])

        cv2.rectangle(img, (p1[1] - thickness//2, p1[0] - thickness - baseline), (p1[1] + text_size[0], p1[0] + text_size[1]), color, -1)

        cv2.putText(img, s, (p1[1], p1[0] + baseline), cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), text_thickness, line_type)

    return img

