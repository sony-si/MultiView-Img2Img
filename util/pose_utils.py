from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

# coco
# 0: "nose",
# 1: "left_eye",
# 2: "right_eye",
# 3: "left_ear",
# 4: "right_ear",
# 5: "left_shoulder",
# 6: "right_shoulder",
# 7: "left_elbow",
# 8: "right_elbow",
# 9: "left_wrist",
# 10: "right_wrist",
# 11: "left_hip",
# 12: "right_hip",
# 13: "left_knee",
# 14: "right_knee",
# 15: "left_ankle",
# 16: "right_ankle"
COCO_JOINT_PAIRS = [
    [15, 13], [13, 11], [11, 5],
    [12, 14], [14, 16], [12, 6],
    [3, 1], [1, 2], [1, 0], [0, 2], [2, 4],
    [9, 7], [7, 5], [5, 6],
    [6, 8], [8, 10]
]

#  MPII
# kps_names = ["r_ankle", "r_knee", "r_hip",
#              "l_hip", "l_knee", "l_ankle",
#              "pelvis", "throax",
#              "upper_neck", "head_top",
#              "r_wrist", "r_elbow", "r_shoulder",
#              "l_shoulder", "l_elbow", "l_wrist"]
MPII_JOINT_PAIRS = [(0, 1), (1, 2), (2, 6), (7, 12),
                    (12, 11), (11, 10), (5, 4), (4, 3),
                    (3, 6), (7, 13), (13, 14),
                    (14, 15), (6, 7), (7, 8), (8, 9)]
# MPII_JOINT_PAIRS = [
#     (2, 6), (7, 12), (12, 11), (11, 10), (3, 6), (7, 13), (13, 14),
#     (14, 15), (6, 7), (7, 8), (8, 9)]



class SkeletonFormat(Enum):
    MPII = 0
    COCO = 1

class PoseVisualizer:
    JOINT_COLORS = np.array([[180, 180, 255],
                             [0, 255, 0],
                             [0, 0, 255],
                             [255, 255, 0],
                             [0, 255, 255],
                             [255, 0, 255],
                             [150, 150, 0],
                             [0, 150, 150],
                             [150, 0, 150],
                             [128, 50, 128],
                             [100, 0, 0],
                             [0, 100, 0],
                             [0, 0, 100],
                             [100, 100, 0],
                             [0, 100, 100],
                             [100, 0, 100],
                             [200, 0, 0],
                             [0, 200, 0],
                             [0, 0, 200],
                             [0, 200, 200],
                             [200, 200, 0],
                             [200, 0, 200],
                             [200, 100, 0],
                             ])

    @staticmethod
    def calc_3d_pose_img(person_3d_pose, validity, skeleton_format, text=""):
        fig = plt.figure()
        fig.suptitle(text, fontsize=14)

        ax = plt.axes(projection='3d')
        ax.set_xlim((-140, 30))
        ax.set_ylim((-140, 30))
        ax.set_zlim((-140, 30))
        ax.view_init(30 ,200)

        if skeleton_format == SkeletonFormat.COCO:
            joint_pairs = COCO_JOINT_PAIRS
        else:
            joint_pairs = MPII_JOINT_PAIRS

        for i, joint_pair in enumerate(joint_pairs):
            k1 = joint_pair[0]
            k2 = joint_pair[1]
            if validity[k1] == 1 and validity[k2 ] ==1 :
                xline = person_3d_pose[[k1, k2], 0]
                yline = person_3d_pose[[k1, k2], 1]
                zline = person_3d_pose[[k1, k2], 2]
                ax.plot3D(xline, zline, yline, color = PoseVisualizer.JOINT_COLORS[i] / 255.0)

        for i in range(0, 17):
            ax.scatter3D(person_3d_pose[i, 0], person_3d_pose[i, 2], person_3d_pose[i, 1],
                         color=[PoseVisualizer.JOINT_COLORS[i] / 255.0])

        ax.invert_zaxis()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        fig.canvas.draw()
        # save it to a numpy array.
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return data

    @staticmethod
    def draw_2d_skeleton(img, joints, joints_vis, skeleton_format=SkeletonFormat.COCO):
        # print("draw 2d skeleton: start")
        img = img.astype(np.uint8)

        joint_index = 0
        for joint, joint_vis in zip(joints, joints_vis):
            x = int(joint[0])
            y = int(joint[1])

            to_draw = (joint_vis > 0.0 and (y > 0 and y < img.shape[0]) and (x> 0 and x< img.shape[1]))
            if to_draw:
                color = [int(PoseVisualizer.JOINT_COLORS[joint_index][0]),
                         int(PoseVisualizer.JOINT_COLORS[joint_index][1]),
                         int(PoseVisualizer.JOINT_COLORS[joint_index][2])]
                cv2.circle(img, (int(joint[0]), int(joint[1])), 2, color, 2)
            joint_index += 1

        if skeleton_format == SkeletonFormat.COCO:
            joint_pairs = COCO_JOINT_PAIRS
        else:
            joint_pairs = MPII_JOINT_PAIRS

        for i, joint_pair in enumerate(joint_pairs):
            joint1 = joint_pair[0]
            joint2 = joint_pair[1]

            if joints_vis[joint1] > 0.0 and joints_vis[joint2] > 0.0:
                joint1_x = int(joints[joint1][0])
                joint1_y = int(joints[joint1][1])

                joint2_x = int(joints[joint2][0])
                joint2_y = int(joints[joint2][1])

                color = [int(PoseVisualizer.JOINT_COLORS[i][0]),
                         int(PoseVisualizer.JOINT_COLORS[i][1]),
                         int(PoseVisualizer.JOINT_COLORS[i][2])]
                cv2.line(img, (joint1_x, joint1_y), (joint2_x, joint2_y), color, 1)


        return img