from util.pose_utils import SkeletonFormat

class CMUPanoptic:

    @staticmethod
    def read_3d_pose_annotation_file(pose_annotation_file, skeleton_format=SkeletonFormat.COCO):
        import json
        import numpy as np
        try:
            with open(pose_annotation_file) as skeleton_file:
                pose_json = json.load(skeleton_file)

                for body in pose_json['bodies']:
                    subject_id = body['id']

                    # Assuming 19 3D joints, stored as an array [x1,y1,z1,c1,x2,y2,z2,c2,...],
                    # where c1 ... c19 are per-joint confidences
                    body_pose = np.array(body['joints19']).reshape((-1, 4))

                    # The 3D skeletons in the CMU-Panoptic dataset have the following keypoint order:
                    # 0: Neck
                    # 1: Nose
                    # 2: BodyCenter(center of hips)
                    # 3: lShoulder
                    # 4: lElbow
                    # 5: lWrist,
                    # 6: lHip
                    # 7: lKnee
                    # 8: lAnkle
                    # 9: rShoulder
                    # 10: rElbow
                    # 11: rWrist
                    # 12: rHip
                    # 13: rKnee
                    # 14: rAnkle
                    # 15: rEye
                    # 16: lEye
                    # 17: rEar
                    # 18: lEar
                    #
                    # cmu real order:
                    # 0: "Neck",
                    # 1: "Nose",
                    # 2: "MidHip",
                    # 3: "LShoulder",
                    # 4: "LElbow",
                    # 5: "LWrist",
                    # 6: "LHip",
                    # 7: "LKnee",
                    # 8: "LAnkle",
                    # 9: "RShoulder",
                    # 10: "RElbow",
                    # 11: "RWrist",
                    # 12: "RHip",
                    # 13: "RKnee",
                    # 14: "RAnkle",
                    # 15: "LEye",
                    # 16: "LEar",
                    # 17: "REye",
                    # 18: "REar",

                    #
                    # coco:
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
                    # valid_joints = (1, 15, 17, 16, 18, 3, 9, 4, 10, 5, 11, 6, 12, 7, 13, 8, 14)

                    # mpii:
                    # (0 - rankle, 1 - r knee, 2 - r hip, 3 - l hip, 4 - l knee, 5 - l ankle, 6 - pelvis, 7 - thorax, 8 - upper neck,
                    # 9 - head top, 10 - r wrist, 11 - r elbow, 12 - r shoulder, 13 - l shoulder, 14 - l elbow, 15 - l wrist).

                    if skeleton_format == SkeletonFormat.COCO:
                        valid_joints = (1, 15, 17, 16, 18, 3, 9, 4, 10, 5, 11, 6, 12, 7, 13, 8, 14)
                        coco_body_pose = body_pose[valid_joints,:]
                        return coco_body_pose
                    else:
                        # MPII
                        # mpii_valid_joints = (14, 13, 12, 6, 7, 8,
                        #                 2, 9 + 3 + 2, 0 + 1, 17 + 15, 11, 10, 9, 3, 4, 5, 1)

                        aux_mpii_valid_joints = (14, 13, 12, 6, 7, 8,
                                        2, 9, 0, 17, 11, 10, 9, 3, 4, 5, 1)

                        mpii_body_pose = body_pose[aux_mpii_valid_joints,:]
                        mpii_body_pose[7] = (body_pose[9]+body_pose[3]+body_pose[2])/3
                        mpii_body_pose[8] = (body_pose[0]+body_pose[1])/2
                        mpii_body_pose[9] = (body_pose[17]+body_pose[15])/2
                        return mpii_body_pose
        except IOError as e:
            print('Error reading {0}\n'.format(pose_annotation_file) + e.strerror)