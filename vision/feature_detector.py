from collections import defaultdict
import cv2
from cv_bridge import CvBridge
import numpy as np
import quaternion
import rospy
import sensor_msgs.msg
import tf
from tf.transformations import quaternion_matrix, translation_matrix
from camera import CameraStreamer
from camera import CameraInfo
from utils.quaternions import quaternion_convert, quaternion_average, pose

# in C++ we could have leveraged the packages
#  - http://wiki.ros.org/image_proc?distro=noetic
#  - http://wiki.ros.org/depth_image_proc
# which are very efficient due to their implementation via nodelets (http://wiki.ros.org/nodelet)

# TODO remove or FIXME i'm inconsistent
class PointCloudSceneBuilder(object):

    def __init__(self, camera_streamer, base_frame, ee_frame, color_optical_frame, depth_optical_frame):
        super(PointCloudSceneBuilder, self).__init__()
        
        self.cs = camera_streamer
        self.model = np.empty(shape=[0, 6])
        self.base_frame = base_frame
        self.ee_frame = ee_frame
        
        self.listener = tf.TransformListener()
        try:
            # wait for the node to initialize https://answers.ros.org/question/203274/#post-id-327336
            self.listener.waitForTransform(ee_frame, base_frame, rospy.Time(0), rospy.Duration(3.0))
            # get useful transforms
            self.T_ee_colorOpt = self._get_transform_matrix(ee_frame, color_optical_frame)
            self.T_ee_depthOpt = self._get_transform_matrix(ee_frame, depth_optical_frame)

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            raise RuntimeError("Could not get transforms")
        
        self.recorded_aruco_poses = []

    def _get_transform_matrix(self, dst, src):
        trans, rot = self.listener.lookupTransform(dst, src, rospy.Time(0))
        return self.listener.fromTranslationRotation(trans, rot)

    def _frame_to_pose(self, T):
        t, r = T[:3, 3], quaternion.from_rotation_matrix(T[:3, :3])
        return np.array([t, r])

    def _pose_to_frame(self, pose):
        t, r = pose[0], quaternion_convert(pose[1])
        return self.listener.fromTranslationRotation(t, r)

    def _T_base_ee(self):
        return self._get_transform_matrix(self.base_frame, self.ee_frame)

    def _T_ee_base(self):
        return self._get_transform_matrix(self.ee_frame, self.base_frame)

    # http://wiki.ros.org/tf2_sensor_msgs
    def record_point_cloud(self, filename):
        points, colors = self.cs.get_points_colored(max_depth=500, filter_nan=True)

        # mm -> m and normalize
        points = np.concatenate([points / 1000, np.ones(shape=[points.shape[0], 1])], axis=1)
        # transform to global frame
        T_base_colorOpt = np.dot(self._T_base_ee(), self.T_ee_colorOpt)
        points = np.dot(T_base_colorOpt, points.T).T[:, :3]
        # m -> mm
        points_colors = np.concatenate([points * 1000, colors], axis=1)
        self.model = np.concatenate([self.model, points_colors], axis=0)

    def _debug_save(self):
        # save as text
        np.savetxt("%s.txt" % (filename), self.model)
        # save 3D model
        points = self.model[:, :3]
        colors = self.model[:, 3:]
        with open("%s.ply" % (filename), 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write("element vertex %s\n" % (points.shape[0]))
            f.write("property double x\n")
            f.write("property double y\n")
            f.write("property double z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            for i in range(points.shape[0]):
                f.write("%f %f %f %d %d %d\n" % (points[i, 0], points[i, 1], points[i, 2], colors[i, 0], colors[i, 1], colors[i, 2]))


class ArucoSceneBuilder(object):

    def __init__(self, camera_topic, ref_frame, camera_frame, debug_mode=False):
        super(ArucoSceneBuilder, self).__init__()
        msg = rospy.wait_for_message("%s/camera_info" % (camera_topic), sensor_msgs.msg.CameraInfo, timeout=5.0)
        self.camera_info = CameraInfo(msg)
        self.bridge = CvBridge()
        self.image = None
        self.image_sub = rospy.Subscriber('%s/image_raw' % (camera_topic), sensor_msgs.msg.Image, callback=self._cb)

        # listen change in pose
        self.listener = tf.TransformListener()
        try:
            # wait for the node to initialize https://answers.ros.org/question/203274/#post-id-327336
            self.listener.waitForTransform(camera_frame, ref_frame, rospy.Time(0), rospy.Duration(3.0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            raise RuntimeError("Could not find the requested transform chain")
        
        # storage
        self._ref_frame = ref_frame
        self._camera_frame = camera_frame
        self._recorded_aruco_poses = defaultdict(list)
        self._debug = debug_mode
        

    def _cb(self, image): 
        self.image = self.bridge.imgmsg_to_cv2(image)

    def _frame_to_pose(self, T):
        t, r = T[:3, 3], quaternion.from_rotation_matrix(T[:3, :3])
        return pose(t, r)

    def _pose_to_frame(self, pose):
        t, r = pose['p'], quaternion_convert(pose['o'][()])
        return self.listener.fromTranslationRotation(t, r)

    def _get_transform_matrix(self, dst, src):
        trans, rot = self.listener.lookupTransform(dst, src, rospy.Time(0))
        return self.listener.fromTranslationRotation(trans, rot)

    def _T_ref_camera(self):
        return self._get_transform_matrix(self._ref_frame, self._camera_frame)

    def get_aruco_poses(self):
        image = self.image
        cam_info = self.camera_info

        # detect aruco markers
        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_1000)
        aruco_params = cv2.aruco.DetectorParameters_create()
        corners, ids, rejected = cv2.aruco.detectMarkers(image, aruco_dict, parameters=aruco_params, cameraMatrix=cam_info.K, distCoeff=cam_info.D)
        if len(corners) == 0:
            return (np.array([]), np.array([]), []), image
        # show markers
        if self._debug:
            image = cv2.aruco.drawDetectedMarkers(image, corners, ids)
        # estimate poses w.r.t. camera frame
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.030, cameraMatrix=cam_info.K, distCoeffs=cam_info.D)
        rvec = quaternion.from_rotation_vector(np.squeeze(rvec, axis=1))
        tvec = np.squeeze(tvec, axis=1)
        # show poses
        if self._debug:
            for r, t in zip(rvec, tvec):
                r = quaternion.as_rotation_vector(r)
                cv2.aruco.drawAxis(image, cam_info.K, cam_info.D, r, t, 0.01)
        return (rvec, tvec, ids.T.tolist()[0]), image
    
    def record_aruco_poses(self):
        (rvec, tvec, ids), image = self.get_aruco_poses()
        T_ref_camera = self._T_ref_camera()
        for r, t, aruco_id in zip(rvec, tvec, ids):
            # transform to ee pose and then to global frame
            T_camera_aruco = self.listener.fromTranslationRotation(t, quaternion_convert(r))
            T_ref_aruco = np.dot(T_ref_camera, T_camera_aruco)
            pose_ref_aruco = self._frame_to_pose(T_ref_aruco)
            # store the pose, grouping by ArUco id
            self._recorded_aruco_poses[aruco_id].append(pose_ref_aruco)
        return (rvec, tvec, ids), image
    
    def get_recorded_aruco_poses(self, wrt_frame=None):
        clusters = defaultdict(list)
        # for each id...
        for aruco_id, poses in self._recorded_aruco_poses.items():
            idxs = set(range(len(poses)))
            # group duplicates
            while len(idxs) > 0:
                i = idxs.pop()
                idxs_near_i = set()
                for j in idxs:
                    d = np.linalg.norm(poses[i]['p'] - poses[j]['p'])
                    if d <= 0.025:
                        idxs_near_i.add(j)
                # remove each j-th pose near the i-th pose
                idxs -= idxs_near_i
                clusters[aruco_id].append(list(idxs_near_i) + [i])
            # average duplicates
            for i in range(len(clusters[aruco_id])):
                cluster = clusters[aruco_id][i]
                if len(cluster) == 1:
                    # no average required
                    clusters[aruco_id][i] = poses[cluster[0]]
                else:
                    # average required
                    cluster_poses = np.array([poses[j] for j in cluster])
                    p = np.mean(cluster_poses['p'], axis=0)
                    o = quaternion_average(cluster_poses['o'])
                    clusters[aruco_id][i] = pose(p, o)
        
        # return the poses
        if wrt_frame != self._ref_frame:
            # we must change reference frame
            T_frame_ref = self._get_transform_matrix(wrt_frame, self._ref_frame)
            for aruco_id, _pose in clusters.items():
                T_ref_aruco = self._pose_to_frame(_pose)
                T_frame_aruco = np.dot(T_frame_ref, T_ref_aruco)
                pose_frame_aruco = self._frame_to_pose(T_frame_aruco)
                clusters[aruco_id] = pose_frame_aruco
        return clusters

    def quit(self):
        self.image_sub.unregister()


if __name__ == '__main__':
    
    rospy.init_node('feature_detector', anonymous=False)

    asb = ArucoSceneBuilder("/wrist_rgbd/color", '/robot_arm_base', '/robot_wrist_rgbd_color_optical_frame')
    asb.record_aruco_poses()
    poses = asb.get_recorded_aruco_poses(wrt_frame='/robot_arm_tool0')

    rvec, tvec, image = asb.get_aruco_poses()
    import matplotlib.pyplot as plt
    plt.imshow(image)
    plt.show()

    asb.quit()
