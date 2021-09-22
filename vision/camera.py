import numpy as np
import rospy
import sensor_msgs.msg
from cv_bridge import CvBridge

#
# the images travel using `image_transport` http://wiki.ros.org/image_transport?distro=kinetic
# (also `compressed_image_transport` and `theora`, but we are not interested in those)
#
# since there are no Python communication APIs, we must go barebones and use a Subscriber on the topics:
#  - /wrist_rgbd/color/image_raw
#  - /wrist_rgbd/depth/image_raw
#  - /wrist_rgbd/infra1/image_raw
#  - /wrist_rgbd/infra2/image_raw
#
# also, we get the camera info via over the topic /wrist_rgbd/*/camera_info
#
# to retrieve the images, we use `cv_bridge` http://wiki.ros.org/cv_bridge
#

class CameraInfo():

    def __init__(self, sensor_msgs_camera_info):
        # The image dimensions with which the camera was calibrated. Normally
        # this will be the full camera resolution in pixels.
        self.h = sensor_msgs_camera_info.height
        self.w = sensor_msgs_camera_info.width

        # Intrinsic camera matrix for the raw (distorted) images.
        #     [fx  0 cx]
        # K = [ 0 fy cy]
        #     [ 0  0  1]
        self.K = np.array(sensor_msgs_camera_info.K).reshape([3, 3])
        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        self.cx = self.K[0, 2]
        self.cy = self.K[1, 2]
        
        # The distortion parameters
        self.distorsion_model = sensor_msgs_camera_info.distortion_model
        self.D = sensor_msgs_camera_info.D

        # Projection/camera matrix
        #     [fx'  0  cx' Tx]
        # P = [ 0  fy' cy' Ty]
        #     [ 0   0   1   0]
        # careful! (Tx, Ty, 0) is optical center displacement of the second camera w.r.t. the first one,
        #          assuming same height z (hence the 0). Tx and Ty are != 0 only when we have a stereo 
        #          pair setup (of course)
        self.P = np.array(sensor_msgs_camera_info.P).reshape([3, 4])

        self.rect_R = np.array(sensor_msgs_camera_info.R).reshape([3, 3])


# TODO remove or FIXME i'm inconsistent
class CameraStreamer(object):

    def __init__(self, *cameras):
        self.camera_info = {}
        self.image = {}
        self.sub = {}
        self.bridge = CvBridge()
        for cam in cameras:
            cam_name = cam.rsplit('/', 1)[1]
            # get camera info
            msg = rospy.wait_for_message("%s/camera_info" % (cam), sensor_msgs.msg.CameraInfo, timeout=5.0)
            self.camera_info[cam_name] = CameraInfo(msg)
            # init image
            cam_info = self.camera_info[cam_name]
            self.image[cam_name] = np.empty(shape=[cam_info.h, cam_info.w, 0])
            # make subscriber
            self.sub[cam_name] = rospy.Subscriber('%s/image_raw' % (cam), sensor_msgs.msg.Image, callback=self._cb, callback_args=[cam_name])

    def _cb(self, image, args): 
        self.image[args[0]] = self.bridge.imgmsg_to_cv2(image)

    def _project(self, M, camera_name='color'):
        m = np.dot(self.camera_info[camera_name].P, M)
        return m / m[2, :]

    # WARNING: this is in local (camera) reference frame
    def get_points(self, min_depth=0, max_depth=300, filter_nan=False, camera_name='depth'): 
        image = self.image[camera_name]
        cam_info = self.camera_info[camera_name]
        i = np.arange(image.shape[0])
        j = np.arange(image.shape[1])
        yy, xx = np.meshgrid(i, j, indexing='ij')
        Zc = np.where(np.logical_and(min_depth < image, image < max_depth), image, np.nan)
        Xc = (xx - cam_info.cx) * Zc / cam_info.fx
        Yc = (yy - cam_info.cy) * Zc / cam_info.fy
        M = np.stack([Xc, Yc, Zc], axis=2)
        if filter_nan:
            M = M.reshape(-1, 3)
            return M[~np.isnan(M).any(axis=1), :]
        return M

    def get_points_colored(self, min_depth=0, max_depth=300, filter_nan=False, camera_depth_name='depth', camera_color_name='color'):
        # get homogeneous points
        points = self.get_points(min_depth, max_depth, filter_nan, camera_depth_name).reshape(-1, 3)
        points = np.concatenate([points, np.ones(shape=[points.shape[0], 1])], axis=1)

        # project, create [(x, y, z) <-> (u, v)] and filter
        cam_color_info = self.camera_info[camera_color_name]
        pixels = self._project(points.T).T
        points_pixels = np.concatenate([points[:, 0:3], pixels[:, 0:2]], axis=1)
        points_pixels = points_pixels[np.logical_and(0 <= points_pixels[:, 3], points_pixels[:, 3] < cam_color_info.w), :]
        points_pixels = points_pixels[np.logical_and(0 <= points_pixels[:, 4], points_pixels[:, 4] < cam_color_info.h), :]
        pixels = np.round(points_pixels[:, 3:]).astype(dtype=np.int)

        # fetch color
        color = self.image[camera_color_name][pixels[:, 1], pixels[:, 0]]
        return points_pixels[:, :3], color

    def _debug_save_capture(self, filepath):
        points, colors = self.get_points_colored(max_depth=500, filter_nan=True)
        if not filepath.endswith(".ply"):
            filepath += ".ply"
        with open(filepath, 'w') as f:
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

    def _debug_show(self):
        import matplotlib.pyplot as plt
        n = len(self.image)
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / float(cols)))
        _, axes = plt.subplots(rows, cols)
        axes = axes.flatten()
        for ax in axes:
            ax.set_axis_off()
        for cam_name, ax in zip(sorted(self.image.keys()), axes):
            ax.set_axis_on()
            im = ax.imshow(self.image[cam_name])
            ax.set_title(cam_name)
            plt.colorbar(im, ax=ax)
        plt.show()

    def quit(self):
        for sub in self.sub.values():
            sub.unregister()


if __name__ == '__main__':

    rospy.init_node("camera_streamer", anonymous=False)

    cs = CameraStreamer("/wrist_rgbd/color", "/wrist_rgbd/depth", "/wrist_rgbd/infra1")
    rospy.sleep(1.0)

    # points, color = cs.get_points_colored(max_depth=500)
    
    # cs._debug_save_capture("test.ply")
    cs._debug_show()
    
    cs.quit()

    
    # # 1. register depth w.r.t. color
    # "depth_image_proc/register"
    # # 2. fusion of color over depth
    # "depth_image_proc/point_cloud_xyzrgb"