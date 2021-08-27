import numpy as np
import rospy
import sensor_msgs.msg


#
# the images travel using `image_transport` http://wiki.ros.org/image_transport?distro=kinetic
# (also `compressed_image_transport` and `theora`, but we are not interested in those)
#
# since there are no Python APIs, we must go barebones and use a Subscriber on the topics:
#  - /wrist_rgbd/color/image_raw
#  - /wrist_rgbd/depth/image_raw
#  - /wrist_rgbd/infra1/image_raw
#  - /wrist_rgbd/infra2/image_raw
#
# also, we get the camera info via over the topic /wrist_rgbd/*/camera_info
#


class CameraStreamer(object):

    def __init__(self):
        self.camera_info_color = rospy.wait_for_message("/wrist_rgbd/color/camera_info", sensor_msgs.msg.CameraInfo, timeout=5.0)
        self.camera_info_depth = rospy.wait_for_message("/wrist_rgbd/depth/camera_info", sensor_msgs.msg.CameraInfo, timeout=5.0)

        self.image_color = np.zeros(shape=[self.camera_info_color.height, self.camera_info_color.width])
        self.image_depth = np.zeros(shape=[self.camera_info_depth.height, self.camera_info_depth.width])

        self.sub_color = rospy.Subscriber('/wrist_rgbd/color/image_raw', sensor_msgs.msg.Image, callback=self._color_cb)
        self.sub_depth = rospy.Subscriber('/wrist_rgbd/depth/image_raw', sensor_msgs.msg.Image, callback=self._depth_cb)

    def _color_cb(self, image):
        self.image_color[:, :] = image.data
                    
    def _depth_cb(self, image):
        self.image_color[:, :] = image.data

    def quit(self):
        self.sub_color.unregister()
        self.sub_depth.unregister()


if __name__ == '__main__':

    cs = CameraStreamer()
    cs.quit()