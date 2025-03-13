#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np

"""
Class for image processing.
It detects red zones, annotates the image, and publishes the processed image along with the center of the detected zone.
"""
class ImageProcessor:
    def __init__(self):
        self.bridge = CvBridge()  # Bridge between OpenCV and ROS for image conversion
        rospy.init_node('image_processor', anonymous=True)

        # Subscribing to necessary topics
        self.image_sub = rospy.Subscriber('/camera/camera/rgb/image_raw', Image, self.image_callback)
        self.prediction_sub = rospy.Subscriber('/prediction_result', String, self.prediction_callback)
        
        # Publishing processed images and the center of the detected red zone
        self.processed_image_publisher = rospy.Publisher("ai/processed_image", Image, queue_size=10)
        self.red_zone_center_publisher = rospy.Publisher("ai/red_zone_center", Point, queue_size=10)
        
        self.prediction = ""  # Variable to store the received prediction
        self.processed_image = None  # Processed image for display
        self.red_zone_center = Point()  # Center of the detected red zone
        
        rospy.loginfo("Image Processor is ready.")

    def prediction_callback(self, msg):
        """ Callback to store the received prediction """
        self.prediction = msg.data

    def image_callback(self, msg):
        """ Callback to process the received image """
        try:
            # Convert ROS message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"Error converting image: {e}")
            return

        # Detect red zones in the image
        self.processed_image, self.red_zone_center = self.detect_red_zones(cv_image)
        
        # Publish the results
        self.processed_image_publisher.publish(self.bridge.cv2_to_imgmsg(self.processed_image, encoding="bgr8"))
        self.red_zone_center_publisher.publish(self.red_zone_center)

    def detect_red_zones(self, image):
        """ Detects red zones and annotates the image """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Convert to HSV

        # Define color ranges for detection
        color_ranges = {
            "red": [(np.array([0, 50, 50]), np.array([10, 255, 255])),
                     (np.array([170, 50, 50]), np.array([180, 255, 255]))]
        }

        point = Point()
        
        for color, bounds in color_ranges.items():
            mask = np.zeros_like(hsv[:, :, 0])
            for lower, upper in bounds:
                mask |= cv2.inRange(hsv, lower, upper)  # Create a mask for the color

            # Improve the mask
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Detect contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if cv2.contourArea(contour) > 400:  # Filter out small objects
                    x, y, w, h = cv2.boundingRect(contour)
                    point.x = x + w / 2
                    point.y = y + h / 2
                    point.z = 0
                    
                    # Draw a rectangle around the detected zone
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    if self.prediction:
                        cv2.putText(image, self.prediction, (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return image, point

if __name__ == '__main__':
    try:
        image_processor = ImageProcessor()
        rospy.spin()  # Keep the ROS node running
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
