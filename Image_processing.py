#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np

"""
Used for image processing
This class allows to create square on image depending on AI results and publish the processed image
"""

class ImageProcessor:
    def __init__(self):
        self.bridge = CvBridge()
        rospy.init_node('image_processor', anonymous=True)

        
        self.image_sub = rospy.Subscriber('/camera/camera/rgb/image_raw', Image, self.image_callback)
        self.prediction_sub = rospy.Subscriber('/prediction_result', String, self.prediction_callback)
        self.processed_image_publisher = rospy.Publisher("ai/processed_image", Image, queue_size=10)
        self.red_zone_center_publisher = rospy.Publisher("ai/red_zone_center", Point, queue_size=10)
        
        self.prediction = ""
        self.processed_image = None  # Image traitée pour affichage
        self.red_zone_center = Point()
        
        #cv2.namedWindow("Processed Image", cv2.WINDOW_AUTOSIZE)

        rospy.loginfo("Image Processor is ready.")

    def prediction_callback(self, msg):
        
        self.prediction = msg.data
        #rospy.loginfo(f"Prediction received: {self.prediction}")

    def image_callback(self, msg):
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"Error converting image: {e}")
            return

        
        self.processed_image, self.red_zone_center = self.detect_red_zones(cv_image)
        #rospy.loginfo("image processed")
        self.processed_image_publisher.publish(self.bridge.cv2_to_imgmsg(self.processed_image, encoding="bgr8"))
        self.red_zone_center_publisher.publish(self.red_zone_center)

    def detect_red_zones(self, image):
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        
        color_ranges = {
            "red": [(np.array([0, 50, 50]), np.array([10, 255, 255])),
                    (np.array([170, 50, 50]), np.array([180, 255, 255]))], 
            "yellow": [(np.array([20, 100, 100]), np.array([30, 255, 255]))],  
            "green": [(np.array([40, 50, 50]), np.array([90, 255, 255]))]  
        }

        
        color_bgr = {"red": (0, 255, 0), "yellow": (0, 0, 255), "green": (0, 0, 255)}
        point = Point()
        
        for color, bounds in color_ranges.items():
            mask = np.zeros_like(hsv[:, :, 0])
            for lower, upper in bounds:
                mask |= cv2.inRange(hsv, lower, upper)

            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if cv2.contourArea(contour) > 400:
                    x, y, w, h = cv2.boundingRect(contour)
                    point.x = x+w/2
                    point.y = y+h/2
                    point.z = 0
                    
                    if color == "red":
                        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        if self.prediction:
                            cv2.putText(image, self.prediction, (x, y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    else:
                        
                        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        cv2.putText(image, "defaut", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                    
                    if color == "red":
                        roi = image[y:y + h, x:x + w]
                        dark_mask = self.detect_dark_zones(roi)

                        dark_contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for dark_contour in dark_contours:
                            if cv2.contourArea(dark_contour) > 50:
                                dark_x, dark_y, dark_w, dark_h = cv2.boundingRect(dark_contour)
                                orig_x = x + dark_x
                                orig_y = y + dark_y
                                cv2.rectangle(image, (orig_x, orig_y), (orig_x + dark_w, orig_y + dark_h), (0, 0, 255), 2)
                                cv2.putText(image, "defaut", (orig_x, orig_y - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return image, point

    def detect_dark_zones(self, roi):
        
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        
        lower_dark = np.array([0, 150, 0])  
        upper_dark = np.array([180, 255, 50])  
        
        dark_mask = cv2.inRange(hsv_roi, lower_dark, upper_dark)

        
        mean_color = cv2.mean(roi)[:3]  
        lower_color = np.array([max(0, mean_color[0] - 20), max(0, mean_color[1] - 20), max(0, mean_color[2] - 20)])
        upper_color = np.array([min(255, mean_color[0] + 20), min(255, mean_color[1] + 20), min(255, mean_color[2] + 20)])
        color_mask = cv2.inRange(roi, lower_color, upper_color)
        color_mask = cv2.bitwise_not(color_mask)  

        
        combined_mask = cv2.bitwise_and(dark_mask, color_mask)

        # Filtrage des bruits
        kernel = np.ones((3, 3), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

        return combined_mask

if __name__ == '__main__':
    try:
        image_processor = ImageProcessor()

        # Boucle principale pour maintenir l'affichage OpenCV
        while not rospy.is_shutdown():
            rospy.spin()
            """if image_processor.processed_image is not None:
                cv2.imshow("Processed Image", image_processor.processed_image)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # Échap pour quitter proprement
                rospy.signal_shutdown("Exit requested")
                break"""

    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
