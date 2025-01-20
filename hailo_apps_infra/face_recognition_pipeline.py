import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import argparse
import multiprocessing
import numpy as np
import setproctitle
import cv2
import time
import hailo
from hailo_apps_infra.hailo_rpi_common import (
    get_default_parser,
    detect_hailo_arch,
)
from hailo_apps_infra.gstreamer_helper_pipelines import(
    QUEUE,
    SOURCE_PIPELINE,
    INFERENCE_PIPELINE,
    INFERENCE_PIPELINE_WRAPPER,
    TRACKER_PIPELINE,
    USER_CALLBACK_PIPELINE,
    DISPLAY_PIPELINE,
    CROPPER_PIPELINE
)
from hailo_apps_infra.gstreamer_app import (
    GStreamerApp,
    app_callback_class,
    dummy_callback
)



# -----------------------------------------------------------------------------------------------
# User Gstreamer Application
# -----------------------------------------------------------------------------------------------

# This class inherits from the hailo_rpi_common.GStreamerApp class
class GStreamerFaceRecognitionApp(GStreamerApp):
    def __init__(self, app_callback, user_data):
        parser = get_default_parser()
        args = parser.parse_args()
        # Call the parent class constructor
        super().__init__(args, user_data)
        # Additional initialization code can be added here
        # Set Hailo parameters these parameters should be set based on the model used
        self.batch_size = 2


        # Determine the architecture if not specified
        if args.arch is None:
            detected_arch = detect_hailo_arch()
            if detected_arch is None:
                raise ValueError("Could not auto-detect Hailo architecture. Please specify --arch manually.")
            self.arch = detected_arch
            print(f"Auto-detected Hailo architecture: {self.arch}")
        else:
            self.arch = args.arch


        # Set the HEF file path based on the arch
        if self.arch == "hailo8":
            self.hef_path_detection = os.path.join(self.current_path, '../resources/scrfd_10g.hef')
            self.hef_path_recognition = os.path.join(self.current_path, '../resources/arcface_mobilefacenet.hef')
            self.detection_func = "scrfd_10g"
            
        else:  # hailo8l
            self.hef_path_detection = os.path.join(self.current_path, '../resources/scrfd_2.5g.hef')
            self.hef_path_recognition = os.path.join(self.current_path, '../resources/arcface_mobilefacenet_h8l.hef')
            self.detection_func = "scrfd_2_5g"
        if args.hef_path is not None:
            self.hef_path_recognition = args.hef_path

        # Set the post-processing shared object file
        self.post_process_so_scrfd = os.path.join(self.current_path, '../resources/libscrfd.so')
        self.post_process_so_face_recognition = os.path.join(self.current_path, '../resources/libface_recognition_post.so')
        self.post_process_so_face_align = os.path.join(self.current_path, '../resources/libvms_face_align.so ')
        self.post_process_so_cropper = os.path.join(self.current_path, '/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/cropping_algorithms/libvms_croppers.so')
        

        self.app_callback = app_callback

        # Set the process title
        setproctitle.setproctitle("Hailo Face Recognition App")

        self.create_pipeline()

    def get_pipeline_string(self):
        if(self.video_source.endswith('example.mp4')):
            self.video_source = os.path.join(self.current_path, '../resources/face_recognition.mp4')
        source_pipeline = SOURCE_PIPELINE(self.video_source, self.video_width, self.video_height)
        
        detection_pipeline = INFERENCE_PIPELINE(
            hef_path=self.hef_path_detection,
            post_process_so=self.post_process_so_scrfd,
            post_function_name=f"{self.detection_func}_letterbox",
            batch_size=self.batch_size,
            config_json="../resources/scrfd.json")
        detection_pipeline_wrapper = INFERENCE_PIPELINE_WRAPPER(detection_pipeline)
        
        tracker_pipeline = TRACKER_PIPELINE(class_id=-1,
                                            kalman_dist_thr=0.7,
                                            iou_thr=0.8,
                                            init_iou_thr=0.9,
                                            keep_new_frames=2,
                                            keep_tracked_frames=6,
                                            keep_lost_frames=8,
                                            keep_past_metadata=True,
                                            name='hailo_face_tracker')
        
        mobile_facenet_pipeline = INFERENCE_PIPELINE(
            hef_path=self.hef_path_recognition,
            post_process_so=self.post_process_so_face_recognition,
            post_function_name="filter",
            batch_size=self.batch_size,
            config_json=None,
            name='face_recognition_inference')
        cropper_pipeline = CROPPER_PIPELINE(inner_pipeline=(f'hailofilter so-path={self.post_process_so_face_align} '
                                                            f'name=face_align_hailofilter use-gst-buffer=true qos=false ! '
                                                            f'queue name=detector_pos_face_align_q leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! '
                                                            f'{mobile_facenet_pipeline}'),
                                         so_path=self.post_process_so_cropper,
                                         function_name="face_recognition",
                                         internal_offset=True)
        
        gallery_pipeline = (f'queue name=hailo_pre_gallery_q leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 !  '
                            f'hailogallery gallery-file-path=../resources/face_recognition_local_gallery_rgba.json '
                            f'load-local-gallery=true similarity-thr=.4 gallery-queue-size=20 class-id=-1 ! '
                            f'queue name=hailo_pre_draw2 leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0')
        
        user_callback_pipeline = USER_CALLBACK_PIPELINE()
        
        display_pipeline = (f'hailooverlay name=hailo_overlay qos=false show-confidence=false local-gallery=true line-thickness=5 font-thickness=2 landmark-point-radius=8 ! '
                            f'queue name=hailo_post_draw leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! '
                            f'videoconvert n-threads=4 qos=false name=display_videoconvert qos=false ! '
                            f'queue name=hailo_display_q_0 leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! '
                            f'fpsdisplaysink video-sink=xvimagesink name=hailo_display sync=false text-overlay=false')
        
        pipeline_string = (
            f'{source_pipeline} ! '
            f'{detection_pipeline_wrapper} ! '
            f'{tracker_pipeline} ! '
            f'{cropper_pipeline} ! '
            f'{gallery_pipeline} ! '
            f'{user_callback_pipeline} ! '
            f'{display_pipeline}'
        )
        print(pipeline_string)
        return pipeline_string

if __name__ == "__main__":
    # Create an instance of the user app callback class
    user_data = app_callback_class()
    app_callback = dummy_callback
    app = GStreamerFaceRecognitionApp(app_callback, user_data)
    app.run()
