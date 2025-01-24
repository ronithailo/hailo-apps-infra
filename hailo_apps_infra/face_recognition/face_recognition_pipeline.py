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
import json
import subprocess
import sys
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
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
        parser.add_argument(
            "--training-mode",
            default=False,
            action="store_true",
            help="For saving faces add this argument",
        )
        parser.add_argument(
            "--faces-path",
            help="Path to faces directory",
        )
        parser.add_argument(
            "--embeddings-path",
            help="Path to gallery embeddings path",
        )
        args = parser.parse_args()
        self.args = args
        # Call the parent class constructor
        super().__init__(args, user_data)
        # Additional initialization code can be added here
        # Set Hailo parameters these parameters should be set based on the model used
        self.batch_size = 1


        # Determine the architecture if not specified
        if args.arch is None:
            detected_arch = detect_hailo_arch()
            if detected_arch is None:
                raise ValueError("Could not auto-detect Hailo architecture. Please specify --arch manually.")
            self.arch = detected_arch
            print(f"Auto-detected Hailo architecture: {self.arch}")
        else:
            self.arch = args.arch

        self.face_recognition_path_org = os.path.dirname(os.path.abspath(__file__))
        self.face_recognition_path = os.path.dirname(sys.argv[0])
        self.face_recognition_resources_path = os.path.join(self.face_recognition_path, 'resources')
        self.check_resources_dir()
        
        
        # Set the HEF file path based on the arch
        if self.arch == "hailo8":
            self.hef_path_detection = os.path.join(self.resources_path, 'hefs/hailo8/scrfd_10g.hef')
            self.hef_path_recognition = os.path.join(self.resources_path, 'hefs/hailo8/arcface_mobilefacenet.hef')
            
        else:  # hailo8l
            self.hef_path_detection = os.path.join(self.resources_path, 'hefs/hailo8l/scrfd_2.5g.hef')
            self.hef_path_recognition = os.path.join(self.resources_path, 'hefs/hailo8l/arcface_mobilefacenet_h8l.hef')

        if "scrfd_10g" in self.hef_path_detection:
            self.detection_func = "scrfd_10g"
        elif "scrfd_2.5g" in self.hef_path_detection:
            self.detection_func = "scrfd_2_5g"

        # Set the post-processing shared object file
        self.post_process_so_scrfd = os.path.join(self.resources_path, 'libscrfd.so')
        self.post_process_so_face_recognition = os.path.join(self.resources_path, 'libface_recognition_post.so')
        self.post_process_so_face_align = os.path.join(self.resources_path, 'libvms_face_align.so ')
        self.post_process_so_cropper = os.path.join(self.postprocess_dir,'cropping_algorithms/libvms_croppers.so')
        if args.embeddings_path:
            self.gallery_path = args.embeddings_path
        else:
            self.gallery_path = os.path.join(self.face_recognition_resources_path,'face_recognition_local_gallery_rgba.json')
        if args.faces_path:
            self.faces_dir = args.faces_path
        else:
            self.faces_dir = os.path.join(self.face_recognition_resources_path,'faces/')
            
        

        self.app_callback = app_callback

        # Set the process title
        setproctitle.setproctitle("Hailo Face Recognition App")

        if not(args.training_mode):
            self.create_pipeline()

    def get_pipeline_string(self):
        if not self.video_source:
            self.video_source = os.path.join(self.resources_path, 'face_recognition.mp4')
        source_pipeline = SOURCE_PIPELINE(self.video_source, self.video_width, self.video_height)
        
        detection_pipeline = INFERENCE_PIPELINE(
            hef_path=self.hef_path_detection,
            post_process_so=self.post_process_so_scrfd,
            post_function_name=f"{self.detection_func}_letterbox",
            batch_size=self.batch_size,
            config_json=os.path.join(self.resources_path, "scrfd.json"))
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
                            f'hailogallery gallery-file-path={self.gallery_path} '
                            f'load-local-gallery=true similarity-thr=.4 gallery-queue-size=20 class-id=-1 ! '
                            f'queue name=hailo_pre_draw2 leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0')
        
        user_callback_pipeline = USER_CALLBACK_PIPELINE()
        
        display_pipeline = (f'hailooverlay name=hailo_overlay qos=false show-confidence=false local-gallery=true line-thickness=5 font-thickness=2 landmark-point-radius=8 ! '
                            f'queue name=hailo_post_draw leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! '
                            f'videoconvert n-threads=4 qos=false name=display_videoconvert qos=false ! '
                            f'queue name=hailo_display_q_0 leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! '
                            f'fpsdisplaysink video-sink=xvimagesink name=hailo_display sync={self.sync} text-overlay={self.show_fps}')
        
        if self.args.training_mode:
            source_pipeline = (f"multifilesrc location={self.faces_dir}{self.current_file} loop=true num-buffers=30 ! "
                               f"decodebin ! videoconvert n-threads=4 qos=false ! video/x-raw, format=RGB, pixel-aspect-ratio=1/1 ")
            
            detection_pipeline_wrapper = detection_pipeline
            
            gallery_pipeline =(f"queue name=hailo_pre_gallery_q leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! "
                           f"hailogallery gallery-file-path={self.gallery_path} save-local-gallery=true similarity-thr=.4 gallery-queue-size=20 class-id=-1 ! "
                           f"queue name=pre_scale_q2 leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ")
            
            display_pipeline = DISPLAY_PIPELINE(video_sink=self.video_sink, sync=self.sync, show_fps=self.show_fps)
            
        pipeline_string = (
            f'{source_pipeline} ! '
            f'{detection_pipeline_wrapper} ! '
            f'{tracker_pipeline} ! '
            f'{cropper_pipeline} ! '
            f'{gallery_pipeline} ! '
            f'{user_callback_pipeline} ! '
            f'{display_pipeline}'
        )
        if not self.args.training_mode:
            print(pipeline_string)
        return pipeline_string


    def run_training(self):
        files = self.get_files_in_directory()
        Gst.init(None)
        self.delete_file_if_exists()
        if files:
            print(f"Found {len(files)} files:")
            for i, file in enumerate(files):
                self.current_file = file
                try:
                    # Generate the pipeline string
                    pipeline_string = self.get_pipeline_string()
                    
                    # Create the GStreamer pipeline
                    pipeline = Gst.parse_launch(pipeline_string)
                    
                    # Set the pipeline to PLAYING
                    print(f"Saving embeddings for: {file}")
                    pipeline.set_state(Gst.State.PLAYING)
                    
                    # Wait for processing to complete
                    time.sleep(1)
                    
                except Exception as e:
                    print(f"Error running pipeline for {file}: {e}")
                finally:
                    # Stop and clean up the pipeline
                    if pipeline:
                        pipeline.set_state(Gst.State.NULL)
                        
                with open(self.gallery_path, 'r') as file_json:
                    data = json.load(file_json)
                    data[i]['FaceRecognition']['Name'] = self.remove_extension(file)
                with open(self.gallery_path, 'w') as file:
                    json.dump(data, file, indent=4)   
    
    def get_files_in_directory(self):
        """
        Get all files from the specified directory (non-recursive).

        :param directory: Path to the directory to search.
        :return: List of file paths.
        """
        try:
            # List all files in the directory
            files = [file for file in os.listdir(self.faces_dir) if os.path.isfile(os.path.join(self.faces_dir, file))]
            return files
        except FileNotFoundError:
            print(f"Error: Directory '{self.faces_dir}' not found.")
            return []
        except PermissionError:
            print(f"Error: Permission denied for directory '{self.faces_dir}'.")
            return []


    def remove_extension(self, filename):
        """
        Remove the extension from a file name.

        :param filename: The file name with extension.
        :return: The file name without the extension.
        """
        return os.path.splitext(filename)[0]


    def delete_file_if_exists(self):
        """
        Delete a file if it exists.

        :param file_path: Path to the file to delete.
        """
        try:
            if os.path.exists(self.gallery_path):
                os.remove(self.gallery_path)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
    
    def check_resources_dir(self):
        dir = self.face_recognition_resources_path
        if not os.path.exists(dir):
            print(f"Resources directory not found: {dir}")
            print("Running ./download_resources...")
            result = subprocess.run([f"{self.face_recognition_path_org}/download_resources.sh {dir}"], shell=True)
            if result.returncode != 0:
                print("Error: Failed to run ./download_resources", file=sys.stderr)
                sys.exit(1)
        else:
            print(f"Resources directory exists: {dir}")


    def run(self):
        if not self.args.training_mode:
            super().run()
        else:
            self.run_training()

                
if __name__ == "__main__":
    # Create an instance of the user app callback class
    user_data = app_callback_class()
    app_callback = dummy_callback
    app = GStreamerFaceRecognitionApp(app_callback, user_data)
    app.run()
    
            