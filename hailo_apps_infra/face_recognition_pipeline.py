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

CURR = ""

def get_files_in_directory(directory):
    """
    Get all files from the specified directory (non-recursive).

    :param directory: Path to the directory to search.
    :return: List of file paths.
    """
    try:
        # List all files in the directory
        files = [file for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]
        return files
    except FileNotFoundError:
        print(f"Error: Directory '{directory}' not found.")
        return []
    except PermissionError:
        print(f"Error: Permission denied for directory '{directory}'.")
        return []


def remove_extension(filename):
    """
    Remove the extension from a file name.

    :param filename: The file name with extension.
    :return: The file name without the extension.
    """
    return os.path.splitext(filename)[0]
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

def delete_file_if_exists(file_path):
    """
    Delete a file if it exists.

    :param file_path: Path to the file to delete.
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")



# -----------------------------------------------------------------------------------------------
# User Gstreamer Application
# -----------------------------------------------------------------------------------------------

# This class inherits from the hailo_rpi_common.GStreamerApp class
class GStreamerFaceRecognitionApp(GStreamerApp):
    def __init__(self, app_callback, user_data):
        parser = get_default_parser()
        parser.add_argument(
            "--save-faces",
            default=False,
            action="store_true",
            help="For saving faces add this argument",
        )
        args = parser.parse_args()
        self.args = args
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
            self.hef_path_detection = os.path.join(self.resources_path, 'scrfd_10g.hef')
            self.hef_path_recognition = os.path.join(self.resources_path, 'arcface_mobilefacenet.hef')
            
        else:  # hailo8l
            self.hef_path_detection = os.path.join(self.resources_path, 'scrfd_2.5g.hef')
            self.hef_path_recognition = os.path.join(self.resources_path, 'arcface_mobilefacenet_h8l.hef')

        if "scrfd_10g" in self.hef_path_detection:
            self.detection_func = "scrfd_10g"
        elif "scrfd_2.5g" in self.hef_path_detection:
            self.detection_func = "scrfd_2_5g"

        # Set the post-processing shared object file
        self.post_process_so_scrfd = os.path.join(self.resources_path, 'libscrfd.so')
        self.post_process_so_face_recognition = os.path.join(self.resources_path, 'libface_recognition_post.so')
        self.post_process_so_face_align = os.path.join(self.resources_path, 'libvms_face_align.so ')
        self.post_process_so_cropper = '/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/cropping_algorithms/libvms_croppers.so'
        self.gallery_path = os.path.join(self.resources_path,'face_recognition_local_gallery_rgba.json')
        self.faces_dir = os.path.join(self.resources_path,'faces/')
        

        self.app_callback = app_callback

        # Set the process title
        setproctitle.setproctitle("Hailo Face Recognition App")

        if not(args.save_faces):
            self.create_pipeline()

    def get_pipeline_string(self):
        if(self.video_source.endswith('example.mp4')):
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
                            f'fpsdisplaysink video-sink=xvimagesink name=hailo_display sync=false text-overlay=false')
        
        if self.args.save_faces:
            print(source_pipeline)
            source_pipeline = (f"multifilesrc location={self.faces_dir}{CURR} loop=true num-buffers=30 ! "
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
        if not self.args.save_faces:
            print(pipeline_string)
        return pipeline_string

if __name__ == "__main__":
    # Create an instance of the user app callback class
    user_data = app_callback_class()
    app_callback = dummy_callback
    app = GStreamerFaceRecognitionApp(app_callback, user_data)
    print(app.args)
    if not app.args.save_faces:
        app.run()
    else:
        files = get_files_in_directory(os.path.join(app.resources_path, 'faces/'))
        face_embeddings_file_path = os.path.join(app.resources_path, 'face_recognition_local_gallery_rgba.json')
        Gst.init(None)
        delete_file_if_exists(face_embeddings_file_path)
        if files:
            print(f"Found {len(files)} files:")
            for file in files:
                CURR = file
                try:
                    # Generate the pipeline string
                    pipeline_string = app.get_pipeline_string()
                    # Create the GStreamer pipeline
                    pipeline = Gst.parse_launch(pipeline_string)
                    
                    # Set the pipeline to PLAYING
                    print(f"Saving embeddings for: {file}")
                    pipeline.set_state(Gst.State.PLAYING)
                    
                    # Wait for processing to complete (adjust as needed)
                    time.sleep(1)
                    
                except Exception as e:
                    print(f"Error running pipeline for {file}: {e}")
                finally:
                    # Stop and clean up the pipeline
                    if pipeline:
                        pipeline.set_state(Gst.State.NULL)
                        
            with open(face_embeddings_file_path, 'r') as file_json:
                data = json.load(file_json)
                for i, file in enumerate(files):
                    data[i]['FaceRecognition']['Name'] = remove_extension(file)
            with open(face_embeddings_file_path, 'w') as file:
                json.dump(data, file, indent=4)   
            