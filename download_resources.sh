#!/bin/bash

# Set the resource directory
RESOURCE_DIR="./resources"
mkdir -p "$RESOURCE_DIR"

download_images() {
  wget -nc "$1" -P ./resources/faces/
}

# Define download function with file existence check and retries
download_model() {
  file_name=$(basename "$1")
  if [ ! -f "$RESOURCE_DIR/$file_name" ]; then
    echo "Downloading $file_name..."
    wget --tries=3 --retry-connrefused --quiet --show-progress "$1" -P "$RESOURCE_DIR" || {
      echo "Failed to download $file_name after multiple attempts."
      exit 1
    }
  else
    echo "File $file_name already exists. Skipping download."
  fi
}

# Define all URLs in arrays
H8_HEFS=(
  "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.13.0/hailo8/yolov8m_pose.hef"
  "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.13.0/hailo8/yolov5m_seg.hef"
  "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.13.0/hailo8/yolov8m.hef"
  "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.13.0/hailo8/scrfd_10g.hef"
  "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.13.0/hailo8/arcface_mobilefacenet.hef"
)

H8L_HEFS=(
  "https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/hefs/h8l_rpi/yolov8s_h8l.hef"
  "https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/hefs/h8l_rpi/yolov5n_seg_h8l_mz.hef"
  "https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/hefs/h8l_rpi/yolov8s_pose_h8l.hef"
  "https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/hefs/h8l_rpi/scrfd_2.5g.hef"
  "https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/hefs/h8l_rpi/arcface_mobilefacenet_h8l.hef"
)

VIDEOS=(
  "https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/video/example.mp4"
  "https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/video/face_recognition.mp4"
)


IMAGES=(
  "https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/images/faces/alon.png"
  "https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/images/faces/itai.png"
  "https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/images/faces/avishay.png"
  "https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/images/faces/gal.png"
  "https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/images/faces/katia.png"
  "https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/images/faces/reut.png"
  "https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/images/faces/roi.png"
  "https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/images/faces/shir.png"
  "https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/images/faces/yuval.png"
)

CONFIGS=(
  "https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/configs/scrfd.json"
  "https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/configs/face_recognition_local_gallery_rgba.json"
)
# If --all flag is provided, download everything in parallel
if [ "$1" == "--all" ]; then
  echo "Downloading all models and video resources..."
  for url in "${H8_HEFS[@]}" "${H8L_HEFS[@]}" "${VIDEOS[@]}"; do
    download_model "$url" &
  done
else
  if [ "$DEVICE_ARCHITECTURE" == "HAILO8L" ]; then
    echo "Downloading HAILO8L models..."
    for url in "${H8L_HEFS[@]}"; do
      download_model "$url" &
    done
  elif [ "$DEVICE_ARCHITECTURE" == "HAILO8" ]; then
    echo "Downloading HAILO8 models..."
    for url in "${H8_HEFS[@]}"; do
      download_model "$url" &
    done
  fi
fi

# Download additional videos
for url in "${VIDEOS[@]}" "${CONFIGS[@]}"; do
  download_model "$url" &
done

for url in "${IMAGES[@]}" ; do
  download_images "$url"
done


# Wait for all background downloads to complete
wait

echo "All downloads completed successfully!"
