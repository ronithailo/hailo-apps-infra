#!/bin/bash

# Set the resource directory
RESOURCE_DIR="$1"
echo $RESOURCE_DIR
mkdir -p "$RESOURCE_DIR"

download_images() {
  wget -nc "$1" -P $RESOURCE_DIR/faces/
}

# Define download function with file existence check and retries
download() {
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
  "https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/configs/face_recognition_local_gallery_rgba.json"
)

# Download additional videos
for url in "${CONFIGS[@]}"; do
  download "$url" &
done

for url in "${IMAGES[@]}" ; do
  download_images "$url"
done


# Wait for all background downloads to complete
wait

echo "All downloads completed successfully!"
