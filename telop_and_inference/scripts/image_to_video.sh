#!/bin/sh  # Ensure the script runs with sh

# Default settings
DEFAULT_RESOLUTION="680x480"
DEFAULT_FRAME_RATE="10"
DEFAULT_IMAGE_PATTERN="%06d.png"
DEFAULT_DELETE_FOLDER="y"  # Automatically delete folder after video creation

# Ask for the parent directory (task directory, e.g., "tack_plates")
echo "Enter the full path to the task directory (e.g., 'stack_plates'):"
read TASK_DIR

# Check if the task directory exists
if [ ! -d "$TASK_DIR" ]; then
    echo "Error: Directory '$TASK_DIR' does not exist!"
    exit 1
fi

# Loop through each episode directory in the task directory
for EPISODE_DIR in "$TASK_DIR"/*/; do
    # Get the episode directory name (e.g., 000001, 000002, ...)
    EPISODE_NAME=$(basename "$EPISODE_DIR")

    # Check if the episode directory exists and contains subdirectories
    if [ -d "$EPISODE_DIR" ]; then
        echo "Processing episode '$EPISODE_NAME'..."

        # Loop through each RGB subdirectory (left_rgb, right_rgb, front_rgb) in the episode
        for RGB_DIR in "$EPISODE_DIR"/*rgb/; do
            # Get the subdirectory name (left_rgb, right_rgb, front_rgb)
            RGB_NAME=$(basename "$RGB_DIR")

            echo "Processing '$RGB_NAME' folder in episode '$EPISODE_NAME'..."

            # Check if the directory contains any PNG images
            if ls "$RGB_DIR"/*.png 1> /dev/null 2>&1; then
                # Automatically set the output video name based on the folder name (e.g., left_rgb -> left.mp4)
                OUTPUT_VIDEO="$EPISODE_DIR/$RGB_NAME.mp4"

                # Use default resolution and frame rate if not provided
                RESOLUTION=${DEFAULT_RESOLUTION}
                FRAME_RATE=${DEFAULT_FRAME_RATE}
                IMAGE_PATTERN=${DEFAULT_IMAGE_PATTERN}

                # Create video from images using FFmpeg
                ffmpeg -framerate "$FRAME_RATE" -i "$RGB_DIR/$IMAGE_PATTERN" -s "$RESOLUTION" -c:v libx264 -pix_fmt yuv420p -r "$FRAME_RATE" "$OUTPUT_VIDEO"

                # Check if the video creation was successful
                if [ $? -eq 0 ]; then
                    echo "Video created successfully: $OUTPUT_VIDEO"

                    # Automatically delete the RGB folder and its contents
                    if [ "$DEFAULT_DELETE_FOLDER" = "y" ]; then
                        rm -rf "$RGB_DIR"
                        echo "'$RGB_DIR' folder and its contents deleted."
                    fi
                else
                    echo "Error creating video for '$RGB_NAME'. No folder was deleted."
                fi
            else
                echo "No PNG images found in the directory '$RGB_DIR'."
            fi
        done
    else
        echo "Skipping non-directory $EPISODE_DIR"
    fi
done
