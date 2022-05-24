# Video motion detection

## Build instructions

    git clone https://github.com/reuseman/video-motion-detection
    cd video-motion-detection
    mkdir build && cd build
    cmake ..
    make

## Run

### 

    ./motion-detection [options]

    Count the number of frames with motion w.r.t the first frame of the video.

    Optional arguments:
    -h --help               shows help message and exits [default: false]
    -v --version            prints version information and exits [default: false]
    -i --input              path of the input video [required]
    -t --threshold          sets the threshold for motion detection [default: 0.6]
    -w --workers            number of workers (0 for sequential) [default: 0]
    -o --opencv-greyscale   use opencv greyscale [default: false]
    -a --blur-algorithm     blur algorithms (H1, H2, H3, H4, BOX_BLUR, BOX_BLUR_MOVING_WINDOW, OPEN_CV) [default: "BOX_BLUR_MOVING_WINDOW"]
    -p --player             shows the video player (ESC to exit) [default: false]
    -b --benchmark          benchmark mode is enabled and appends the results with the specified name in results.csv
    -i --iterations         benchmark mode is executed with the specified number of iterations [default: 1]
    --verbose               verbose mode [default: false]


### Example
    
Run in sequential mode, with a threshold of 60% of difference, using the `BOX_BLUR_MOVING_WINDOW` algorithm.

    ./motion-detection -i video.mp4 -t 0.6 -w 0 -a BOX_BLUR_MOVING_WINDOW

Run in parallel mode with 16 workers, with a threshold of 60% of difference, using the `BOX_BLUR` algorithm.

    ./motion-detection -i video.mp4 -t 0.6 -w 16 -a BOX_BLUR


// ./MotionDetection -i input.mp4 -t 0.6 -w 0 -o -a BOX_BLUR_MOVING_WINDOW -p --verbose
// ./MotionDetection -i input.mp4 -t 0.6 -a BOX_BLUR_MOVING_WINDOW -b test -i 16