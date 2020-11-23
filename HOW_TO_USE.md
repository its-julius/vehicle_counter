## How to set imaginary line:
1. Run 'python3 point_capture.py --source /video/path.ext' (--rescale if needed)
2. Hit SPACEBAR to pause or resume the video
3. Left click when the video pauses to mark the reference point (save / write down the reference point)
4. Right click to remove the reference point
5. Press ESC to exit the program
6. Open detect4.py, edit and insert imaginary line reference point at: 
` map2d.setLine((x1, y1), (x2, y2)) `


## How to use crossing counter
1. Run 'python3 detect4.py --source #stream link or video path# --view-img
