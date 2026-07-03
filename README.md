# Upper Body Segmentation


This tool enables users to define custom zones on the face of a subject in a video and track these zones as the subject moves, in the context of a seated interview. It also tracks the position of the upper body, as well as the whole head.


![](examples/il.png)

##### Features:     

- Zone Definition: Users can draw zones on the face (e.g., eyes, mouth) using an interface.
- Real-time Tracking: This tool employs computer vision algorithms to track the defined zones across consecutive frames of a video.

##### Limitations

- Only videos with one person can be handled properly.
- The pipeline only tracks two user-defined zones (eye zone and nose-mouth zone), as well as the whole face, the head and the upper body. 
- The foreground detection model sometimes fails to produce a coherent output.

## Getting started

### Installing dependencies


```
virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Defining the zones


Zones are stored as a list of indexes corresponding to a polygon of facial landmarks (see https://storage.googleapis.com/mediapipe-assets/documentation/mediapipe_face_landmark_fullsize.png)

NOTE: The zone definition only matters at the very end of the pipeline: you can apply the pipeline to process your video(s) and define the zones later.
NOTE: To allow for non-convex zones, the order of landmark selection matters.

Baseline zones are already defined in the default configuration file `configurations/config.txt`.

##### If you wish to define custom zones:  

Run `zone_selection_app.py`
We will use a representation of the face of Isaac Newton as a basis for zone definition. You can change this with the argument --image 

```
python zone_selection_app.py --image examples/isaac.jpeg
```

> You can now select zones by clicking on landmarks. 
Use {n} to confirm the current selected zone and select a next zone.
Use {c} to cancel the current point.
Use {q} to confirm the selected zones and quit.
>

Once you press q and exit, a new configuration file is created in `configurations/`
Rename it to `config.txt` and you are good to go!

### Running the pipeline 

The pipeline consists of 6 stages:
1. Frame by frame face and landmarks detection
2. Omniscient landmark denoising (using the fact that landmarks in succeeding video frames should be close)
3. Landmarks previsualization
4. Frame by frame upper body detection 
5. Omniscient upper body detection (using the fact that the upper body should not move too much in succeeding video frames. This is really important as the RMBG model used at stage 4 will randomly pick up background objects)
6. Zone tracking 

You need to run `app.py` and specify the video file and the folder you want the results to be stored in.
To run the whole pipeline:
```
python app.py video.mp4 video_data/
```
To run the pipeline from stage i to stage j (assuming the pipeline has already been run from steps 1 to i and you didn't change the directory argument):
```
python app.py video.mp4 --start-stage i --end-stage j
```
