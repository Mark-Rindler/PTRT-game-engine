All of the renders in the following folders are in the renderers balanced mode. This mode balances performance constraints with quality wants. Render
mode comp demonstrates all rendering modes on the showcase scene. 

NOTE: Due to size constraints, I have no included all of the videos of the engine but rather just pictures.

The rendering modes are:
fast
performance
balanced
quality
ultra

sorted by computational difficulty/output image quality.

Here are the FPS stats for each mode.
** These were not achieved via a preprogrammed set of frames to be rendered, rather I manually walked through scenes. 
** This might cause very slightly misrepresented numbers. Furthermore, there is slight confounding because I am recording
** As I am testing the FPS. This test occured on a scene with about 1 million triangles, and 8 separate models. 
** Ultra and ultra ultra are not meant for real time, rather an evalutation of the engine
fast:           
FPS Summary
Frames: 10038
Run time (s): 56.117
Average FPS: 178.876
Lowest FPS: 8

performance:    
FPS Summary
Frames: 5189
Run time (s): 64.5651
Average FPS: 80.3685
Lowest FPS: 6

balanced:       


quality:        
FPS Summary
Frames: 3129
Run time (s): 62.3698
Average FPS: 50.1685
Lowest FPS: 8

ultra:          
Frames: 1492
Run time (s): 75.0379
Average FPS: 19.8833
Lowest FPS: 7

ultra ultra (not really in the program but the settings are here):
            perfSettings.enableDenoiser = false;
            perfSettings.enableBloom = false;
            perfSettings.enableMotionVectors = false;
            perfSettings.samplesPerPixel = 256;
            perfSettings.maxBounceDepth = 32;
            perfSettings.resolutionScale = 1.0f;
            perfSettings.russianRouletteStartBounce = 16;

FPS Summary
Frames: 406
Run time (s): 22.7586
Average FPS: 17.8394
Lowest FPS: 0
