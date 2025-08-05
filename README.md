# Mosaic

This Repo is undergoing a move from C++ to WebAsm so it can be run in a small React App -- Stay tuned! View the first release for a working, downloadable verison that can be run from the cli. 

The algorithm works as follows: 

Process Image with resize, grayscale, blur, canny filter, contour detection, and custom segmentation to obtain major "pen-strokes".
Take outlines of strokes, and dfs tile-march from a random starting point using pca of underlying region for orientation.
Flood fill outward from tiles placed over strokes by using bfs style search, orientation obtained from distance map.
Gap fill any points that might have been missed -- very similar to Appollonian Gasket.
Sample RGB values under tiles. 

And like magic... here is an exmaple output:

![Example Gif](example.gif)
