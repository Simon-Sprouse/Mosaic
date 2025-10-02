# Mosaic

This project is now hosted on a free website: https://simon-sprouse.github.io/Mosaic/

The algorithm works as follows: 

Process Image with resize, grayscale, blur, canny filter, contour detection, and custom segmentation to obtain major "pen-strokes".
Take outlines of strokes, and dfs tile-march from a random starting point using pca of underlying region for orientation.
Flood fill outward from tiles placed over strokes by using bfs style search, orientation obtained from distance map.
Gap fill any points that might have been missed -- very similar to Appollonian Gasket.
Sample RGB values under tiles. 

And like magic... here is an exmaple output:

![Example Gif](https://github.com/Simon-Sprouse/Mosaic-Assets/blob/f73133f2ac8d8066f1d1538676e2141f1836e607/mosaic.gif)

