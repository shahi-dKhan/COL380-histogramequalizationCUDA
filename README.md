# COL730-histogramequalizationCUDA

This is an assignment for the course COL730 - Parallel Programming. The task is to implement histogram equalization for images using CUDA. Not really images, it is a graph structured data, but the idea is the same. The code is implemented in C++, and uses CUDA for parallel processing on the GPU. 

Basically, we will be improving the contrast of a 3D color data, So, we will be given a 3D point cloud data, which consists of N points per se and each point has 3 spatial coordinates (x, y, z) and an intensity value $I \belonging {0, 1 .... 255}$. 

Then we have to perform histogram equalization using the k nearest nrighbours.


