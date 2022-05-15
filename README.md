

# Implementation of the UNET segmentation software in PyTorch.

Implements UNET faithfully to original paper with copy and convolve approach. This differs from many UNETS on github that choose to pad the upsampling array.

Also included is a warp function that locally distorts the image and mask. This generates data that looks like authentic biological imagery.

Results (partway through training). Due to the loss of surround in the UNET output the mask and predections are slighly zoomed. This can be fixed in final predictions by using a tiled approach with mirrored padding.

![image](https://user-images.githubusercontent.com/45679976/167451000-a6984649-5dff-43c7-a0cd-25e49c6454c7.png)

![image](https://user-images.githubusercontent.com/45679976/167451091-8d168968-d8cb-4f5c-a000-e39b9b1f79d4.png)

