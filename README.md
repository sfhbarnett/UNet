

# Implementation of the UNET segmentation software in PyTorch.

Implements UNET faithfully to original paper with copy and convolve approach. This differs from many UNETS on github that choose to pad the upsampling array.

Also included is a warp function that locally distorts the image and mask. This generates data that looks like authentic biological imagery.

Results (partway through training). Due to the loss of surround in the UNET output the mask and predections are slighly zoomed. This can be fixed in final predictions by using a tiled approach with mirrored padding.

<img width="1283" alt="image" src="https://user-images.githubusercontent.com/45679976/190645169-c1919b50-5568-440d-853e-9884b5dcd544.png">
