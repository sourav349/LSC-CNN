## LSC-CNN
# LSC-CNN: Long-Short Connection CNN for Non-Uniformity Correction
1. Overview
Non-Uniformity Correction (NUC) aims to eliminate fixed-pattern noise (FPN) caused by non-uniform pixel response in infrared imaging systems. Traditional methods often struggle to balance noise removal with detail preservation. To overcome this, the Long-Short Connection Convolutional Neural Network (LSC-CNN) was proposed, leveraging deep learning with architectural innovations to preserve fine image details while effectively removing FPN.

2. Motivation for LSC-CNN
Existing CNN-based NUC methods face two key limitations:

Gradient vanishing or explosion in deep networks, reducing training effectiveness.

Loss of image details, especially edges and textures, due to aggressive denoising or shallow networks.

LSC-CNN introduces long and short connections in a residual learning framework to address both challenges simultaneously.

3. Architecture Design
3.1 Input
The input is a raw infrared image affected by fixed-pattern noise.

Preprocessing steps may include grayscale normalization and resizing.

3.2 Core Components
Convolutional Layers: Extract spatial features and noise patterns.

Residual Blocks: Learn to predict the noise (FPN) rather than the clean image, improving convergence.

Short Connections: Connect shallower layers to deeper ones, helping gradients flow during backpropagation. These aid in better noise learning and training stability.

Long Connections: Connect the input directly to the output, preserving original scene details and preventing over-smoothing.

3.3 Multiply Operation (Enhancement Block)
A pixel-wise multiplication layer enhances the contrast of the corrected image by combining denoised output with weighted features.

This emphasizes edges and fine details lost in earlier layers.

3.4 Output Layer
Produces a noise estimation map, which is subtracted from the original noisy input to generate the final corrected image.

4. Training Process
4.1 Dataset
The network is trained on paired infrared images:

Input: noisy IR image with FPN.

Target: clean or synthetically denoised version (ground truth).

4.2 Loss Function
Mean Squared Error (MSE): Measures the difference between the corrected image and ground truth.

SSIM Loss (optional): Ensures structural similarity is preserved.

4.3 Optimization
Optimizer: Adam or SGD with learning rate decay.

Batch Normalization: Used after each convolution to improve convergence speed.

Epochs: Trained over several epochs with validation checks to avoid overfitting.
