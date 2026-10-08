"""Loading screen tips, all from the UCS668 syllabus."""

import random

TIPS = [
    # task identification (module 1)
    ("Choosing the task", "Ask what the answer looks like. One label for the image: classification. Labels plus boxes: detection. A label for every pixel: segmentation."),
    ("Choosing the task", "Counting people in a queue needs detection. Measuring how much of a leaf is diseased needs segmentation. Telling ripe from unripe fruit on a belt can be classification."),
    ("Choosing the task", "The cheapest task that answers your question is usually the right one. Segmentation is not 'better' than classification, it answers a different question."),
    # classification / transfer learning (module 2)
    ("Classification", "A classifier always picks from its own label list. Show an ImageNet model something it has never seen and it will still give you one of its 1000 classes."),
    ("Transfer learning", "Instead of training from scratch, keep a pretrained backbone and retrain only the last layer on your classes. It works with a few hundred images instead of a million."),
    ("Transfer learning", "Early layers of a CNN learn edges and textures that are useful for almost any image. That's why transfer learning works across very different datasets."),
    ("Softmax", "Softmax turns raw scores (logits) into probabilities that add up to 1. A 90% softmax score is the model's confidence, not a guarantee that it is right."),
    # network design (module 3)
    ("Conv2D", "A 3×3 convolution with 64 input and 128 output channels has 3·3·64·128 + 128 = 73,856 parameters, no matter how big the image is."),
    ("Pooling", "Max pooling halves the width and height, so the next layer sees a bigger part of the image for the same cost."),
    ("Tensor shapes", "Images go into PyTorch as (batch, channels, height, width). Most shape errors in a CNN are a mix-up between those four numbers."),
    ("Depthwise convolution", "MobileNet-style depthwise separable convolutions split one big convolution into two cheap ones and use roughly 8–9× fewer operations."),
    ("Skip connections", "ResNet adds a layer's input back to its output. Gradients get a shortcut, so very deep networks still train."),
    ("U-Net", "A U-Net shrinks the image to learn what is in it, then grows it back to full size. Skip connections carry the fine detail of where things are across the U."),
    # augmentation (module 4)
    ("Augmentation", "Augmentation is applied during training only. At test time the model sees the image as it is."),
    ("Augmentation", "Only augment in ways that can really happen. Flipping a cat is fine. Flipping a road sign with text, or a digit like 6, can change its meaning."),
    ("Augmentation", "If your camera will work at night, darken some of the training images too. Then the model doesn't depend on someone enhancing every frame."),
    ("Mixup and CutMix", "Mixup blends two images and their labels. CutMix pastes a patch of one image into another. Both stop a model from becoming overconfident."),
    # optimization, loss, metrics, activations (module 5)
    ("Optimizers", "SGD takes a step down the gradient. Momentum remembers the last few steps. Adam also scales each weight's step by how noisy its gradient has been."),
    ("Learning rate", "Too high and the loss jumps around or explodes. Too low and training takes forever. It's usually the first hyperparameter to tune."),
    ("Loss functions", "Classification usually uses cross-entropy. Detection adds a box regression loss on top. Segmentation often uses per-pixel cross-entropy or Dice loss."),
    ("ReLU", "ReLU is max(0, x). It's cheap, and it doesn't saturate for positive inputs the way sigmoid does, so gradients survive deep networks."),
    ("Precision and recall", "Precision: of what you flagged, how much was real? Recall: of what was real, how much did you flag? You usually trade one for the other."),
    ("Accuracy can lie", "If 99% of frames have no intruder, a model that always says 'no intruder' is 99% accurate and completely useless."),
    # detection and segmentation (module 6)
    ("IoU", "Intersection over Union: overlap area divided by combined area. A predicted box usually counts as correct at IoU ≥ 0.5."),
    ("mAP", "mAP averages precision over all recall levels and all classes. COCO's version also averages over IoU thresholds from 0.5 to 0.95."),
    ("NMS", "Detectors often predict several boxes for one object. Non-maximum suppression keeps the most confident one and drops the overlapping rest. YOLO26 is trained to not need it."),
    ("Detection labels", "Detection datasets need a box and a class for every object in every image. That's far more labelling work than classification folders."),
    ("COCO", "COCO has 80 object classes: people, vehicles, animals, furniture, kitchen items. No padlocks, no stains, no ghosts."),
    ("Segmentation", "Semantic segmentation labels every pixel with a class. Instance segmentation also separates one chair from the other."),
    # edge AI (module 7)
    ("Edge AI", "Running the model on the device means no network round-trip, it works offline, and the images never leave the camera."),
    ("Model size", "Every parameter is stored as a number. 10 million parameters at 32 bits each is 40 MB, at 16 bits 20 MB, at 8 bits 10 MB."),
    ("Quantization", "INT8 quantization stores weights as 8-bit integers instead of 32-bit floats. The model gets about 4× smaller and often faster, usually for a small accuracy drop."),
    ("Pruning", "Pruning removes weights or whole channels that contribute little. Removing whole channels is what actually makes inference faster on normal hardware."),
    ("Latency budget", "A 30 fps camera gives you 33 ms per frame. Preprocessing, the model and drawing the result all have to fit inside that."),
    ("Input resolution", "Halving the input width and height cuts a CNN's compute by about 4×. Small objects are the first thing you lose."),
    ("FLOPs vs speed", "Fewer FLOPs doesn't always mean faster. Memory access, the operations used, and what the chip accelerates all matter."),
    ("Accelerators", "Phones and boards like the Jetson have GPUs or NPUs for neural networks. The same model can be 10× faster there than on the CPU."),
    ("Export formats", "Training happens in PyTorch, but devices usually run an exported model: ONNX, TensorRT, Core ML or TFLite."),
    ("Power", "On a battery device, every millisecond of compute costs energy. A model that's twice as fast can mean twice the battery life."),
    ("The edge trade-off", "The most accurate model isn't automatically the best one to ship. It has to fit the memory, meet the frame rate, and still be accurate enough."),
    # module 8
    ("Normalization", "Models expect inputs scaled the same way as during training, e.g. ImageNet mean and std. Skip that step and accuracy quietly falls apart."),
    ("Batch norm", "BatchNorm keeps each layer's inputs at a stable scale during training, which lets you use higher learning rates."),
    ("Overfitting", "Training accuracy keeps going up, validation accuracy stops. Augmentation, dropout, weight decay and more data all fight overfitting."),
    ("Train, validate, test", "Tune on the validation set, report on the test set, and never let the test images influence a decision."),
    # preprocessing
    ("Gamma", "Gamma correction is a lookup table of 256 values. It's one of the cheapest ways to lift detail out of dark images."),
    ("Histograms", "A histogram piled against the left edge means a dark image. Piled against both edges means clipped pixels, and that detail is gone."),
    ("CLAHE", "CLAHE equalizes contrast in small tiles with a limit, so it brings out local detail without blowing up noise in flat areas."),
]


def random_tip(exclude: set[int] | None = None) -> tuple[int, str, str]:
    choices = [i for i in range(len(TIPS)) if not exclude or i not in exclude] or list(range(len(TIPS)))
    i = random.choice(choices)
    return i, *TIPS[i]
