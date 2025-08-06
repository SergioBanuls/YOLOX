# Prepare datasets

If you have a dataset directory, you could use os environment variable named `YOLOX_DATADIR`. Under this directory, YOLOX will look for datasets in the structure described below, if needed.

```
$YOLOX_DATADIR/
  COCO/
```

You can set the location for builtin datasets by

```shell
export YOLOX_DATADIR=/path/to/your/datasets
```

If `YOLOX_DATADIR` is not set, the default value of dataset directory is `./datasets` relative to your current working directory.

## Expected dataset structure for [COCO detection](https://cocodataset.org/#download):

```
COCO/
  annotations/
    instances_{train,val}2017.json
  {train,val}2017/
    # image files that are mentioned in the corresponding json
```

You can use the 2014 version of the dataset as well.

## Preparing Custom VOC-style Dataset for YOLOX

To prepare your custom dataset for training with YOLOX, follow these steps:

### 1. Directory Structure

Ensure your dataset follows the VOC directory structure:

```
datasets/
  VOCdevkit/
    VOC2020/  # or your custom year
      JPEGImages/      # Place all your images here
      Annotations/     # Place all your XML annotation files here
      ImageSets/
        Main/
          train.txt    # List of training image names (without .xml extension)
          val.txt      # List of validation image names (without .xml extension)
```

### 2. Image Preparation

-   **Place all images in the `JPEGImages/` folder**
-   Images should be in JPEG format (.jpg extension)
-   **Each image must have a corresponding XML annotation file with the exact same name**
    -   Example: `image001.jpg` → `image001.xml`

### 3. XML Annotation Format

Each XML file in the `Annotations/` folder must follow the VOC format:

```xml
<annotation>
  <filename>image001.jpg</filename>
  <size>
    <width>2268</width>
    <height>4032</height>
    <depth>3</depth>
  </size>
  <object>
    <name>your_class_name</name>
    <bndbox>
      <xmin>100</xmin>
      <ymin>200</ymin>
      <xmax>300</xmax>
      <ymax>400</ymax>
    </bndbox>
  </object>
  <!-- Add more objects as needed -->
</annotation>
```

### 4. Image Size Configuration

-   **Current YOLOX input size is set to (640, 640) pixels**
-   If your training images have a different aspect ratio or size, you may need to adjust the input size in the configuration
-   Check `yolox/exp/yolox_base.py` and modify `self.input_size = (640, 640)` if needed
-   Consider the trade-off between accuracy and training speed when choosing the input size

### 5. Dataset Split Files

Generate two text files in `ImageSets/Main/`:

#### `train.txt`

Contains the list of image names (without .xml extension) for training:

```
image001
image002
image003
...
```

#### `val.txt`

Contains the list of image names (without .xml extension) for validation:

```
image004
image005
image006
...
```

**Important Notes:**

-   **Each line should contain only the filename without extension** (no .jpg or .xml)
-   Typically use an 80/20 split (80% training, 20% validation)
-   The names in these files must correspond to existing images and XML files
-   Each image listed must have both a `.jpg` file in `JPEGImages/` and a `.xml` file in `Annotations/`

### 6. Verification Checklist

Before training, ensure:

-   [ ] All images are in `JPEGImages/` folder
-   [ ] All XML annotations are in `Annotations/` folder
-   [ ] Each image has a corresponding XML file with the same name
-   [ ] `train.txt` and `val.txt` files are properly generated
-   [ ] XML files follow the correct VOC format
-   [ ] Input size configuration matches your dataset requirements
-   [ ] No missing files referenced in train.txt or val.txt
