# UClick

`UClick` is an open source software written in Python to register Control Points on images for further use in [UCalib](https://github.com/Ulises-ICM-UPC/UCalib) and [UDrone](https://github.com/Ulises-ICM-UPC/UDrone).

### Description

The program is aimed to assist the registration of pixel coordinates corresponding to Control Points for subsequent image calibration. Two types of control points are distinguished: points located on the horizon (Horizon Points, HP) and points located in known coordinates (Ground Control Points, GCP), for which a file with their coordinates will be supplied. The output files can then be used afterwards for manual calibration of images in [UCalib](https://github.com/Ulises-ICM-UPC/UCalib) and [UDrone](https://github.com/Ulises-ICM-UPC/UDrone).

### Requirements and project structure
To run the software it is necessary to have Python (3.8) and install the following dependencies:
- cv2 (4.2.0)
- numpy (1.19.5)
- matplotlib (3.1.2)

In parenthesis we indicate the version with which the software has been tested. It is possible that it works with older versions. 

The structure of the project is the following:
* `example.py`
* `example_notebook.py`
* **`uclick`**
  * `uclick.py`
  * `ulises_uclick.py`
* **`example`**
  * `GCPs.txt`
  * `GCPs.jpg`
  * **`basis`**
    * `image000001.jpg`
    * `image000001cdg.txt`
    * `image000001cdh.txt`
    * . . .
  * **`TMP`**
    * `image000001cdg_check.jpg`
    * `image000001cdh_check.jpg`
    * . . .

The local modules of `UClick` are located in the **`uclick`** folder.

To run a demo with the video in folder **`example`** and a set of frames in **`basis`** using a Jupyter Notebook we provide the file `example_notebook.ipynb`. For experienced users, the `example.py` file can be run in a terminal. `UClick` handles `PNG` and `JPEG` image formats.

## Registration of Control Points

Import modules:


```python
import os
import sys
sys.path.insert(0, 'uclick')
import uclick as uclick
import matplotlib
matplotlib.use('Qt4Agg')
```

Set the main path and the path of the folder where the images are placed:


```python
pathFolderMain = 'example'
pathFolderBasis = pathFolderMain + os.sep + 'basis'
```

### Horizon Points

Each of the images `<image>.jpg` in the folder **`basis`** will be offered to register the HPs. In the case that for an `<image>.jpg` image located in the folder **`basis`** has already been registered, set `overwrite = True` to register it again and `False` if you want to keep it. To facilitate the verification that the HPs have been correctly registered, set parameter `verbosePlot = True` to generate an image `<image>cdh_check.jpg` on a **`TMP`** folder showing the HPs, and to `False` otherwise. 



```python
overwrite = True
verbosePlot = True
```

#### Recording of pixel coordinates
The recording of a pixel in an image corresponding to a Control Point is done in two steps. Using the left mouse button, a first click zooms the image region in which the point is located and a second click records the pixel coordinates (column and row). Press the middle button or the `Return` or `Esc` keys to end point recording of an image.

Run the code to click HP:


```python
uclick.ClickHorizon(pathFolderBasis, overwrite, verbosePlot)
```

As a result, for each images a file `<image>cdh.txt` containing the pixel coordinates of the HPs is generated. The structure of this file is the following:
* `<basisFrame>cdh.txt`: For each HP one line with
>`pixel-column`, `pixel-row`

### Ground Control Points

The available GCPs are listed in the file `GCPs.txt`. The location of these points is shown in the reference image `GCPs.jpg`.The structure of this file the following:
* `GCPs.txt`: For each GCP one line with
>`code`, `x-coordinate`, `y-coordinate`, `z-coordinate`, `switch`

Quantities must be separated by at least one blank space between them and the last record should not be continued with a newline (return).

For each of the images `<image>.jpg` in the folder **`basis`**, the GCPs with `switch = on` will be offered to be registered. In the case that an `<image>.jpg` has already been registered, set `overwrite = True` to register it again and `False` if you want to keep the  points that have already been registered. To verificate the GCPs have been registered, set parameter `verbosePlot = True` to generate an image `<image>cdg_check.jpg` on a **`TMP`**, and to `False` otherwise. 



```python
overwrite = True
verbosePlot = True
```

#### Recording of pixel coordinates
The recording of a pixel in an image follows the procedure explained above. Now, the middle button or the `Return` or `Esc` keys are used to skip a point. If you do so, its pixel-coordinates will be recorded with values `-999`.

Run the code to click GCP:


```python
uclick.ClickGCPs(pathFolderMain, pathFolderBasis, overwrite, verbosePlot)
```

As a result, for each images a file `<image>cdg.txt` containing the coordinates of the GCPs is generated. The structure of this file is the following:
* `<image>cdg.txt`: For each GCP one line with
>`pixel-column`, `pixel-row`, `x-coordinate`, `y-coordinate`, `z-coordinate`, `code`

Real-world coordinates are given in the same coordinate system as the GCPs.

New GCPs can be added to `<image>cdg.txt` by changing the value of `switch = off` to `on` in the file `GCPs.txt` and then re-running the code to click GCP. In case that `switch = on` changes to `off` the corresponding GCP will be deleted in `<image>cdg.txt`.

As the points that were excluded during the registration process are saved with coordinates `-999`, if you want to register them again in mode `overwrite = False`, you will have to delete them manually from the file. 


## Contact us

Are you experiencing problems? Do you want to give us a comment? Do you need to get in touch with us? Please contact us!

To do so, we ask you to use the [Issues section](https://github.com/Ulises-ICM-UPC/UClick/issues) instead of emailing us.

## Contributions

Contributions to this project are welcome. To do a clean pull request, please follow these [guidelines](https://github.com/MarcDiethelm/contributing/blob/master/README.md)

## License

UClick is released under a [AGPL-3.0 license](https://github.com/Ulises-ICM-UPC/UClick/blob/master/LICENSE). If you use UClick in an academic work, please cite:

    @Online{ulisesclick, 
      author = {Simarro, Gonzalo and Calvete, Daniel},
      title = {UClick},
      year = 2021,
      url = {https://github.com/Ulises-ICM-UPC/UClick}
      }

