ALSA - Automatic Fracture Trace Extraction
==========================================

Usage
-----

Python
~~~~~~

All code is within the ``alsa`` package. Main entrypoints are
``alsa.crack_train`` for training a model and ``alsa.crack_main`` for
predicting with a trained model.

Example training workflow:

.. code:: python

   from alsa import crack_train, cli
   from pathlib import Path

   # Choose working directory
   # E.g. to choose directory on the C drive on windows:
   # Note that forward slash can be used when the path is given
   # to a Path object constructor, which handles all cross-platform differences.
   work_dir = Path("C:/alsa-working-directory")

   # Setup training and validation directories in the working directory
   crack_train.train_directory_setup(work_dir=work_dir)

   # At this point you either need to set up the training and validation
   # data in the created directories, if they do not already exist.

   # The training and validation image-trace-area combinations
   # can be checked with report_target_lists
   # It will print them to stdout without doing anything else.
   cli.report_target_lists(work_dir=work_dir)

   # Choose input parameters
   epochs = 10
   validation_steps = 10
   steps_per_epoch = 10
   
   # Choose trace width in coordinate system units (probably meters)
   trace_width = 0.01

   # Size of sub-image in training
   cell_size = 256

   # Batch size inputted to trainGenerator
   batch_size = 64

   # Start training!
   crack_train.train_main(
      work_dir=work_dir,
      epochs=epochs,
      validation_steps=validation_steps,
      steps_per_epoch=steps_per_epoch,
      trace_width=trace_width,
      cell_size=cell_size,
      batch_size=batch_size,
      # Weights, training plot and csv are outputted into the work_dir
      # unless specified here.
      old_weight_path=None,
      new_weight_path=None,
      training_plot_output=None,
      history_csv_path=None,
   )

   # See inputted working directory for outputs including the model weights


Example prediction workflow:

.. code:: python

   from alsa import crack_main
   from pathlib import Path

   # Choose working directory
   work_dir = Path("C:/alsa-working-directory")

   # Path to png-image to predict on
   img_path = Path("C:/alsa-working-directory/image.png")

   # Path to file with bounding area within the image
   area_file_path = Path("C:/alsa-working-directory/bounding_area.shp")

   # Path to file with trained weights
   unet_weights_path = Path("C:/alsa-working-directory/unet_weights.hdf5")

   # Path to predicted traces output
   unet_weights_path = Path("C:/alsa-working-directory/predicted_traces.shp")

   # Run prediction
   crack_main.crack_main(
       work_dir=work_dir,
       img_path=img_path,
       area_file_path=area_file_path,
       unet_weights_path=unet_weights_path,
       predicted_output_path=predicted_output_path,
       width=256,
       height=256,
       override_ridge_configs=dict(),
   )

   # Predicted traces are found at predicted_output_path
   # but other outputs are scattered in the working directory.


Command-line
~~~~~~~~~~~~

The package is callable from the command-line. However, it is not installable
meaning that to use the command-line interface you must be in the same
directory as the ``alsa`` code directory (that contains e.g.
``crack_train.py``).

To access the interface and get short help on its usage:

.. code:: bash

   python -m alsa --help

Currently three sub-interfaces are implemented, one for training, one for prediction
and one for checking training inputs (training and validation data).

.. code:: bash

   # Training interface
   python -m alsa train --help

   # Prediction interface
   python -m alsa predict --help

   # Check interface
   python -m alsa check --help


If training and validation data setup in ``C:/alsa-working-directory``
you can invoke the training from the command-line as follows:

.. code:: bash

   # Choose parameters as wanted
   # Note that paths must use the correct slash depending on OS
   # (backward slash on Windows)
   python -m alsa train C:\alsa-working-directory \
       --epochs 10 \
       --validation-steps 5 \
       --steps-per-epoch 5 \
       --trace-width 0.015 \
       --batch-size 32

If you wish to before training check that the training and validation
data are correctly recognized you can use the ``check`` subsommand:

.. code:: bash

   # Note that paths must use the correct slash depending on OS
   # (backward slash on Windows)
   python -m alsa check C:\alsa-working-directory

   # You can also use the same command to create the training
   # and validation directory structure by passing a flag:
   python -m alsa check C:\alsa-working-directory --setup-dirs

After training, you can predict traces. If the image you wish to predict traces
is at ``C:\alsa-working-directory\image.png``, the area bound file for that
image is at ``C:\alsa-working-directory\bounds.shp``, trained weights are at
``C:\alsa-working-directory\unet_weights.hdf5`` and you wish output traces to
go to ``C:\alsa-working-directory\predicted_traces.shp``:

.. code:: bash

   python -m alsa predict C:\alsa-working-directory \
           --img-path C:\alsa-working-directory\image.png \
           --area-file-path  C:\alsa-working-directory\bounds.shp \
           --unet-weights-path C:\alsa-working-directory\unet_weights.hdf5 \
           --predicted-output-path C:\alsa-working-directory\predicted_traces.shp

Usage (old & partly deprecated)
-------------------------------

For both CrackTrain and CrackMain:
-	Extract a .png image of the area to be analyzed
	-	This image should have black background
	- 	If this image is used for training, the quality of the image should be same across the images
	-	If this image is used for prediction, the quality of the image should be around the same as used for the training
	-	This image needs to be the smallest rectangle that covers the area
	-	THE NAME OF THIS .PNG IMAGE MUST BE A SUBSTRING OF THE SHAPEFILES
		-	If the name of the .png image is 'ABC123.png', the shapefiles must have 'ABC123' in their filenames somewhere.
		-	For this reason, if you have shapefiles named 'abc_1.shp' and 'abc_2.shp', don't name the .png image as 'abc.png' as it can confuse the 2 shapefiles.
-	Install the packages described in the requirements.txt


For prediction:

-	The program first asks for the .png image's relative or full path (including the .png at the end). Type it in.
-	The program then asks for the path to the .shp-file containing the polygon of the area to be analyzed.
-	The program then asks for the path to the .hdf5-file containing the weights of the CNN-model. By default, this is named 'unet-weights.hdf5'. If not found, try to train model first.
-	Finally the program asks for the name of the .shp-file to be produced.

For training:

-	The CrackTrain looks for Training folder and contents within it. If this is missing, run the module once and it creates them.
-	Navigate to Training\Shapefiles
	-	\Areas should contain the .shp files containing the polygon of the area to be analyzed.
	-	\Labels should contain the .shp files containing the lines you wish the program detects.
-	Navigate to Training\Images\Originals
	-	Place the .png images you wish to train for in here.
-	THE FOLDER Training\Images\Generated IS CLEARED AT THE START OF THE PROGRAM! DO NOT STORE ANYTHING HERE!
-	Running the CrackTrain module will create/overwrite a file named 'unet_weights.hdf5'. This is the file that's to be used when predicting.

Changes by BC
-------------

General changes
in the relevant files I changed 
“from keras._____ import ____” to “from tensorflow.keras.___ import ____”
example 
from tensorflow.keras.models import *
#instead of: from keras.models import *
This may not be needed, just depends on how keras is installed. 

Specific Changes

CrackTrain_BC.py
L89 – L145: Reading validation images, etc.
L165-L167: Validation data generator
L170: monitoring validation loss instead of training loss
L173: changed model.fit_generator to model.fit 
L180 – L208: Saving training history file, and basic plot of the loss and accuracy scores
L225 – L233: Added paths for validation data

CrackMain_BC.py
L93 – L96: added paths to files, instead of prompts; hence code at L98-L141 is deactivated.

Model_BC.py
No major changes, have been trying different parameters for model.compile() using different accuracy metrics, etc.

Other than these, I am in the process of adding class weights/sample weights to the training step. That's still WIP.

Proposed improvements by Jonne
------------------------------

-   Create a parametrization for the connecting line which is solely
    used to compare and decide which connector should
    be in the CrackNetWork.connect
-   Create a method for eliminating the case where a line segment
    crosses another one more than once.
-   Specify in CrackNetWork.connect when to use exact angle
    difference calculations
-   Parameter optimization
-   Improve parametrization functions to better emphasize on finding
    the correct angle and less on the distance
