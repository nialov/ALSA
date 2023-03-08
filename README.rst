ALSA - Automatic Fracture Trace Extraction
==========================================

Installation
------------

conda
~~~~~

Recommended method for ``Windows``.

.. code:: bash

   # Create environment
   conda env create -n alsa -f environment.yml

   # Activate environment
   conda activate alsa

   # Start an IDE within environment in current folder
   # E.g. Visual Studio Code
   code .

   # Or test the command-line interface
   python -m alsa --help

Python (poetry)
~~~~~~~~~~~~~~~

Poetry is a  package manager for Python. It uses the ``pyproject.toml`` and
``poetry.lock`` files to recreate a Python environment. If ``poetry.lock`` is
missing it uses the dependency specification of ``pyproject.toml`` to create
it. Otherwise it uses ``poetry.lock`` to exactly recreate the environment.

The installation with ``poetry`` likely only works on ``linux``-systems.

.. code:: bash
   
   # Need to have poetry installed on system
   # Install environment with dependencies
   poetry install

   # Run command within environment
   # E.g. to test command-line interface
   poetry run python -m alsa

   # Enter a shell with the environment
   poetry shell

Reproduction of manucript results
---------------------------------

To reproduce the data structure for the manucript, *Automated mapping of
bedrock-fracture traces from UAV-acquired images using U-NET
convolutional neural networks*, use the ``scripts/reproduce.py`` script
to download and organize trace, area, orthomosaic and model data. The
script requires a Linux environment.

To list options:

.. code:: bash

   python3 scripts/reproduce.py --help
   
To download all data to a new ``./reproduction`` directory:

.. code:: bash

   python3 scripts/reproduce.py reproduction/

After running the script, the output directory can be used as the
working directory that is passed to the ``alsa`` Python functions or
command-line entrypoint to train a new model, or use the included
already trained unet model to generate trace predictions from existing
or new image data. See below for guidance on the ``alsa`` `Python
function <#python>`__ and `command-line <#command-line>`__ interfaces.

The reproduction process of training and predicting will take a
considerable amount of time due to both intensive machine learning and
vectorization processes. For testing purposes we suggest using very
limited datasets.

Commands for reproduction from the command-line:

0. Download all data to a new ``./reproduction`` directory (if not
   already done):

.. code:: bash

   python3 scripts/reproduce.py reproduction/

1. Check that ``./reproduction``  directory contents are valid:

.. code:: bash

   python -m alsa check reproduction/

2. Train model

.. code:: bash

   python -m alsa train reproduction/ \
       --epochs 100 \
       --validation-steps 100 \
       --steps-per-epoch 300

3. Generate predicted traces using trained model

.. code:: bash

   python -m alsa predict reproduction/ \
           --img-path reproduction/prediction/Images/og1.png \
           --area-file-path reproduction/prediction/Shapefiles/Areas/og1_area.shp \
           --unet-weights-path reproduction/unet_weights.hdf5 \
           --predicted-output-path reproduction/og1_predicted_traces.shp

Usage
-----

Training and Validation data setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Training and validation data for training are given by putting ``png``-images,
traces (labels) and target areas (bounds) to specific directories
within a chosen working directory. ``alsa`` will link the images to associated
traces and areas using the filenames. Directory names and structure
are defined in ``alsa/crack_train.py``.

Training data:

-  Images for training: ``Training/Images/Originals``
-  Traces for training: ``Training/Shapefiles/Labels``
-  Areas for training: ``Training/Shapefiles/Areas``

Validation data:

-  Images for validation: ``Validation/Images/Originals``
-  Traces for validation: ``Validation/Shapefiles/Labels``
-  Areas for validation: ``Validation/Shapefiles/Areas``

Images are linked by taking the stem of the image filename and checking if the
traces and area filenames contain that stem. E.g. if the image filename is
``kl5.png``, the stem is ``kl5`` and trace and area filenames must contain that
stem. E.g. ``kl5_traces.shp`` and ``kl5_area.shp`` will get matched. Be careful
in naming the files as checks for duplicate pairing is not implemented.

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
   predicted_output_path = Path("C:/alsa-working-directory/predicted_traces.shp")

   # ridge-detection configuration can be overridden
   # see alsa.signal_proc.DEFAULT_RIDGE_CONFIG
   # for default values
   override_ridge_config = {
         "optional_parameters": {"Line_width": 3}
   }

   # Run prediction
   crack_main.crack_main(
       work_dir=work_dir,
       img_path=img_path,
       area_file_path=area_file_path,
       unet_weights_path=unet_weights_path,
       predicted_output_path=predicted_output_path,
       width=256,
       height=256,
       override_ridge_config=override_ridge_config,
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

Furthermore, if the working directory contains a ``ridge_config.json`` file, it
will be read for configuration of ``ridge-detection``. See below:

Prediction ridge-detection Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both from the Python and command-line interface you can pass configuration to
the post-processing ``ridge-detection`` functions calls. You can create a
``json`` file with the wanted configuration overrides. Passing a file rather
than command-line options was chosen as the configuration that can be passed to
``ridge-detection`` is extensive. See ``alsa.signal_proc.DEFAULT_RIDGE_CONFIG``
for the default config that is passed to ``ridge-detection``. New options can
be set or old ones overridden within a ``json`` file, e.g.

.. code:: json

   {
     "optional_parameters": {
       "Line_width": 15
     }
   }

By default this configuration is looked for in
``<work_dir>/ridge_config.json``. If it is missing the default
configuration (``DEFAULT_RIDGE_CONFIG``) is used without overrides.

Old Usage Guide (old & partly deprecated)
-----------------------------------------

For both CrackTrain and CrackMain:

-  Extract a .png image of the area to be analyzed

   -  This image should have black background

   -  If this image is used for training, the quality of the image should be
      same across the images

   -  If this image is used for prediction, the quality of the image should be around
      the same as used for the training

   -  This image needs to be the smallest rectangle that covers the area

   -  THE NAME OF THIS .PNG IMAGE MUST BE A SUBSTRING OF THE SHAPEFILES

      -  If the name of the .png image is 'ABC123.png', the shapefiles must have 'ABC123' 
         in their filenames somewhere.

      -  For this reason, if you have shapefiles named 'abc_1.shp' and 'abc_2.shp',
         don't name the .png image as 'abc.png' as it can confuse the 2 shapefiles.

-  Install the packages described in the requirements.txt

For prediction:

-  The program first asks for the .png image's relative or full path
   (including the .png at the end). Type it in.

-  The program then asks for the path to the .shp-file containing the polygon of the
   area to be analyzed.

-  The program then asks for the path to the .hdf5-file containing the weights
   of the CNN-model. By default, this is named 'unet-weights.hdf5'.
   If not found, try to train model first.

-  Finally the program asks for the name of the .shp-file to be produced.

For training:

-  The CrackTrain looks for Training folder and contents within it.
   If this is missing, run the module once and it creates them.

-  Navigate to Training\Shapefiles

   -  \Areas should contain the .shp files containing the polygon of the area
      to be analyzed.

   -  \Labels should contain the .shp files containing the lines you wish
      the program detects.

-  Navigate to Training\Images\Originals

   -  Place the .png images you wish to train for in here.

-  THE FOLDER Training\Images\Generated IS CLEARED AT THE START OF THE PROGRAM!
   DO NOT STORE ANYTHING HERE!

-  Running the CrackTrain module will create/overwrite a file named
   'unet_weights.hdf5'. This is the file that's to be used when predicting.

Proposed improvements by Jonne (2020-2021)
------------------------------------------

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

Proposed improvements by Nikolas (2022)
---------------------------------------

-  Refactor the training and validation directory setup
   so that filepaths to both can be passed in a config file
   rather than explicitly putting them in set directories which
   is cumbersome.

-  Refactor ``CrackNetWork`` code as it is slow and complicated.
   However, it works, so it might not be a priority.

-  Find alternatives to ``ridge-detection`` or create a fork
   of that project and modify the source code to fit best coding
   practices.

-  Make the code installable as a ``Python`` package. This is
   easy when installing with ``pip`` (or ``poetry``) but less
   so when using ``conda``. Dependency specification in ``pyproject.toml``
   must match ``conda`` environment.

-  Configuration for training and prediction can be passed
   from command-line and from a ``json`` file for ridge-detection
   post-processing. Maybe all configuration could
   be passed from a single ``json`` file? Currently there's
   opportunity for confusion...
