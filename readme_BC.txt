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