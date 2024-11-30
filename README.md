# Comparison of handwritten signatures

## Project structure:

- [models][df1] - the folder that stores the trained files for all models (cleaner, detector, extractor)
- [signver][df1] - the folder that stores the base code for cleaner, detector and extractor models
- [batch_sign_test.py][df1] - Python file that provides a batch calculation of distances for different signatures
- [extractor_model_testing.py][df1] - Python file that provides tests for the extractor model on a test datasets
- [create_csv.py][df1] - Python file that gives the possibilty to create a CSV file required for a dataset inference
- [sign_test.py][df1] - Python file intended for testing signatures in manual mode

## Requirements
Requirements are given in [requirements.txt][df1]
To install the dependencies:

```sh
conda install --yes --file requirements.txt
```

## Description of [batch_sign_test.py] functions
- ### [batch_only_signature_matcher] 
    Compare already extracted signatures stored in separate files using initial (old) extractor model
  
    INPUTS:
    - reference_folder_path: path to a folder with reference signatures
    - match_folder_path: path to a folder with signatures to be compared with reference signatures
    - extractor_model_path: path to the extractor model. DEFAULT = "models/extractor/metric"
    - cleaner_model_path: path to the cleaner model. DEFAULT = "models/cleaner/small".
  
    OUTPUTS:
  
    Print to a console the distances between each signature in the reference folder and each signature in the match folder

- ### [batch_signature_matcher] 
    Compare signatures within the document contexts. Before comparison, signatures are extracted from the documents
  
    INPUTS:
    - reference_folder_path: path to a folder with reference signatures
    - match_folder_path: path to a folder with signatures to be compared with reference signatures
    - detector_model_path: path to the detector model. DEFAULT = "models/detector/detection.onnx",
    - extractor_model_path: path to the extractor model. DEFAULT = "models/extractor/metric"
    - cleaner_model_path: path to the cleaner model. DEFAULT = "models/cleaner/small"
  
    OUTPUTS:
  
    Print to console the distances between each signature in the reference folder and each signature in the match folder

- ### [batch_only_signature_matcher_rev2]
    The same as a [batch_only_signature_matcher] but for a new trained model.

- ### [batch_signature_matcher_rev2] 
    The same as a [batch_signature_matcher] but for a new trained model.


## Description of [create_csv.py] functions
- ### [create_csv]
  Create CSV file from a signature dataset folder

   INPUTS:
  - dataset_folder - path to a dataset folder with structure, like: "name", "name_forg", where:
  
    - "name"       - is the name of the folder with original signatures
  
    - "name_forg"  - is the name of the folder with forgery signatures

  - csv_path - path to a target CSV file
 
   OUTPUTS:
  
   Create a CSV file in a corresponding location


## Description of [extractor_model_testing.py] functions
- ### [get_model_distribution]
  Get model TRUE/FALSE distribution on a test dataset

   INPUTS:
  - model   - model to be processed
  - dataset - dataset (an exemplar of SignatureLabelDataset)
  - device  - 'cuda:0' or 'cpu'
 
   OUTPUTS:
  
   Stores a distribition graph in model_results.jpeg file

- ### [test_model_accuracy]
  Test model accuracies (in defined range) on a test dataset.
 
  Gives the possibility to select the best threshold in order to receive the best accuracy on a given dataset.

   INPUTS:
  - model   - model to be processed
  - dataset - dataset (an exemplar of SignatureLabelDataset)
  - device  - 'cuda:0' or 'cpu'
  - batch_size       - batch size used for computation (any int number, limited only by the capacity of device)
  - threshold_range - range of threshold deviation (min, max)
  - threshold_step   - step of threshold change within a threshold_range

   OUTPUTS:
  
   Print in a console a list of accuracies for each threshold step


  

  
  
