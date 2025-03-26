# Description: This file is used to import all the functions from the data_processing folder.

## this piece of code is from the template and will be ignored: 
# from cd4ml.data_processing.ingest_data import ingest_data
# from cd4ml.data_processing.ingest_data import get_data
# from cd4ml.data_processing.split_train_test import get_train_test_split, split_train_test
# from cd4ml.data_processing.transform_data import get_transformed_data, transform_data
# from cd4ml.data_processing.validate_data import validate_data
# from cd4ml.data_processing.track_data import track_data


########## this is the new code applied to RAKUTEN use case:
# central import for all step*_ modules in the data_processing folder
from cd4ml.data_processing.step01_combine_xy import load_combined_data
from cd4ml.data_processing.step02_text_cleaning import clean_text
from cd4ml.data_processing.step03_split_data import split_dataset
from cd4ml.data_processing.step04_tfidf_transform import apply_tfidf