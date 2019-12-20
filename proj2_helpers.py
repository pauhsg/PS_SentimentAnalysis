#!/usr/bin/env python3
import csv
import pickle


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})
        
        
def open_pickle_file(file_path):    
    '''
    Opens a pickle file and and loads it into a variable
    '''
    with open(file_path, 'rb') as f:
        obj = pickle.load(f)
    return obj

def create_submission(df_test, y_pred, submission_path):
    '''
    Uses the DataFrame of the test data to get submissions id, 
    verifies length of the submission (should be 10'000 in our case)
    and finally creates the .csv submission using the ids and y_pred
    '''
    test_id = df_test['Tweet_submission_id'].to_numpy()
    print('verify length of test: ', len(test_id))
    create_csv_submission(test_id, y_pred, submission_path)

    
    