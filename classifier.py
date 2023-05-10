# import the necessary packages
import os
from skimage import feature
import numpy as np
from sklearn.svm import LinearSVC
import argparse
import cv2
import matplotlib.pyplot as plt
from skimage import feature, exposure, filters
from skimage import io
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import pandas as pd



# Create an argument parser for passing the directories of the train, validation, and test datasets.
parser = argparse.ArgumentParser(description='Argument parser for specifying directories of train, validation, and test datasets.')
parser.add_argument('path_train', type=str)
parser.add_argument('path_val', type=str)
parser.add_argument('path_test', type=str)
args = parser.parse_args()

# Assign the directory paths to variables.
train_dir = args.path_train
val_dir = args.path_val
test_dir = args.path_test

def lbp_image(imagePath,n,r):
    """
    Compute Local Binary Pattern (LBP) features of the given image.

    Args:
        imagePath: Path of the image to compute LBP features.
        n: Number of points in the circularly symmetric neighborhood.
        r: Radius of the circle.

    Returns:
        A list containing the LBP image features after equalizing and smoothing the original image.

    """
    # Load the image
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute the LBP features
    lbp = feature.local_binary_pattern(image, n, r, method='uniform')

    # Rescale the LBP image to a larger size
    lbp_rescaled = np.uint8(255 * lbp / np.max(lbp))

    # Apply histogram equalization to the LBP image
    lbp_equalized = exposure.equalize_hist(lbp_rescaled)

    # Apply Gaussian filtering to the equalized LBP image
    lbp_smoothed = filters.gaussian(lbp_equalized, sigma=2)
    
    # Resize the smoothed LBP image to a fixed size
    lbp_smoothed = cv2.resize(lbp_smoothed, (120, 120))

    # Return the LBP image features as a list
    return list(lbp_smoothed)


def get_batch(dir_path,n,r):
    """
    Generate a batch of LBP images and their corresponding labels from the given directory path.

    Args:
        dir_path: Path of the directory containing the images.
        n: Number of points in the circularly symmetric neighborhood.
        r: Radius of the circle.

    Returns:
        Two lists containing the LBP image features and their corresponding labels.

    """
    genders=os.listdir(dir_path)
    batch_images=[]
    batch_labels=[]
    dict_classes={'male':1,'female':0}

    # Iterate over each gender directory in the given directory path
    for gender in genders:
        images_gender_path=os.path.join(dir_path, gender)
        images_in_gender=os.listdir(images_gender_path)

        # Iterate over each image in the gender directory
        for image in images_in_gender:
            image_path=os.path.join(images_gender_path, image)

            # Compute the LBP image features of the current image
            image = lbp_image(image_path,n,r)

            # Append the LBP image features and their corresponding label to the batch lists
            batch_images.append(image)
            batch_labels.append(int(dict_classes[gender]))

    # Return the batch lists
    return batch_images, batch_labels




def permutate(nparr1, nparr2):
    """
    Randomly permute two numpy arrays in the same order using permutation indices.

    Args:
    nparr1: Numpy array of shape (n_samples, n_features).
    nparr2: Numpy array of shape (n_samples,).

    Returns:
    Tuple of two numpy arrays that have been randomly permuted in the same order.
    """
    # Get the permutation indices
    permutation = np.random.permutation(len(nparr1))

    # Apply the permutation to both arrays
    nparr1 = nparr1[permutation]
    nparr2 = nparr2[permutation]
    return nparr1, nparr2


def reshape_data(images):
    """
    Reshape the image data from 3D to 2D numpy array.

    Args:
    images: Numpy array of shape (n_samples, n_rows, n_cols).

    Returns:
    Reshaped numpy array of shape (n_samples, n_rows * n_cols).
    """
    nsamples, nx, ny = images.shape
    data = images.reshape((nsamples, nx*ny))
    return data


def test_all_params(n, r):
    """
    Test different combinations of SVM parameters (C and gamma) and kernels (linear and RBF)
    to determine the best combination of parameters for classification.

    Args:
    n: Number of neighbors for Local Binary Patterns algorithm.
    r: Radius for Local Binary Patterns algorithm.

    Returns:
    Dictionary with results of each parameter combination, including kernel, C value, gamma value,
    and accuracy on the validation set.
    """
    # Get train batch
    batch_train_image, batch_train_label = get_batch(train_dir, n, r)
    batch_train_image = np.asarray(batch_train_image)
    batch_train_label = np.asarray(batch_train_label, dtype=float)

    X_train = batch_train_image
    Y_train = batch_train_label

    # Permute the training data
    X_train, Y_train = permutate(X_train, Y_train)

    # Reshape the training data
    X_train = reshape_data(X_train)

    batch_val_image, batch_val_label = get_batch(val_dir, n, r)
    batch_val_image = np.asarray(batch_val_image)
    batch_val_label = np.asarray(batch_val_label, dtype=float)

    X_val = batch_val_image
    Y_val = batch_val_label

    # Permute the validation data
    X_val, Y_val = permutate(X_val, Y_val)

    # Reshape the validation data
    X_val = reshape_data(X_val)

    # Define a dictionary to store results
    results_dict = {}
    index = 0

    # Test different combinations of SVM parameters and kernels
    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001, 0.00001, 10]}
    different_kernels = ['linear', 'rbf']
    print("Testing params...")
    for kernel in different_kernels:
        for c_param in param_grid['C']:
            for g_param in param_grid['gamma']:
                # Create SVM classifier based on RBF kernel.
                clf = svm.SVC(kernel=kernel, C=c_param, gamma=g_param)
                # Train the classifier
                clf.fit(X_train, Y_train)
                # Calculate accuracy on the validation set
                Acc = clf.score(X_val, Y_val)
                # Store the results in a dictionary
                results_dict[index] = [kernel, c_param, g_param, Acc]
                index += 1
    # Return the dictionary          
    return results_dict


# Test SVM model accuracy with different sets of hyperparameters
dict_16_3 = test_all_params(16, 3)  # Test on batch size 16 and resize factor 3
dict_8_3 = test_all_params(8, 3)  # Test on batch size 8 and resize factor 3

def get_max_acc(rslt_dict):
    """
    Given a dictionary that maps each set of hyperparameters to its corresponding test accuracy,
    return the set of hyperparameters that yields the highest accuracy.
    
    Args:
        rslt_dict (dict): Dictionary that maps each set of hyperparameters to its corresponding
                          test accuracy
        
    Returns:
        list: List of hyperparameters that yields the highest accuracy
    """
    lists_of_params = list(map(lambda x: x[3], rslt_dict.values()))
    index_of_highest_acc = lists_of_params.index(max(lists_of_params))
    list_of_index = rslt_dict[index_of_highest_acc]
    return list_of_index

# Load test data
batch_test_image, batch_test_label = get_batch(test_dir, 8, 3)  # Load test data with batch size 8 and resize factor 3

# Preprocess test data
batch_test_image = np.asarray(batch_test_image)
batch_test_label = np.asarray(batch_test_label, dtype=float)
X_test = batch_test_image
Y_test = batch_test_label
X_test, Y_test = permutate(X_test, Y_test)  # Randomly permute the test data
X_test = reshape_data(X_test)  # Reshape the test data to a 2D array

# Load train data
batch_train_image, batch_train_label = get_batch(train_dir, 8, 3)  # Load train data with batch size 8 and resize factor 3

# Preprocess train data
batch_train_image = np.asarray(batch_train_image)
batch_train_label = np.asarray(batch_train_label, dtype=float)
X_train = batch_train_image
Y_train = batch_train_label
X_train, Y_train = permutate(X_train, Y_train)  # Randomly permute the train data
X_train = reshape_data(X_train)  # Reshape the train data to a 2D array

# Train SVM model with selected hyperparameters
clf_test = svm.SVC(kernel='rbf', C=1, gamma=0.01)  # Create SVM classifier with RBF kernel and hyperparameters C=1, gamma=0.01
clf_test.fit(X_train, Y_train)  # Train SVM classifier on training data

# Evaluate performance on test data
predicted_label = clf_test.predict(X_test)  # Predict labels for test data
true_label = Y_test  # Ground truth labels for test data
tn, fp, fn, tp = confusion_matrix(predicted_label, true_label).ravel()  # Compute confusion matrix
test_acc = (tp + tn) / (tp + tn + fp + fn)  # Compute test accuracy


def save_results_to_txt(results_dict_16_3,n1,r1,results_dict_8_3,n2,r2):
    """
    This function saves the results of the SVM classification to a text file named "results.txt". 
    It takes two result dictionaries as inputs, which are the output of the "test_all_params" function for different radii and neighbor sizes.
    The function writes the following information to the text file:
      - The radius and neighbor size used for the first set of results (corresponding to the "results_dict_16_3" dictionary).
      - For each set of SVM parameters in "results_dict_16_3", it writes the corresponding accuracy, kernel type, value of C and Gamma to the text file.
      - The SVM parameter set that yields the highest accuracy for "results_dict_16_3" is also written to the file.
      - The same information is written for the second set of results (corresponding to the "results_dict_8_3" dictionary).
      - The accuracy and confusion matrix for the test set is also written to the file.
      
    Args:
    - results_dict_16_3: a dictionary containing the results of the SVM classification with radius 16 and 3 neighbors.
    - n1: an integer representing the radius used for the first set of results (corresponding to the "results_dict_16_3" dictionary).
    - r1: an integer representing the number of neighbors used for the first set of results.
    - results_dict_8_3: a dictionary containing the results of the SVM classification with radius 8 and 3 neighbors.
    - n2: an integer representing the radius used for the second set of results (corresponding to the "results_dict_8_3" dictionary).
    - r2: an integer representing the number of neighbors used for the second set of results.
    
    Returns:
    - None: The function only writes the results to a text file.
    """
    # open the text file in write mode
    with open('results.txt', 'w') as f:
        # write the values of radius and neighbors for the first model to the text file
        f.write(f'Radius={n1}\tNeighbors={r1}\n')
        # iterate over the values of the first dictionary to write the accuracy, kernel, C and gamma values to the text file
        for lst in list(results_dict_16_3.values()):
            f.write(f'Accuracy: {"%.2f" % lst[3]}%,\t\tkernel: {lst[0]}\t\tC: {lst[1]},\t\tGamma: {lst[2]}\n')
        # get the index of the maximum accuracy value for the first model and write the corresponding values to the text file
        list_of_index=get_max_acc(dict_16_3)
        f.write(f'Highest Accuracy: {"%.2f" % list_of_index[3]}%,\t\tkernel: {list_of_index[0]}\t\tC: {list_of_index[1]},\t\tGamma: {list_of_index[2]}\n\n\n')

        # write the values of radius and neighbors for the second model to the text file
        f.write(f'Radius={n2}\tNeighbors={r2}\n')
        # iterate over the values of the second dictionary to write the accuracy, kernel, C and gamma values to the text file
        for lst in list(dict_8_3.values()):
            f.write(f'Accuracy: {"%.2f" % lst[3]}%,\t\tkernel: {lst[0]}\t\t\tC: {lst[1]},\t\tGamma: {lst[2]}\n')
        # get the index of the maximum accuracy value for the second model and write the corresponding values to the text file
        list_of_index=get_max_acc(dict_8_3)
        f.write(f'Highest Accuracy: {"%.2f" % list_of_index[3]}%,\t\tkernel: {list_of_index[0]}\t\tC: {list_of_index[1]},\t\tGamma: {list_of_index[2]}\n\n\n\n')

        # write the test accuracy and the kernel, C and gamma values for the test model to the text file
        f.write(f'Test Accuracy: {"%.2f" % test_acc}%,\tkernel: rbf\t\tC: 1,\tGamma: 0.01\n\n')
        # write the confusion matrix to the text file
        f.write(f'Confusion Matrix:\n\t\tMale\tFemale\nMale\t{tp}\t\t{fn}\nFemale\t{fp}\t\t{tn}')

# Call the save_results_to_txt function with the required arguments
save_results_to_txt(dict_16_3,16,3,dict_8_3,8,3)
