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



parser = argparse.ArgumentParser(description='')
parser.add_argument('path_train', type=str)
parser.add_argument('path_val', type=str)
parser.add_argument('path_test', type=str)
args = parser.parse_args()


train_dir = args.path_train
val_dir = args.path_val
test_dir = args.path_test




def lbp_image(imagePath,n,r):
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
    lbp_smoothed=cv2.resize(lbp_smoothed,(120,120))
    return list(lbp_smoothed)




def get_batch(dir_path,n,r):
    print("1")
    genders=os.listdir(dir_path)
    batch_images=[]
    batch_labels=[]
    dict_classes={'male':1,'female':0}
    for gender in genders:
        images_gender_path=os.path.join(dir_path, gender)
        images_in_gender=os.listdir(images_gender_path)
        for image in images_in_gender:
            image_path=os.path.join(images_gender_path, image)
            image = lbp_image(image_path,n,r)
            batch_images.append(image)
            batch_labels.append(int(dict_classes[gender]))
    return batch_images,batch_labels




def permutate(nparr1,nparr2):
    # Get the permutation indices
    permutation = np.random.permutation(len(nparr1))

    # Apply the permutation to both arrays
    nparr1 = nparr1[permutation]
    nparr2 = nparr2[permutation]
    return nparr1,nparr2




def reshape_data(images):
    nsamples, nx, ny = images.shape
    data = images.reshape((nsamples,nx*ny))
    return data




def test_all_params(n,r):

    #get train  batch
    batch_train_image,batch_train_label=get_batch(train_dir,n,r)
    batch_train_image=np.asarray(batch_train_image)
    batch_train_label=np.asarray(batch_train_label,dtype=float)

    X_train=batch_train_image
    Y_train=batch_train_label

    X_train,Y_train=permutate(X_train,Y_train)

    X_train=reshape_data(X_train)

    batch_val_image, batch_val_label=get_batch(val_dir,n,r)
    batch_val_image=np.asarray(batch_val_image)
    batch_val_label=np.asarray(batch_val_label,dtype=float)

    X_val=batch_val_image
    Y_val=batch_val_label

    X_val,Y_val=permutate(X_val,Y_val)
    X_val=reshape_data(X_val)

    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001, 0.00001, 10]}
    different_kernels=['linear','rbf']
    results_dict={}
    index=0
    print("testing params")
    for kernel in different_kernels:
        for c_param in param_grid['C']:
            for g_param in param_grid['gamma']:
                # Create SVM classifier based on RBF kernel.
                clf=svm.SVC(kernel=kernel, C = c_param, gamma=g_param)
                #Train the classifier
                clf.fit(X_train,Y_train)
                Acc=clf.score(X_val,Y_val)
                results_dict[index]=[kernel,c_param,g_param,Acc]
                index+=1
    return results_dict






dict_16_3=test_all_params(16,3)
dict_8_3=test_all_params(8,3)


def get_max_acc(rslt_dict):
    lists_of_params=list(map(lambda x: x[3],rslt_dict.values()))
    index_of_highest_acc=lists_of_params.index(max(lists_of_params))
    list_of_index=rslt_dict[index_of_highest_acc]
    return list_of_index






#C: 1, Gamma: 0.01,kernel=RBF, n=8, r=3
batch_test_image, batch_test_label=get_batch(test_dir,8,3)


batch_test_image=np.asarray(batch_test_image)
batch_test_label=np.asarray(batch_test_label,dtype=float)

X_test=batch_test_image
Y_test=batch_test_label

X_test,Y_test=permutate(X_test,Y_test)
X_test=reshape_data(X_test)





#get train  batch
batch_train_image,batch_train_label=get_batch(train_dir,8,3)
batch_train_image=np.asarray(batch_train_image)
batch_train_label=np.asarray(batch_train_label,dtype=float)

X_train=batch_train_image
Y_train=batch_train_label

X_train,Y_train=permutate(X_train,Y_train)

X_train=reshape_data(X_train)



clf_test=svm.SVC(kernel='rbf', C = 1, gamma=0.01)
clf_test.fit(X_train,Y_train)
predicted_label=clf_test.predict(X_test)
true_label=Y_test


tn, fp, fn, tp = confusion_matrix(predicted_label, true_label).ravel()
test_acc=(tp+tn)/(tp+tn+fp+fn)



def save_results_to_txt(results_dict_16_3,n1,r1,results_dict_8_3,n2,r2):
    with open('results.txt', 'w') as f:
        f.write(f'Radius={n1}\tNeighbors={r1}\n')
        for lst in list(results_dict_16_3.values()):
            f.write(f'Accuracy: {"%.2f" % lst[3]}%,\t\tkernel: {lst[0]}\t\tC: {lst[1]},\t\tGamma: {lst[2]}\n')
        list_of_index=get_max_acc(dict_16_3)
        f.write(f'Highest Accuracy: {"%.2f" % list_of_index[3]}%,\t\tkernel: {list_of_index[0]}\t\tC: {list_of_index[1]},\t\tGamma: {list_of_index[2]}\n\n\n')


        f.write(f'Radius={n2}\tNeighbors={r2}\n')
        for lst in list(dict_8_3.values()):
            f.write(f'Accuracy: {"%.2f" % lst[3]}%,\t\tkernel: {lst[0]}\t\t\tC: {lst[1]},\t\tGamma: {lst[2]}\n')
        list_of_index=get_max_acc(dict_8_3)
        f.write(f'Highest Accuracy: {"%.2f" % list_of_index[3]}%,\t\tkernel: {list_of_index[0]}\t\tC: {list_of_index[1]},\t\tGamma: {list_of_index[2]}\n\n\n\n')


        f.write(f'Test Accuracy: {"%.2f" % test_acc}%,\tkernel: rbf\t\tC: 1,\tGamma: 0.01\n\n')
        f.write(f'Confusion Matrix:\n\t\tMale\tFemale\nMale\t{tp}\t\t{fn}\nFemale\t{fp}\t\t{tn}')


save_results_to_txt(dict_16_3,16,3,dict_8_3,8,3)