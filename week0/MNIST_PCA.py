from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt

ds = load_dataset("ylecun/mnist")
data = ds['train']
X = np.array(data['image'])
labels = np.array(data['label'])
data = X.reshape(X.shape[0],-1) #flatten image to 1d
data = data[~np.isnan(data).any(axis=1)] #removes images with NA
data = data.transpose() # get feature vectors on columns

# CONVENTION :- feature dimensions, axes vertically. Samples/observations are the columns

mean = data.mean(axis=1).reshape(data.shape[0], 1) 
# SKIPPED SCALING as all axes same units
# variance = np.sum((data-mean)*(data-mean), axis=1).reshape(data.shape[0], 1) / data.shape[1]
# standard_deviation = np.sqrt(variance)
# #lots of S.D. along axes = 0 (always dark or bright), remove those axes
# mask = (standard_deviation != 0)
# mask = mask[:, 0]  # flatten mask to 1D

# data = data[mask, :]
# mean = mean[mask, :]
# standard_deviation = standard_deviation[mask, :]


#generate covariance matrix from outer product of data, XX^T
data_centered= (data-mean)
covariance_mat = data_centered@ data_centered.transpose()
eigenvalues, eigenvectors = np.linalg.eig(covariance_mat)

#sort eigens in descending order of e value
indices = np.argsort(eigenvalues)[eigenvalues.shape[0]::-1]
eigenvalues = eigenvalues[indices].reshape(1,eigenvalues.shape[0])
eigenvectors = eigenvectors[:,indices]

#scree plot of e values
Y=eigenvalues.flatten()/eigenvalues.sum()
X = np.arange(1, eigenvalues.shape[1] + 1)
plt.plot(X,Y,'o-', label='Explained Variance Ratio')
Y_cumulative= np.cumsum(Y)
plt.plot(X,Y_cumulative,'o-', label='Cumulative Variance')
plt.title("Scree Plot")
plt.xlabel("Principal Component")
plt.show()

#take 1 PCA from e vectors
num_PCA_axes = 20
chosen_PCAs = eigenvectors[:,:num_PCA_axes]
chosen_PCAs= chosen_PCAs.reshape(data.shape[0],num_PCA_axes) #reshape
#print("Unit vector/s of PCA (in 1e-6 units)" + np.array2string(chosen_PCAs, formatter={'float_kind': lambda x: f"{x*1e6}:.2f"}))
compressed_values_along_PCA = (chosen_PCAs.transpose() @ (data-mean)).reshape(num_PCA_axes,data.shape[1])
print( f"{eigenvalues[:,:num_PCA_axes].sum()/eigenvalues.sum():.2f} variance along the first {num_PCA_axes} axes")
