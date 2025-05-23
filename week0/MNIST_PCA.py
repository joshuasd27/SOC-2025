from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt

ds = load_dataset("ylecun/mnist")
data = ds['train']
train_images = np.array(data['image'])
labels = np.array(data['label'])
data = train_images.reshape(train_images.shape[0],-1) #flatten image to 1d
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
fig, axs = plt.subplots(2, 1, figsize=(8, 4))

Y=eigenvalues.flatten()/eigenvalues.sum()
X = np.arange(1, eigenvalues.shape[1] + 1)
axs[0].plot(X,Y,'o-', label='Explained Variance Ratio')
axs[0].set_xlabel("Principal Component")

Y_cumulative= np.cumsum(Y)
axs[1].plot(X,Y_cumulative,'ro-', label='Cumulative Variance')
axs[1].set_xlabel("Principal Component")

plt.title("Scree Plot")
plt.show()

#show images for inputted PCA axes taken
num_user_int = 0
user_input_buffer = []
user_image_buffer = []
#which image to show
image_idx = 1

while True:
    user_input = input("Enter no. of PCA axes to take (or type 'q' to quit): ")
    try:
        if user_input.lower() == 'q':
            print("Goodbye!")
            break  # exit the loop
        
        num_PCA_axes = int(user_input)
        num_user_int+=1
        user_input_buffer.append(user_input)
        
        chosen_PCAs = eigenvectors[:,:num_PCA_axes]
        chosen_PCAs= chosen_PCAs.reshape(data.shape[0],num_PCA_axes) #reshape
        #print("Unit vector/s of PCA (in 1e-6 units)" + np.array2string(chosen_PCAs, formatter={'float_kind': lambda x: f"{x*1e6}:.2f"}))
        compressed_values_along_PCA = (chosen_PCAs.transpose() @ (data-mean)).reshape(num_PCA_axes,data.shape[1])
        #reconstruct from PCA
        reconstructed_images = mean + chosen_PCAs @compressed_values_along_PCA
        reconstructed_images = reconstructed_images.transpose() #now feature vectors are rows, so we can reshape to normal shape below
        reconstructed_images= reconstructed_images.reshape(train_images.shape[0],train_images.shape[1], train_images.shape[2])

        #example image shown
        fig, axs = plt.subplots(1, 1+num_user_int, figsize=(8, 4))

        # Show first image
        axs[0].imshow(train_images[image_idx], cmap='gray')
        axs[0].set_title('Image 1')
        axs[0].axis('off')

        # Show user images
        for i in range(1,num_user_int):
            axs[i].imshow(user_image_buffer[i-1], cmap='gray')
            axs[i].set_title(f'{user_input_buffer[i-1]} PCA axes')
            axs[i].axis('off')

        axs[num_user_int].imshow(reconstructed_images[image_idx], cmap='gray')
        axs[num_user_int].set_title(f'{user_input_buffer[num_user_int-1]} PCA axes')
        axs[num_user_int].axis('off')

        user_image_buffer.append(reconstructed_images[image_idx])

        plt.show()
        print( f"{eigenvalues[:,:num_PCA_axes].sum()/eigenvalues.sum():.2f} variance along the first {num_PCA_axes} axes")
    except ValueError:
        print("Input is NOT an integer.")

