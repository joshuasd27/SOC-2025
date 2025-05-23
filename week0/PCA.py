import numpy as np
import matplotlib.pyplot as plt

N=1000 
def generate_dataset1():
    x = np.random.rand(1,N)
    y = x + 0.05*np.random.rand(1,N)
    data = np.vstack([x,y])
    return data

def generate_dataset2():
    import numpy as np # Assuming numpy is available
    x = np.random.rand(1,N)
    y = (0.5 - x**2)*(0.7 - x) + 1 
    data = np.vstack([x,y])
    return data

# Example usage (for demonstration, maybe not needed in final code block)
data= generate_dataset2()
mean = data.mean(axis=1).reshape(data.shape[0], 1)
variance = np.sum((data-mean)*(data-mean), axis=1).reshape(data.shape[0], 1) / data.shape[1]
standard_deviation = np.sqrt(variance)
#generate covariance matrix from outer product of data, XX^T
data_centered= (data-mean)/standard_deviation
covariance_mat = data_centered@ data.transpose()
eigenvalues, eigenvectors = np.linalg.eig(covariance_mat)
#sort eigens in descending order of e value
indices = np.argsort(eigenvalues)[eigenvalues.shape[0]::-1]
eigenvalues = eigenvalues[indices]
eigenvectors = eigenvectors[:,indices]

#take 1 PCA from e vectors
PCA = eigenvectors[:,0].reshape(data.shape[0],1) #in standardized units
PCA = (PCA*standard_deviation) # in orig data units,
PCA = PCA/ np.linalg.norm(PCA) #  normalized to unit vector again
print("Unit vector of PCA" + str(PCA))
compressed_values_along_PCA = (PCA.transpose() @ (data-mean)).reshape(1,data.shape[1])
print( f"{eigenvalues[0]/eigenvalues.sum():.2%} variance along this/these axis")

#plot 
if (data.shape[0]==2):
    plt.plot(data[0],data[1], 'bo',label= "Actual Data")
    compressed_datapoints = mean+PCA@ compressed_values_along_PCA
    plt.plot( compressed_datapoints[0], compressed_datapoints[1], 'go', label='Compressed Data Along PCA')
    m= PCA[1]/PCA[0]
    plt.plot([mean[0]- 2,  mean[0]+2], [mean[1]-2*m,mean[1]+2*m])
    plt.plot(mean[0],mean[1], 'ro', label ='Mean')
    plt.grid(1)
    plt.legend()

    xmin, xmax = min(data[0]), max(data[0])
    ymin, ymax = min(data[1]), max(data[1])
    # Calculate range
    xrange = xmax - xmin
    yrange = ymax - ymin
    # Calculate center points
    xmid = (xmax + xmin) / 2
    ymid = (ymax + ymin) / 2
    # Find the max range to keep aspect ratio 1:1
    max_range = max(xrange, yrange)
    # Add some padding (optional)
    padding = 0.05 * max_range
    # Set axis limits with padding, centered around midpoints
    plt.xlim(xmid - max_range / 2 - padding, xmid + max_range / 2 + padding)
    plt.ylim(ymid - max_range / 2 - padding, ymid + max_range / 2 + padding)
    # Lock aspect ratio to 1:1
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

