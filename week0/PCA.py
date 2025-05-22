import numpy as np
import matplotlib.pyplot as plt

N=100
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
standard_deviation = np.sum(data*data, axis=1).reshape(data.shape[0], 1)

data_centered= (data-mean)/standard_deviation
covariance_mat = data_centered@ data.transpose()
eigenvalues, eigenvectors = np.linalg.eig(covariance_mat)

indices = np.argsort(eigenvalues)[eigenvalues.shape[0]::-1]
eigenvalues = eigenvalues[indices]
eigenvectors = eigenvectors[:,indices]

#take PCA
PCA = eigenvectors[:,0]
print("Unit vector of PCA" + str(PCA))
compressed_data = PCA.transpose() @ data_centered
print( f"{eigenvalues[0]/eigenvalues.sum():%} variance along this axis")

#plot
plt.plot(data[0],data[1], 'bo')
m= PCA[1]/PCA[0]
plt.plot([mean[0]- 2,  mean[0]+2], [mean[1]-2*m,mean[1]+2*m])
plt.plot(mean[0],mean[1], 'ro')
plt.grid(1)
plt.xlim(np.min(data[0]), np.max(data[0]))
plt.ylim(np.min(data[1]), np.max(data[1]))
plt.show()
# [x2, y2] = generate_dataset2()
