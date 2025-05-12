import scipy.io
import matplotlib.pyplot as plt


from warnings import filterwarnings
filterwarnings('ignore')

mat = scipy.io.loadmat('Xtrain.mat')
data = mat['Xtrain']

plt.figure(figsize=(12, 5))
plt.plot(data)
plt.title("Original Data")
plt.show()
plt.savefig("results/original_data.png")
