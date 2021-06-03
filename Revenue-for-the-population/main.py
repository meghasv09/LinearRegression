# Univariate Linear Regression
#%%
import numpy as np
from numpy.core.function_base import logspace
from numpy.core.numeric import ones
from numpy.lib.function_base import meshgrid
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns

def read_data():
    dataframe=pd.read_csv("/home/megha/Desktop/ML-Python/Revenue-for-the-population/ex1data1.txt",sep=",",skiprows=1)
    #print(dataframe)
    #print(dataframe.size)
    m=len(dataframe)
    '''X=[np.ones((m,1)),dataframe.iloc[:,0]]#dataframe.iloc[] can also be used to copy data of that column
    print(dataframe.iloc[:,0])
    y=dataframe.iloc[:,1]
    print(y)'''
    arr=dataframe.to_numpy()
    X=arr[:,0]
    X=np.reshape(X,(-1,1))
    #print(X)
    #print(np.shape(X))
    y=arr[:,1]
    y=np.reshape(y,(-1,1))
    #print(y)
    #print(np.shape(y))
    print('Reading the data..done!')
    return (X,y,m)


def normalise_data(x,y,m):
    '''x_mean=np.mean(x)
    y_mean=np.mean(y)
    x=x-np.mean(x)'''
    #since input and output are in the same range normalisation of data is not required here.

def plot_data(X,y):
    print("Plotting the data..")
    plt.scatter(X,y,marker='x', c='r', edgecolor='b')
    plt.xlabel("Population(in 10,000s)")
    plt.ylabel("profi(in $10,000)")
    

def find_cost(X,y,theta,m):
    J=0
    '''for i in range(0,m):
       J=J +((theta[0]*X[i,0]+theta[1]*X[i,1]) - y[i])**2'''
    #J= np.sum(np.square(np.matmul(X,theta) - y))  
    #k=np.matmul(X,theta)  #np.matmul() n X @ theta gives the same result as of np.dot()
    #k=X @ theta
    k=np.dot(X,theta)
    k=np.reshape(k,(-1,1))
    #print(np.shape(k))
    sub=k-y
    #print(y)
    J=np.sum(np.square(sub))
    return J/(2*m)

def grad_descent(X,y,m,theta,n_iters):
    alpha=0.01
    for i in range(0,n_iters):
            dJ0=0;
            dJ1=0;
            k=np.dot(X,theta)
            k=np.reshape(k,(-1,1))
            sub=k- y                  #basically we are doing (pred-y),in vectorised form (x*theta-y)
            dJ0=np.sum(sub)/m
            x1=X[:,1]
            x1=np.reshape(x1,(-1,1))
            x1=np.transpose(x1)
            #print(np.shape(X[:,1]),np.shape(x1))
            dJ1=(x1 @ sub)/m
            theta[0]=theta[0]-alpha*dJ0
            theta[1]=theta[1]-alpha*dJ1
            J_new=find_cost(X,y,theta,m)
            print("for theta:",theta,"cost obtained:",J_new)
    return (J_new,theta)

def visualise_cost(X,y,m,theta):
    t0=np.linspace(-10,10,100)
    t1=np.linspace(-1,4,100)
    J_plot=np.zeros((len(t0),len(t1)))
    for i in range(len(t0)):
	    for j in range(len(t1)):
		    t=[t0[i],t1[j]]
		    J_plot[i,j]=find_cost(X,y,t,m)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(t0,t1,J_plot, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
    ax.set_xlabel('theta0')
    ax.set_ylabel('theta1')
    ax.set_zlabel('J(theta0,theta1)');
    plt.show()
    plt.contour(t0,t1,J_plot, extent=[-10,10,-2,5], cmap='RdGy');
    plt.plot(theta[0],theta[1],'bx')
    plt.xlabel('theta0')
    plt.ylabel('theta1')
    plt.show()


def main():
    #Read the data
    x,y,m=read_data()
    #Plot the data
    plot_data(x,y)
    plt.show()
    X=np.concatenate((np.ones((m,1)),x),axis=1)
    #print(type(X))
    #print(X,np.shape(X))
    theta=np.zeros((2,1))
    # Compute cost function for initial theta values
    J=find_cost(X,y,theta,m)
    print("cost value:",J)
    #Compute Gradient Descent and find Cost for the theta values obtained after performing GD
    J_new,theta=grad_descent(X,y,m,theta,n_iters=1500)
    print('theta values for line of best fit:',theta)
    #ploting line of best fit
    plot_data(x,y)
    plt.plot(X[:,1],(X @ theta))
    plt.legend(["Linear Regressio","Training Data",])
    plt.show()
    #Visualize Cost Function 
    print('Visualising the data...')
    visualise_cost(X,y,m,theta)


if __name__== "__main__":
            main()
# %%
