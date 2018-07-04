import numpy as np
import random
import scipy.io as sio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import ticker,cm
from matplotlib.ticker import LinearLocator,FormatStrFormatter
import matplotlib.mlab as mlab
from scipy.optimize import minimize, rosen, rosen_der

def sigmoid(z):
    g=np.matrix(np.zeros(np.shape(z)))
    g=1/(1+np.exp(-1*z))
    return g

def sigmoidGradient(z):
    a=sigmoid(z)
    g=np.multiply(a,(1-a))
    return g

def predict(Theta1, Theta2, X):
    (m,n)=np.shape(X)
    p=np.matrix(np.zeros((m,1)))
    a1=np.matrix(np.hstack((np.ones((m,1)),X)))
    a2=sigmoid(np.matmul(Theta1,a1.T))
    (m1,n1)=np.shape(a2)
    a2=np.matrix(np.vstack((np.ones((1,n1)),a2)))
    a3=sigmoid(np.matmul(Theta2,a2))
    p=np.argmax(a3.T,axis=1)
    p=p+1
    return p


def nnCostFunction(nn_parms,input_layer_size,hidden_layer_size,num_labels,X,y,lamda):
    J=0
    d_point=((input_layer_size+1)*hidden_layer_size)
    T1=(((nn_parms[0:d_point]).T).reshape((input_layer_size+1,hidden_layer_size))).T
    d_point1=((hidden_layer_size+1)*num_labels)
    T2=(((nn_parms[d_point:d_point+d_point1]).T).reshape((hidden_layer_size+1,num_labels))).T
    (m,n)=np.shape(X)
    
    ## FeedForward Part
    a1=np.matrix(np.hstack((np.ones((m,1)),X)))
    a2=sigmoid(np.matmul(T1,a1.T))
    (m1,n1)=np.shape(a2)
    a2=np.matrix(np.vstack((np.ones((1,n1)),a2)))
    a3=sigmoid((np.matmul(T2,a2)))
    
    
   
    for i in range(m):
       temp_y=np.zeros((num_labels,1))
       temp_y[y[i]-1]=1
       J=J+np.sum(np.multiply(-1*(temp_y),np.log(a3[:,i]))+np.multiply((-1*(1-temp_y)),np.log(1-a3[:,i])))
    
    J=J+( (lamda/2) * ( np.sum( np.power(T1[:,1:np.shape(T1)[1]] ,2) ) + np.sum( np.power(T2[:,1:np.shape(T2)[1]] ,2) ) ) )
    
    J=J/m

    return J






def nnGradFunction(nn_parms,input_layer_size,hidden_layer_size,num_labels,X,y,lamda):
    Grad=0
    d_point=((input_layer_size+1)*hidden_layer_size)
    T1=(((nn_parms[0:d_point]).T).reshape((input_layer_size+1,hidden_layer_size))).T
    d_point1=((hidden_layer_size+1)*num_labels)
    T2=(((nn_parms[d_point:d_point+d_point1]).T).reshape((hidden_layer_size+1,num_labels))).T
    (m,n)=np.shape(X)
    Theta1_grad = np.matrix(np.zeros(np.shape(T1)))
    Theta2_grad = np.matrix(np.zeros(np.shape(T2)))
    ## FeedForward Part
    a1=np.matrix(np.hstack((np.ones((m,1)),X)))
    z2=np.matmul(T1,a1.T)
    a2=sigmoid(z2)
    (m1,n1)=np.shape(a2)
    a2=np.matrix(np.vstack((np.ones((1,n1)),a2)))
    z3=np.matmul(T2,a2)
    a3=sigmoid(z3)
    
    for i in range(m):
        temp_y=np.zeros((num_labels,1))
        temp_y[y[i]-1]=1
        delta_3=a3[:,i]-temp_y
        #print(sigmoidGradient(np.vstack((np.ones((1,1)),z2[:,i]))))
        delta_2=np.multiply(np.matmul(T2.T,delta_3),sigmoidGradient(np.vstack((np.ones((1,1)),z2[:,i]))))
        delta_2=delta_2[1:np.shape(delta_2)[0]]
        Theta2_grad=Theta2_grad+np.matmul(delta_3,a2[:,i].T)
        Theta1_grad=Theta1_grad+np.matmul(delta_2,a1[i,:])
    Theta2_grad=Theta2_grad/m
    Theta1_grad=Theta1_grad/m
    
    temp_t1=np.matrix(np.zeros(np.shape(T1))+T1)
    temp_t2=np.matrix(np.zeros(np.shape(T2))+T2)
    (r,c)=np.shape(T2)
    temp_t2[:,0]=np.matrix(np.zeros((r,1)))
    Theta2_grad=Theta2_grad+((lamda/m)*temp_t2)
    (r,c)=np.shape(T1)
    temp_t1[:,0]=np.matrix(np.zeros((r,1)))
    Theta1_grad=Theta1_grad+((lamda/m)*temp_t1)
       
    Grad= np.matrix(np.hstack((Theta1_grad.flatten('F'),Theta2_grad.flatten('F')))) 

    return  np.squeeze(np.asarray(Grad))

def debugInitializeWeights(fan_out, fan_in):
    W = np.matrix(np.zeros((fan_out, 1 + fan_in)))
    W = np.matrix((np.sin(np.arange(1,W.size+1))/10).reshape((1+fan_in,fan_out))).T
    return W

def computeNumericalGradient(theta,input_layer_size,hidden_layer_size,num_labels,X,y,l):
    numgrad =np.matrix(np.zeros(np.shape(theta)))
    perturb =np.matrix(np.zeros(np.shape(theta)))
    ev=1e-4
    for i in range(theta.size):
        perturb[i]=ev
        loss1=nnCostFunction(theta-perturb,input_layer_size,hidden_layer_size,num_labels,X,y,l)
        loss2=nnCostFunction(theta+perturb,input_layer_size,hidden_layer_size,num_labels,X,y,l)
        numgrad[i] = (loss2 - loss1) / (2*ev)
        perturb[i]=0
    return numgrad


def checkNNGradients(l=0):
    input_layer_size_t = 3
    hidden_layer_size_t = 5
    num_labels_t = 3
    m_t = 5
    T1_t = debugInitializeWeights(hidden_layer_size_t, input_layer_size_t)
    T2_t = debugInitializeWeights(num_labels_t, hidden_layer_size_t)
    X_t  = debugInitializeWeights(m_t, input_layer_size_t - 1)
    y_t=1+(np.matrix(np.remainder(np.arange(1,m_t+1),num_labels_t))).T
    nn_parms_t=np.matrix(np.hstack((T1_t.flatten('F'),T2_t.flatten('F')))).T
    grad=nnGradFunction(nn_parms_t,input_layer_size_t,hidden_layer_size_t,num_labels_t,X_t,y_t,l)
    grad=np.matrix(grad).T
    #cost=nnCostFunction(nn_parms_t,input_layer_size_t,hidden_layer_size_t,num_labels_t,X_t,y_t,l)
    num_grad=computeNumericalGradient(nn_parms_t,input_layer_size_t,hidden_layer_size_t,num_labels_t,X_t,y_t,l)
    print("Grad :")
    print(grad)
    print("Num Grad :")
    print(num_grad)
    diff=np.linalg.norm(num_grad-grad)/np.linalg.norm(num_grad+grad)
    print("If your backpropagation implementation is correct, then the relative difference will be small (less than 1e-9).")
    print(" Relative Difference: "+str(diff))
    return

def randInitializeWeights(L_in, L_out):
    W = np.matrix(np.zeros((L_out, 1 + L_in)))
    epsilon_init = 0.12
    W = np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init
    W=np.matrix(W)
    return W



input_layer_size  = 400
hidden_layer_size = 25
num_labels = 10     


file_name="ex4data1.mat"
data_content=sio.loadmat(file_name)
X=np.matrix(data_content['X'])
y=np.matrix(data_content['y'])
(m,n)=np.shape(X)
file_name="ex4weights.mat"
param_content=sio.loadmat(file_name)

Theta1=np.matrix(param_content['Theta1'])
Theta2=np.matrix(param_content['Theta2'])

nn_parms=np.matrix(np.hstack((Theta1.flatten('F'),Theta2.flatten('F')))).T

lamda = 0
cost= nnCostFunction(nn_parms, input_layer_size, hidden_layer_size,num_labels, X, y, lamda)
print("cost of loaded parameters(0.287629) for lamda 0:")
print(cost)

lamda = 1
cost= nnCostFunction(nn_parms, input_layer_size, hidden_layer_size,num_labels, X, y, lamda)
print("cost of loaded parameters(0.383770) for lamda 1:")
print(cost)


gr=sigmoidGradient(np.matrix(np.array([-1,-0.5,0,0.5,1])))

print("Sigmoid gradient evaluated at [-1 -0.5 0 0.5 1]:")
print(gr)


checkNNGradients()
checkNNGradients(3)

print("Cost at (fixed) debugging parameters (w/ lambda = 3) (for lambda = 3, this value should be about 0.576051):")
print(nnCostFunction(nn_parms, input_layer_size, hidden_layer_size,num_labels, X, y, 3))


print("Training Neural Network................")


lamda=1

initial_T1=randInitializeWeights(input_layer_size, hidden_layer_size)
initial_T2 = randInitializeWeights(hidden_layer_size, num_labels)
initial_theta=np.matrix(np.hstack((initial_T1.flatten('F'),initial_T2.flatten('F')))).T
res = minimize(fun=nnCostFunction, x0=initial_theta, args=(input_layer_size, hidden_layer_size,num_labels,X,y,lamda), method='CG', jac=nnGradFunction,options={'maxiter': 50, 'disp': True})

FT=np.matrix(res.x).T
dp=((input_layer_size+1)*hidden_layer_size)
FT1=(((FT[0:dp]).T).reshape((input_layer_size+1,hidden_layer_size))).T
dp1=((hidden_layer_size+1)*num_labels)
FT2=(((FT[dp:dp+dp1]).T).reshape((hidden_layer_size+1,num_labels))).T

pr = predict(FT1, FT2, X)


print("Train Accuracy:")
print(np.mean(pr==y)*100)

