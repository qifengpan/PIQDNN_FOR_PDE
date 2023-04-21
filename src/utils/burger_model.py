from model import *
from qiskit import QuantumCircuit, transpile, Aer, IBMQ
from qiskit.tools.jupyter import *
from qiskit.visualization import *
#from ibm_quantum_widgets import *
from qiskit.providers.aer import QasmSimulator
from qiskit import QuantumCircuit, Aer
from qiskit.algorithms.optimizers import COBYLA
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap, ZZFeatureMap, RealAmplitudes
from qiskit.opflow import AerPauliExpectation, PauliSumOp
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit_machine_learning.algorithms.regressors import NeuralNetworkRegressor
from qiskit_machine_learning.neural_networks import TwoLayerQNN, EstimatorQNN
from qiskit.algorithms.optimizers import L_BFGS_B,ADAM,GradientDescent
from qiskit.circuit import Parameter
from qiskit_machine_learning.utils.loss_functions.loss_functions import Loss

import matplotlib.pyplot as plt
import numpy as np
import math


#@qifeng: add physical loss part here.
class PhysicalLossBurger(Loss):
    def __init__(self,m,mu,stept,stepx,lenx,lent,xmax,umax,tmax):
        self.m  = m
        self.mu = mu
        self.stepx  = stepx
        self.stept  = stept
        self.lenx= lenx
        self.lent= lent
        self.xmax= xmax
        self.tmax= tmax
        self.umax= umax
    def __second_order_central(self,xb,x,xa,step):
        return (xb+xa-2*x)/step**2
    def __first_order_central(self,xb,xa,step):
        return (xa-xb)/(2*(step))
    def __first_order_upwind(self,xb,x,step):
        return (x-xb)/step
    def __phy_Inner_loss(self,u,ubt,ubx,uax):
        # loss = a_x*u_x + a*u_xx + a_y*u_y + a*u_yy + f
        # abx means a(x-h), aax ,means a(x+h), analogous for y
        u_t  = self.__first_order_upwind(u,ubt,self.stept)
        u_x  = self.__first_order_central(ubx,uax,self.stepx)
        u_xx = self.__second_order_central(ubx,u,uax,self.stepx)

        return (self.tmax/self.umax* u_t + self.tmax/self.umax**2*self.m*u*u_x - self.xmax**2/self.umax * self.mu*u_xx) 
    def evaluate(self, predict: np.ndarray, target: np.ndarray) -> np.ndarray:
        # evaluate L2 Loss first |u-u_predict|_2
        self._validate_shapes(predict, target)
        if len(predict.shape) <= 1:
            L2_loss = (predict - target)**2
        else:
            L2_loss = np.linalg.norm(predict - target, ord = 1,axis=tuple(range(1, len(predict.shape))))**2
        # reshape to the mesh
        u = predict.reshape(self.lenx,self.lent)
        Phy_loss_inner = np.zeros([self.lenx,self.lent])
        # evaluate physical loss for inner and boundary area
        for i in range(self.lenx):
            for j in range(self.lent):
                if(i == self.lenx-1 or j == self.lent -1 or i ==0 or j == 0):
                    pass
                else:
                    Phy_loss_inner[i,j] = self.__phy_Inner_loss(u[i,j],u[i-1,j],u[i,j-1],u[i,j+1])
        return 0.99999*L2_loss + 0.00001*Phy_loss_inner.reshape(-1,1)
    def gradient(self, predict: np.ndarray, target: np.ndarray) -> np.ndarray:
        # For a given point, it has value u, with loss u_loss, the gradient is define as D(u_loss)/D(u) if we consider the
        # loss is a function of the value of current posint. 
        # For L1 norm, it is easy to derive that gradient is sign(u_predict-target).
        # For physical loss, specially for inner loss, analytical solution is not easy to derive, then we calculate 
        # the gradient via:
        # (u_loss(u)-u_loss(u+delta*u))/delta*u
        self._validate_shapes(predict, target)
        u = predict.reshape(self.lenx,self.lent)
        Phs_loss_inner_gradient = np.zeros([self.lenx,self.lent])
        delta = 0.01
        for i in range(self.lenx):
            for j in range(self.lent):
                if(i == self.lenx-1 or j == self.lent -1 or i ==0 or j == 0):
                    pass
                else:
                    orgi = self.__phy_Inner_loss(u[i,j],u[i-1,j],u[i,j-1],u[i,j+1])
                    disturbed = self.__phy_Inner_loss(u[i,j]*(1+delta),u[i-1,j],u[i,j-1],u[i,j+1])
                    Phs_loss_inner_gradient[i,j] = (disturbed - orgi) / (delta*u[i,j])
        return 1 * 0.99999*(predict - target) + 0.00001*Phs_loss_inner_gradient.reshape(-1,1)#*Phs_loss_inner.reshape(self.N*self.N,1)

class BurgerModel():
    def __init__(self,Losstype,dimension,MaxIter,m,mu):
        self.Losstype = Losstype
        self.dim = dimension
        self.MaxIter = MaxIter
        self.x_input = []
        self.y_output = []
        self.m = m
        self.mu = mu
        self.__preprocessing()
        self.__generate_NN()
    def __preprocessing(self):
        number_of_samples = 2000
        noise_level = 0.1
        path = "../Data/Burgers/"
        x= np.load(path + 'burgers_x'+'_'+str(number_of_samples)+'.npy')
        t= np.load(path + 'burgers_t'+'_'+str(number_of_samples)+'.npy')
        u= np.array(np.load(path + 'burgers_u'+'_'+str(number_of_samples)+'.npy'),dtype=np.float32).reshape(len(x),len(t))
        self.stepX = x[-1]-x[-2]
        self.stepT = t[-1]-t[-2]
#        noise_l = noise_level*np.std(u)*np.random.randn(u.shape[0],u.shape[1])
#        u = u + noise_l
        self.lenx = len(x)
        self.lent = len(t)
        x_grid,  t_grid = np.meshgrid(x,  t, indexing="ij")
        xs = x_grid.reshape(-1,1)
        ts = t_grid.reshape(-1,1)

        self.xmax = x.max()
        self.umax = u.max()
        self.tmax = t.max()

#        idx = resample(np.arange(u1_noisy.ravel().shape[0]),replace=False,random_state=44,n_samples=number_of_samples)
#
#        u1_noisy_shuffled = (u1_noisy.ravel())[idx]
#        xs = ((x_grid.ravel())[idx]).reshape(-1,1)
#        ts = ((t_grid).ravel()[idx]).reshape(-1,1)
#
#        scale_from_outputs = []
        u_d = []
#
#        # scaling and subsampling
#        u_sampled = u1_noisy_shuffled.reshape(-1,1)
#        u_max = u_sampled.max()
#        scale_from_outputs.append(u_max)
        u_d.append(u/u.max())   


        # Data Normalization
        X_data = np.concatenate([ts/t.max(),xs/x.max()],axis=1)
        y_data = np.concatenate(u_d,axis=0)

#        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

        self.x_input = X_data
        self.y_input = y_data

#        self.x_test = X_test
#        self.y_test = y_test


    def __generate_NN(self):
        #quantum_instance = QuantumInstance(Aer.get_backend("statevector_simulator"), shots=1024)


        feature_map = ZFeatureMap(2)
        # feature_map = RealAmplitudes(5,reps = 3)
        ansatz = QuantumCircuit(2, name="Ansatz")

        ansatz = QuantumCircuit(2, name="Ansatz")

        # First Convolutional Layer
        ansatz.compose(conv_layer(2, "с1"), list(range(2)), inplace=True)

        # First Pooling Layer
        ansatz.compose(pool_layer([0], [1], "p1"), list(range(2)), inplace=True)

        # Second Convolutional Layer
        ansatz.compose(conv_layer(2, "c2"), list(range(2)), inplace=True)

        # Second Pooling Layer
        ansatz.compose(pool_layer([0], [1], "p3"), list(range(2)), inplace=True)

        # Combining the feature map and ansatz
        circuit = QuantumCircuit(2)
        circuit.compose(feature_map, range(2), inplace=True)
        circuit.compose(ansatz, range(2), inplace=True)

        #observable = PauliSumOp.from_list([("Z" + "I" * 7, 1)])

        # specify the observable
        observable = PauliSumOp.from_list([("Z" * 2, 1)])

        self.qnn =EstimatorQNN(
            circuit=circuit.decompose(),
            observables=observable,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
        )

    def generate_Regressor(self):
        if(self.Losstype=="sq_error"):
            regressor = NeuralNetworkRegressor(
            neural_network=self.qnn,
            loss="squared_error",
            optimizer=L_BFGS_B(maxiter=self.MaxIter),
            callback=callback_graph,
        )
            
        if(self.Losstype =="PI_error"):
            regressor = NeuralNetworkRegressor(
            neural_network=self.qnn,
            loss= PhysicalLossBurger(self.m,self.mu,self.stepX,self.stepT,self.lenx,self.lent,\
                                     self.xmax,self.umax,self.tmax),
        #    loss="squared_error",
            optimizer=L_BFGS_B(maxiter=self.MaxIter),
            callback=callback_graph,
            )
        return regressor

