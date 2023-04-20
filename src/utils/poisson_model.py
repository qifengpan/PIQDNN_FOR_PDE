from model import *
from qiskit_machine_learning.utils.loss_functions.loss_functions import Loss
from qiskit.circuit.library import ZFeatureMap, ZZFeatureMap, RealAmplitudes
from qiskit import QuantumCircuit, Aer
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.opflow import AerPauliExpectation, PauliSumOp
from qiskit_machine_learning.neural_networks import TwoLayerQNN,EstimatorQNN
from qiskit_machine_learning.algorithms.regressors import NeuralNetworkRegressor
from qiskit.algorithms.optimizers import L_BFGS_B,ADAM,GradientDescent
import pandas as pd
import csv
import math
import numpy as np

class PhysicalLossPoisson(Loss):
    def __init__(self,N,a,f):
        self.dx = 1.0/(N-1)
        self.N  = N
        self.a  = a
        self.f  = f
    def __second_order_central(self,xb,x,xa):
        return (xb+xa-2*x)/self.dx**2
    def __first_order_central(self,xb,xa):
        return (xa-xb)/(2*(self.dx))
    def __phy_Inner_loss(self,xb,x,xa,yb,ya,f,a,abx,aax,aby,aay):
        # loss = a_x*u_x + a*u_xx + a_y*u_y + a*u_yy + f
        # abx means a(x-h), aax ,means a(x+h), analogous for y
        a_x  = self.__first_order_central(abx,aax)
        u_x  = self.__first_order_central(xb,xa)
        u_xx = self.__second_order_central(xb,x,xa)
        a_y  = self.__first_order_central(aby,aay)
        u_y  = self.__first_order_central(yb,ya)
        u_yy = self.__second_order_central(yb,x,ya)
        return (a_x*u_x + a*u_xx + a_y*u_y + a*u_yy) + f
    def evaluate(self, predict: np.ndarray, target: np.ndarray) -> np.ndarray:
        # evaluate L2 Loss first |u-u_predict|_2
        self._validate_shapes(predict, target)
        if len(predict.shape) <= 1:
            L2_loss = (predict - target)**2
        else:
            L2_loss = np.linalg.norm(predict - target, ord = 1,axis=tuple(range(1, len(predict.shape))))**2
        # reshape to the mesh
        u = predict.reshape(self.N,self.N)
        Phy_loss_inner = np.zeros([self.N,self.N])
        Phy_loss_Boundary = np.zeros([self.N,self.N])
        # evaluate physical loss for inner and boundary area
        for i in range(self.N):
            for j in range(self.N):
                if(i == self.N-1 or j == self.N -1 or i ==0 or j == 0):
                    Phy_loss_Boundary[i,j] = (u[i,j])**2
                else:
                    Phy_loss_inner[i,j] = self.__phy_Inner_loss(u[i-1,j],u[i,j],u[i+1,j],u[i,j-1],u[i,j+1],\
                                                                            self.f[i,j],self.a[i,j],self.a[i-1,j],self.a[i+1,j],\
                                                                            self.a[i,j-1],self.a[i,j+1])
        return 0.999*L2_loss + 0.001* Phy_loss_Boundary.reshape(self.N*self.N,1) + 0.001*Phy_loss_inner.reshape(self.N*self.N,1)
    def gradient(self, predict: np.ndarray, target: np.ndarray) -> np.ndarray:
        # For a given point, it has value u, with loss u_loss, the gradient is define as D(u_loss)/D(u) if we consider the
        # loss is a function of the value of current posint. 
        # For L1 norm, it is easy to derive that gradient is sign(u_predict-target).
        # For physical loss, specially for inner loss, analytical solution is not easy to derive, then we calculate 
        # the gradient via:
        # (u_loss(u)-u_loss(u+delta*u))/delta*u
        self._validate_shapes(predict, target)
        u = predict.reshape(self.N,self.N)
        Phy_loss_Boundary_gradient = np.zeros([self.N,self.N])
        Phs_loss_inner_gradient = np.zeros([self.N,self.N])
        Phs_loss_inner = np.zeros([self.N,self.N])
        delta = 0.01
        for i in range(self.N):
            for j in range(self.N):
                if(i == self.N-1 or j == self.N -1 or i ==0 or j == 0):
                    Phy_loss_Boundary_gradient[i,j] = u[i,j]
                else:
                    orgi = self.__phy_Inner_loss(u[i-1,j],u[i,j],u[i+1,j],u[i,j-1],u[i,j+1],\
                                                                            self.f[i,j],self.a[i,j],self.a[i-1,j],self.a[i+1,j],\
                                                                            self.a[i,j-1],self.a[i,j+1])
                    disturbed = self.__phy_Inner_loss(u[i-1,j],u[i,j]*(1+delta),u[i+1,j],u[i,j-1],u[i,j+1],\
                                                                            self.f[i,j],self.a[i,j],self.a[i-1,j],self.a[i+1,j],\
                                                                            self.a[i,j-1],self.a[i,j+1])
                    Phs_loss_inner_gradient[i,j] = (disturbed - orgi) / (delta*u[i,j])
                    Phs_loss_inner[i,j] = orgi
        return 1 * 0.999*(predict - target) + 2 * 0.001*Phy_loss_Boundary_gradient.reshape(self.N*self.N,1)\
                 +0.01*Phs_loss_inner_gradient.reshape(self.N*self.N,1)#*Phs_loss_inner.reshape(self.N*self.N,1)


class Poisson_model:
    def __init__(self,inputLosstype,dimension,MaxIter):
        self.Losstype = inputLosstype
        self.dim = dimension
        self.MaxIter = MaxIter
        self.x_input = []
        self.y_output = []
        self.__Preprocessing()
        self.__generate_NN()
    def __Preprocessing(self):
        path = "../Data/Poisson/"
        res_tensor = []#list for np.array
        for i in range(4,14,2):#file 4-92(45 files in total)
            filename = 'output_'+ str(i)+ ".csv"
            with open(path+filename) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for row in csv_reader:
                    res = ", ".join(row)
                res = res.split(',')
                res = res[0:len(res)-1] # remove thelast element.(a space)
                res_tensor.append(np.array(res))
        
        a_tensor = []
        f_tensor = []
        x_tensor = []
        y_tensor = []
        dimension_tensor = []
        for i in range(4,14,2):
            N = i + 1# number of points
            a = np.ones(N*N)
            f = np.zeros(N*N)

            a_tensor.append(a)
            x = np.linspace(0,1,N)
            y = np.linspace(0,1,N)
            for i in range(N*N):
                x_i = x[int(i / N)]
                y_i = y[i % N]
                f[i] = 2*pow(math.pi,2)*math.sin(math.pi*x_i)*math.sin(math.pi*y_i)
            x_mesh,y_mesh= np.meshgrid(x,y,indexing = 'ij')
            x_tensor.append(x_mesh.reshape(N*N))
            y_tensor.append(y_mesh.reshape(N*N))
            dimension_tensor.append(np.ones(N*N)*N*N)
            f_tensor.append(f)
        x_position=[]
        y_position=[]
        a = []
        dimension = []
        f = []
        res_inone = []
        for i in range(5):
            x_position = x_position + x_tensor[i].tolist()
            y_position = y_position + y_tensor[i].tolist()
            a = a + a_tensor[i].tolist()
            dimension = dimension + dimension_tensor[i].tolist()
            f = f + f_tensor[i].tolist()
            res_inone = res_inone + res_tensor[i].tolist()
        res_float=[]
        for i in res_inone:
            res_float.append(float(i))
        d = {"x_position":x_position,"y_position":y_position,"f":f,"a3":a,"dim":dimension}
        input_data = pd.DataFrame(data = d)
        y_out = {"res":res_float}
        output_data = pd.DataFrame(data = y_out)
        x = input_data.to_numpy()
        y = output_data.to_numpy()
        if self.dim == 11:
            self.x_input = x[155:276,:]
            self.y_output = y[155:276,:]
        if self.dim == 9:
            self.x_input = x[74:155,:]
            self.y_output = y[74:155,:]
        if self.dim == 7:
            self.x_input = x[25:74,:]
            self.y_output = y[25:74,:]
        if self.dim == 5:
            self.x_input = x[0:25,:]
            self.y_output = y[0:25,:]

    def __generate_NN(self):
        #quantum_instance = QuantumInstance(Aer.get_backend("statevector_simulator"), shots=1024)


        feature_map = ZFeatureMap(5,reps=1)
        # feature_map = RealAmplitudes(5,reps = 3)
        ansatz = QuantumCircuit(5, name="Ansatz")
        
        # First Convolutional Layer
        ansatz.compose(conv_layer(5, "с1"), list(range(5)), inplace=True)
        
        # First Pooling Layer
        ansatz.compose(pool_layer([0,1],[2,3,4], "p1"), list(range(5)), inplace=True)
        
        # Second Convolutional Layer
        ansatz.compose(conv_layer(5, "c2"), list(range(5)), inplace=True)
        
        # Second Pooling Layer
        ansatz.compose(pool_layer([0,1],[2,3,4], "p2"), list(range(5)), inplace=True)
        
        # # Third Convolutional Layer
        # ansatz.compose(conv_layer(5, "c22"), list(range(5)), inplace=True)
        
        # # Third Pooling Layer
        # ansatz.compose(pool_layer([0,1],[2,3,4], "p22"), list(range(5)), inplace=True)
        
        # Third Convolutional Layer
        ansatz.compose(conv_layer(3, "c3"), list(range(2,5)), inplace=True)
        
        # Third Pooling Layer
        ansatz.compose(pool_layer([0,1], [2], "p3"), list(range(2,5)), inplace=True)
        
        # Fourth Convolutional Layer
        ansatz.compose(conv_layer(2, "c4"), list(range(3,5)), inplace=True)
        
        # Fourth Pooling Layer
        ansatz.compose(pool_layer([0],[1], "p4"), list(range(3,5)), inplace=True)
        
        # Combining the feature map and ansatz
        circuit = QuantumCircuit(5)
        circuit.compose(feature_map, range(5), inplace=True)
        circuit.compose(ansatz, range(5), inplace=True)
        observable = PauliSumOp.from_list([("Z" + "I" * 4, 1)])
        # specify the observable
        # observable = PauliSumOp.from_list([("Z" * 5, 1)])
        
        # qnn_ps = TwoLayerQNN(
        #     num_qubits=5,
        #     feature_map=feature_map,
        #     ansatz=ansatz,
        #     observable=observable,
        #     exp_val=AerPauliExpectation(),
        #     quantum_instance=quantum_instance,
        # )
        self.qnn_ps = EstimatorQNN(
            circuit=circuit.decompose(),
            observables=observable,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
        )

    def generate_Regressor(self):
        objective_func_vals = []
        def callback_graph(weights, obj_func_eval):
            clear_output(wait=True)
            objective_func_vals.append(obj_func_eval)
            plt.title("Objective function value against iteration")
            plt.xlabel("Iteration")
            plt.ylabel("Objective function value")
            plt.plot(range(len(objective_func_vals)), objective_func_vals)
            plt.show()
        if(self.Losstype=="sq_error"):
            regressor = NeuralNetworkRegressor(
            neural_network=self.qnn_ps,
    #     loss= Physical_loss(5,a.reshape(5,5),f.reshape(5,5)),
            loss="squared_error",
            optimizer=L_BFGS_B(maxiter=self.MaxIter),
            callback=callback_graph,
        )
        if(self.Losstype =="PI_error"):
            a = self.x_input[:,3]
            f = self.x_input[:,2]
            regressor = NeuralNetworkRegressor(
            neural_network=self.qnn_ps,
            loss= Physical_loss(self.dim,a.reshape(self.dim,self.dim),\
                    f.reshape(self.dim,self.dim)),
        #    loss="squared_error",
            optimizer=L_BFGS_B(maxiter=self.MaxIter),
            callback=callback_graph,
            )
        return regressor



