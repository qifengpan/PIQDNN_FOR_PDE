import model
from qiskit import QuantumCircuit, transpile, Aer, IBMQ
from qiskit.tools.jupyter import *
from qiskit.visualization import *
from ibm_quantum_widgets import *
from qiskit.providers.aer import QasmSimulator
from qiskit import QuantumCircuit, Aer
from qiskit.algorithms.optimizers import COBYLA
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap, ZZFeatureMap, RealAmplitudes
from qiskit.opflow import AerPauliExpectation, PauliSumOp
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit_machine_learning.algorithms.regressors import NeuralNetworkRegressor
from qiskit_machine_learning.neural_networks import TwoLayerQNN
from qiskit.algorithms.optimizers import L_BFGS_B,ADAM,GradientDescent
from qiskit.circuit import Parameter
from qiskit_machine_learning.utils.loss_functions.loss_functions import Loss

import matplotlib.pyplot as plt
import math


#@qifeng: add physical loss part here.
class PhysicalLossBurger(Loss):



class BurgerModel(Losstype,dimension,MaxIter):
    def __init__(self,Losstype,dimension):
        self.Losstype = Losstype
        self.dim = dimension
        self.MaxIter = MaxIter
    def __preprocessing(self):
        number_of_samples = 2000
        noise_level = 0.1
        x= np.load('dataset/burgers_x'+'_'+str(number_of_samples)+'.npy')
        t= np.load('dataset/burgers_t'+'_'+str(number_of_samples)+'.npy')
        u= np.array(np.load('dataset/burgers_u'+'_'+str(number_of_samples)+'.npy'),dtype=np.float32).reshape(len(x),len(t))

        noise_l = noise_level*np.std(u)*np.random.randn(u.shape[0],u.shape[1])
        u = u + noise_l

        x_grid,  t_grid = np.meshgrid(x,  t, indexing="ij")


        idx = resample(np.arange(u1_noisy.ravel().shape[0]),replace=False,random_state=44,n_samples=number_of_samples)

        u1_noisy_shuffled = (u1_noisy.ravel())[idx]
        xs = ((x_grid.ravel())[idx]).reshape(-1,1)
        ts = ((t_grid).ravel()[idx]).reshape(-1,1)

        scale_from_outputs = []
        u_d = []

        # scaling and subsampling
        u_sampled = u1_noisy_shuffled.reshape(-1,1)
        u_max = u_sampled.max()
        scale_from_outputs.append(u_max)
        u_d.append(u_sampled/u_max)   


        # Data Normalization
        X_data = np.concatenate([ts/t.max(),xs/x.max()],axis=1)
        y_data = np.concatenate(u_d,axis=0)

        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

        self.x_input = X_train
        self.y_input = y_train

        self.x_test = X_test
        self.y_test = y_test


    def __generate_NN(self):
        quantum_instance = QuantumInstance(Aer.get_backend("statevector_simulator"), shots=1024)


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

        self.qnn = TwoLayerQNN(
            num_qubits=2,
            feature_map=feature_map,
            ansatz=ansatz,
            observable=observable,
            exp_val=AerPauliExpectation(),
            quantum_instance=quantum_instance,
        )

    def generate_Regressor(self):
        self.__preprocessing()
        self.__generate_NN()
        if(self.Losstype=="sq_error"):
            regressor = NeuralNetworkRegressor(
            neural_network=qnn_ps,
            loss="squared_error",
            optimizer=L_BFGS_B(maxiter=10),
            callback=callback_graph,
        )
            
        #TODO: @qifeng
        ##############    
        if(self.Losstype =="PI_error"):
            a = self.x_input[:,3]
            f = self.x_input[:,2]
            regressor = NeuralNetworkRegressor(
            neural_network=qnn_ps,
            loss= Physical_loss(self.dim,a.reshape(self.dim,self.dim),\
                    f.reshape(self.dim,self.dim)),
        #    loss="squared_error",
            optimizer=L_BFGS_B(maxiter=self.MaxIter),
            callback=callback_graph,
            )
        return regressor

