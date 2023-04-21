import sys
sys.path.append('./utils')
from poisson_model import Poisson_model
from burger_model import BurgerModel
from model import callback_graph
import matplotlib.pyplot as plt
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--problem-type', dest='problem_type', help='Please clarify which problem you want to solve: "Poisson" or "Burger"')
args = parser.parse_args()
problem_type = args.problem_type


def main():
    if (problem_type =="Poisson"):
        PoissonNN5_5 = Poisson_model("sq_error",5,10)
        Regressor5_5_sq_error = PoissonNN5_5.generate_Regressor()
        objective_func_vals = []
        plt.rcParams["figure.figsize"] = (12, 6)
        print(PoissonNN5_5.y_output)
        Regressor5_5_sq_error.fit(PoissonNN5_5.x_input,PoissonNN5_5.y_output )
        # return to default figsize
        #PoissonNN5_5 = Poisson_model("PI_error",5,10)
        #Regressor5_5_PI_error = PoissonNN5_5.generate_Regressor()
        #objective_func_vals = []
        #plt.rcParams["figure.figsize"] = (12, 6)
        #print(PoissonNN5_5.y_output)
        #Regressor5_5_PI_error.fit(PoissonNN5_5.x_input,PoissonNN5_5.y_output )
        # return to default figsize
        plt.rcParams["figure.figsize"] = (6, 4)
        plt.rcParams["figure.figsize"] = (6, 4)
    if (problem_type == 'Burger'):
        BurgerSolver = BurgerModel("sq_error",5,10,0.5,0.01)
        Burger_sq_error = BurgerSolver.generate_Regressor()
        objective_func_vals = []
        plt.rcParams["figure.figsize"] = (12, 6)
        Burger_sq_error.fit(BurgerSolver.x_input,BurgerSolver.y_input )
        ## return to default figsize
        plt.rcParams["figure.figsize"] = (6, 4)
        #BurgerSolver = BurgerModel("PI_error",5,10,0.5,0.01)
        #Burger_sq_error = BurgerSolver.generate_Regressor()
        #objective_func_vals = []
        #plt.rcParams["figure.figsize"] = (12, 6)
        #Burger_sq_error.fit(BurgerSolver.x_input,BurgerSolver.y_input )
        # return to default figsize
        #plt.rcParams["figure.figsize"] = (6, 4)




if __name__ == "__main__":
    sys.exit(main())
