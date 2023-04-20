import sys
sys.path.append('./utils')
from poisson_model import Poisson_model
from model import callback_graph
import matplotlib.pyplot as plt
import os

def main():
    PoissonNN5_5 = Poisson_model("sq_error",5,10)
    Regressor5_5_sq_error = PoissonNN5_5.generate_Regressor()
    objective_func_vals = []
    plt.rcParams["figure.figsize"] = (12, 6)
    print(PoissonNN5_5.y_output)
    Regressor5_5_sq_error.fit(PoissonNN5_5.x_input,PoissonNN5_5.y_output )
    # return to default figsize
    plt.rcParams["figure.figsize"] = (6, 4)

if __name__ == "__main__":
    sys.exit(main())
