# Mckean-VlasovOptimalControl
Optimal feedback for Mckean-Vlasov control problems via artificial neural newtworks. 

CoefficientsAndCost.py   - modify discretization parameters, neural network parameters, parameters of the controlled state equation and cost functional
GalerkinSetting.py       - setting for Galerkin discretization of the control problem\\
RicSave.py               - creates and saves the optimal feedback control for a linear quadratic control problem
SolveAdjoint.py          - solver for adjoint equation and approximation of the gradient of the cost functional
SolveSpde.py             - solver for controlled spde
generateRef.py           - generates a reference profile/target profile for the controlled equation
optimization.py          - gradient descent algorithm
solveRicc.py             - solver for Riccatti equation for linear quadratic control problem
