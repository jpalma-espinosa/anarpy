# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 14:56:13 2015
Wilson Cowan Model with plasticity and network topology
"""
import numpy as np
from numba import njit
import toml
from numba.core.errors import NumbaPerformanceWarning
import warnings
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

def create(params_file):
    """
    Carga los parámetros desde un archivo TOML y los hace accesibles de forma global.
        * param params_file: Ruta del archivo TOML con parámetros.
    """
    with open(params_file, "r", encoding="utf-8") as f:
        modelParameters = toml.load(f)

    for key,val in zip(modelParameters.keys(),modelParameters.values()):
        exec(f"{key} = {val}",globals())
    
    global CM
    CM = np.random.binomial(1, 0.1, (N, N)).astype(np.float64)

@njit
def S(x):
    return (1/(1+np.exp(-(x-mu)/sigma)))

@njit
def wilsonCowan(t,X):
    E,I,a_ei = X
    noise=np.random.normal(0,sqdtD,size=N)
    return np.vstack(((-E + (1-rE*E)*S(a_ee*E - a_ei*I + G*np.dot(CM,E) + P + noise))/tauE,
                     (-I + (1-rI*I)*S(a_ie*E - a_ii*I ))/tauI,
                     (I*(E-rhoE))/tau_ip))

@njit
def wilsonCowanDet(t,X):
    E,I,a_ei = X
    return np.vstack(((-E + (1-rE*E)*S(a_ee*E - a_ei*I + G*np.dot(CM,E) + P))/tauE,
                     (-I + (1-rI*I)*S(a_ie*E - a_ii*I ))/tauI,
                     (I*(E-rhoE))/tau_ip))

def SimAdapt(Init=None):
    """
    Runs two deterministic simulations of timeTrans. 
    
    First with tau_ip=0.05
    Second with tau_ip=tau_ip/2 and noise if D>0
    """
    global timeTrans, tau_ip
    if Init is None:
        Var=np.array([E0,I0,a_ei_0])[:,None]*np.ones((1,N))
    else:
        Var=Init
    # generate the vector again in case variables have changed
    timeTrans=np.arange(0,tTrans,dtSim)
   
    old_tau_ip=tau_ip
    tau_ip=0.05
    wilsonCowanDet.recompile()
    
    for i,t in enumerate(timeTrans):
        # Varinit[i]=Var
        Var+=dtSim*wilsonCowanDet(t,Var)
      
    tau_ip=old_tau_ip/2
    
    if D==0:
        wilsonCowanDet.recompile()
        for i,t in enumerate(timeTrans):
            Var+=dtSim*wilsonCowanDet(t,Var)
    else:
        sqdtD=D/np.sqrt(dtSim)
        wilsonCowan.recompile()
        for i,t in enumerate(timeTrans):
            Var+=dtSim*wilsonCowan(t,Var)
    tau_ip=old_tau_ip
    return Var

def Sim(Var0=None,verbose=False):
    """
    Run a network simulation with the current parameter values.
    
    If D==0, run deterministic simulation.
    Note that the time unit in this model is seconds.

    Parameters
    ----------
    Var0 : ndarray (3,N), ndarray (3,), or None
        Initial values. Either one value per each node (3,N), 
        one value for all (3,) or None. If None, an equilibrium simulation
        is run with faster plasticity kinetics. The default is None.
    verbose : Boolean, optional
        If True, some intermediate messages are shown.
        The default is False.

    Raises
    ------
    ValueError
        An error raises if the dimensions of CM and the number of nodes
        do not match.

    Returns
    -------
    Y_t : ndarray
        Time trajectory for the three variables of each node.
    time : TYPE
        Values of time.

    """
    global CM,sqdtD,downsamp

    if CM.shape[0]!=CM.shape[1] or CM.shape[0]!=N:
        raise ValueError("check CM dimensions (",CM.shape,") and number of nodes (",N,")")
    
    if CM.dtype is not np.dtype('float64'):
        try:
            CM=CM.astype(np.float64)
        except:
            raise TypeError("CM must be of numeric type, preferred float")
    
    
    if type(Var0)==np.ndarray:
        if len(Var0.shape)==1:
            Var=Var0*np.ones((1,N))
        else:
            Var=Var0
    elif Var0 is None:
        Var=SimAdapt()

    timeSim=np.arange(0,tstop,dtSim)
    time=np.arange(0,tstop,dt)
    downsamp=int(dt/dtSim)
         
    Y_t=np.zeros((len(time),3,N))  #Vector para guardar datos

    if verbose:
        print("Simulating %g s dt=%g, Total %d steps"%(tstop,dtSim,len(timeSim)))

    if verbose and D==0:
        wilsonCowanDet.recompile()
        for i,t in enumerate(timeSim):
            if i%downsamp==0:
                Y_t[i//downsamp]=Var
            if t%10==0:
                print("%g of %g s"%(t,tstop))
            Var += dtSim*wilsonCowanDet(t,Var)

    if verbose and D>0:
        sqdtD=D/np.sqrt(dtSim)
        wilsonCowan.recompile()
        for i,t in enumerate(timeSim):
            if i%downsamp==0:
                Y_t[i//downsamp]=Var
            if t%10==0:
                print("%g of %g ms"%(t,tstop))
            Var += dtSim*wilsonCowan(t,Var)

    if not verbose and D==0:
        wilsonCowanDet.recompile()
        for i,t in enumerate(timeSim):
            if i%downsamp==0:
                Y_t[i//downsamp]=Var
            Var += dtSim*wilsonCowanDet(t,Var)

    if not verbose and D>0:
        sqdtD=D/np.sqrt(dtSim)
        wilsonCowan.recompile()
        for i,t in enumerate(timeSim):
            if i%downsamp==0:
                Y_t[i//downsamp]=Var
            Var += dtSim*wilsonCowan(t,Var)
            
    return Y_t,time