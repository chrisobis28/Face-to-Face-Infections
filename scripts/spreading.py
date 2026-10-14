"""
First implementation of basic SIR / SIS spreading (continuous and discrete) on the networks. 
Dynamics of epidemiological spread. 
- SIR = susceptible, infected, recovered. Immunity once recovered. 
- SIS = susceptible, infected, susceptible. No immunity. 
Continuous uses ODEs to solve for continous spreading, discrete assigns labels at each timestep. 
beta = rate of infection.  
gamma = rate of recovery. 
"""

import networkx as nx
import numpy as np
import pandas as pd 
from scipy.integrate import solve_ivp

S, I, R = 0, 1, 2

def discrete(graph, model = "SIR", beta = 0.05, gamma = 0.02, n_seeds = 5, timesteps=500, seed = None):
  """Returns a dataframe with number of S, I, R nodes at each time step"""
  rng = np.random.default_rng(seed)
  nodes = list(graph.nodes)
  idx = {n: k for k,n in enumerate(nodes)}
  #adjacency matrix, two implementations, with and without weights.  should actually be weighted if we use aggregated graph 
  # A = nx.to_scipy_sparse_array(graph, nodelist=nodes, weight=None, format="csr") 
  A = nx.to_scipy_sparse_array(graph, nodelist=nodes, weight="weight", format="csr")
  A = A / A.max()

  state = np.zeros(len(nodes))
  state[rng.choice(len(nodes), n_seeds, replace=False)] = I #seed nodes that are infected 
  if model =="SIR": 
    recovered_to = R
  elif model =="SIS":
    recovered_to = S
  else: 
    raise ValueError(f"unknown model type: {model}")

  history = []
  for i in range(timesteps):
    history.append([(state ==S).sum(), (state == I).sum(), (state==R).sum()])  #gather SIR distribution at current timestep 
    if (state==I).sum() == 0:  #infection has died down  
      history.extend([history[-1]] * (timesteps - len(history)))
      break
    infected_nb = A @ (state==I) #infected neighbors per node
    prob_inf = 1 - (1-beta) ** infected_nb #chance of being infected 
    new_i = (state == S) & (rng.random(len(nodes)) < prob_inf)
    new_r = (state == I) & (rng.random(len(nodes)) < gamma)
    state[new_i] = I
    state[new_r] = recovered_to
  return pd.DataFrame(history, columns=["S", "I", "R"]).rename_axis("t")

def simulate_temporal(contacts, nodes, model="SIR", beta=0.1, gamma=0.001,
                      n_seeds=5, time_col="t", window_s=20, seed=None):
    """discrete simulation on the temporal network, not aggregated 
    every time step is a contactwindow of window_s seconds. Only pairs (i,j) that had contact in this window are able to infect. 
    beta and gamma are rates of infection/recovery within each window!! (20s, so small window. )
    """
    rng = np.random.default_rng(seed)
    ids = list(nodes.index)
    idx = {n: k for k, n in enumerate(ids)}

    if model =="SIR": 
      recovered_to = R
    elif model =="SIS":
      recovered_to = S
    else: 
      raise ValueError(f"unknown model type: {model}")
 
    state = np.zeros(len(ids))
    state[rng.choice(len(ids), n_seeds, replace=False)] = I
 
    c = contacts.assign(_i=contacts["i"].map(idx), _j=contacts["j"].map(idx))
    c["_w"] = (c[time_col] - c[time_col].min()) // window_s
    by_window = {w: g[["_i", "_j"]].to_numpy() for w, g in c.groupby("_w")}
 
    history = []
    for w in range(int(c["_w"].max()) + 1):
        history.append([(state == S).sum(), (state == I).sum(), (state == R).sum()])
        new_i = np.zeros(len(ids), dtype=bool)
 
        pairs = by_window.get(w)
        if pairs is not None:
            a, b = pairs[:, 0], pairs[:, 1]
            # I -> S in both directions
            for src, dst in ((a, b), (b, a)):
                m = (state[src] == I) & (state[dst] == S)
                hit = m & (rng.random(len(src)) < beta)
                new_i[dst[hit]] = True
 
        new_r = (state == I) & (rng.random(len(ids)) < gamma)
        state[new_i] = I
        state[new_r] = recovered_to
 
    t = np.arange(len(history)) * window_s
    return pd.DataFrame(history, columns=["S", "I", "R"], index=pd.Index(t, name="t_s"))
  
def simulate_ode(model="SIR", beta=0.05, gamma=0.02, k_mean=10.0,
                 i0=0.01, t_max=500, n_points=500):
    """ODEs, S + I + R = 1
 
    SIR:  dS/dt = -beta*<k>*S*I
          dI/dt =  beta*<k>*S*I - gamma*I
          dR/dt =  gamma*I
    SIS:  dS/dt = -beta*<k>*S*I + gamma*I
          dI/dt =  beta*<k>*S*I - gamma*I
 
    dt = 1, beta chance of infection per contact 
    R0 = beta*<k>/gamma.
    """
    b = beta * k_mean
 
    def sir(t, y):
        s, i, r = y
        return [-b * s * i, b * s * i - gamma * i, gamma * i]
 
    def sis(t, y):
        s, i, r = y
        return [-b * s * i + gamma * i, b * s * i - gamma * i, 0.0]
 
    t_eval = np.linspace(0, t_max, n_points)
    sol = solve_ivp(sir if model == "SIR" else sis, (0, t_max),
                    [1 - i0, i0, 0.0], t_eval=t_eval)
    return pd.DataFrame(sol.y.T, index=pd.Index(sol.t, name="t"), columns=["S", "I", "R"])

