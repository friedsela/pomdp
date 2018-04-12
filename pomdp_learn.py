import numpy as np

#s        - number of actions
#n        - number of states
#r        - number of observations
#A[a,i,j] - probability of transition from state i to state j                            
#           given action a
#O[a,j,o] - probability of observing o when entering state j                              
#           given action a
#pi[j]    - probability that the initial state is j
#Act      - a list of actions
#Obs      - a list of observations, the j-th observation corresponds to the 
#           j-th action

class POMDP:
    def __init__(self, Obs, Act, A, O, pi):
        self.T = len(Obs)
        self.Obs = Obs
        self.Act = Act        
        self.A = A
        self.O = O
        self.pi = pi
        self.s = self.A.shape[0]
        self.n = self.A.shape[1]
        self.r = self.O.shape[2]
        self.alpha = np.zeros((self.T+1,self.n))
        self.beta = np.zeros((self.T+1,self.n))
        self.chi = np.zeros((self.T+1,self.n))
        self.xi = np.zeros((self.T+1,self.n,self.n))

    def alpha_beta_calc(self):
        for j in range(self.n):
            self.alpha[0,j] = self.pi[j]
            self.beta[self.T,j]= 1.0            

        for t in range(1, self.T+1):
            for j in range(self.n):                
                self.alpha[t,j] = sum([self.A[self.Act[t-1],i,j]*self.alpha[t-1,i]*self.O[self.Act[t-1],i,self.Obs[t-1]] for i in range(self.n)])
                self.beta[self.T-t,j] = sum([self.A[self.Act[self.T-t],j,i]*self.O[self.Act[self.T-t],j,self.Obs[self.T- t]]*self.beta[self.T- t+1,i] for i in range(self.n)])

    def xi_chi_calc(self):      
        for t in range(self.T+1):
            l = sum([self.alpha[t,k]*self.beta[t,k] for k in range(self.n)])      
            for i in range(self.n):
                self.chi[t,i]=self.alpha[t,i]*self.beta[t,i]/l
                for j in range(self.n):
                    self.xi[t,i,j]=(self.alpha[t-1,i]*self.A[self.Act[t-1],i,j]*self.O[self.Act[t-1],i,self.Obs[t- 1]]*self.beta[t,j])/l

    def pi_calc(self):
        for j in range(self.n):
            self.pi[j] = self.chi[0,j]

    def A_calc(self):        
        for i in range(self.n):
            for j in range(self.n):
                for a in range(1):
                    nom = sum([self.xi[t,i,j] for t in range(1,self.T+1) if self.Act[t-1] == a])
                    denom = sum([self.chi[t-1,i] for t in range(1,self.T+1) if self.Act[t-1] == a])
                    if denom != 0: self.A[a,i,j] = nom/denom

    def O_calc(self):
        for j in range(self.n):
            for k in range(self.r):
                for a in range(self.s):
                    nom = sum([self.chi[t,j] for t in range(1, self.T+1) if (self.Act[t-1] == a) and (self.Obs[t-1] == k)])
                    denom = sum([self.chi[t,j] for t in range(1, self.T+1) if self.Act[t-1] == a])
                    self.O[a,j,k] = nom/denom

def sim(l, A, O, pi):
    obs=[]
    act=[]
    if np.random.rand()>pi[0]: curr = 1
    else: curr = 0
    states = [curr]
    for _ in range(l):
        a = np.random.randint(0,2)

        if np.random.rand()<A[a, curr, 1]: curr = 1
        else: curr = 0
        if np.random.rand()<O[a, curr, 1]: o = 1
        else: o =0
        obs.append(o)
        act.append(a)
        states.append(curr)
    return act,obs,states

def generate(length):
    A = np.random.rand(2,2,2)
    O = np.random.rand(2,2,2)
    for i in range(2):
        for j in range(2):
            A[i,j,1]=1.0-A[i,j,0]
            O[i,j,1]=1.0-O[i,j,0]

    pi = np.random.rand(2)    
    pi[1]=1.0-pi[0]
    act,obs,states = sim(length, A, O, pi)
    return A, O, pi, act,obs,states

#generate a random pomdp and a simulation of a given length
A, O, pi,act,obs,states = generate(1000)

#initialize matrices randomly
A, O, pi,act1,obs1,states1 = generate(1)
pomdp = POMDP(obs, act, A, O, pi)

for _ in range(100):
    pomdp.alpha_beta_calc()        
    pomdp.xi_chi_calc()
    pomdp.A_calc()
    pomdp.O_calc()
    pomdp.pi_calc()

