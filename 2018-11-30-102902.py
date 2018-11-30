#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


r0 = 0.005
T1, T2, n = 0., .5, 4
t = np.linspace(T1, T2, num = n+1)
Rn = r0 + np.zeros(n+1)

for i in range(n):
    Rn[i+1] = Rn[i] + 2.11*(0.02-Rn[i])*(t[i+1]-t[i])+0.033*np.sqrt(t[i+1]-t[i])*np.random.normal()
    
print(Rn)


# In[5]:


def SDE_path(r0, T1, T2, n):
    t = np.linspace(T1, T2, num = n+1)
    Rn = r0 + np.zeros(n+1)
    for i in range(n):
        Rn[i+1] = Rn[i] + 2.11*(0.02-Rn[i])*(t[i+1]-t[i])+0.033*np.sqrt(t[i+1]-t[i])*np.random.normal()
        
    return t, Rn


# In[7]:


p = []
for i in range(10):
    [t, Y] = SDE_path(0.005, 0., 10., 100);
    p.append(Y[-1])
    plt.plot(t,Y);


# In[9]:


print(np.mean(p))


# 2.Find explicit form of $\mathbb E [r_t]$ and $\lim_{t\to \infty} \mathbb E [r_t]$.
# 
# 
# Derivation of the explicit form
# 
# $$ dr_t=a(b-r_t)dt+\sigma dW_t $$
# 
# We know from the Ito's formula,can get:
# 
# 
# $$r_t=e^{-at}[r_0+\int_0^tabe^{au}du+\sigma \int_0^te^{au}dW_u] = \mu_t +\sigma \int_0^t e^{a(u-t)}dW_u   $$
# 
# The integral of the vasicek model is :
# 
# $$ r_t= r_0+\int_0^t a(b-r_u)du+\sigma dW_u             $$
# 
# Compares the previous two equations :
# 
# $$   \mu_t=\mathbb E[r_t]=r_0+\int_0^ta(b-\mathbb E[r_u])du      $$
# 
# Solves  linear ODE, the expection of short rate is :
# 
# $$ \mathbb E[r_t]=e^{-at}[r_0+b(e^{at}-1)]      $$
# 
# If the stochastic term is included, the explicit solution can be found:
# 
# $$ r_t=e^{-at}[r_0+b(e^{at}-1)] + \sigma \int_0^t dW_t    $$
# 
# So,
# $$ lim_{t \rightarrow \infty} \mathbb E[r_t] = b $$
# 

# 3.Zero bond price has the formula $$P(0, T) = \mathbb E[\exp\{-\int_0^T r(s) ds\}].$$ Find the exact value of $P(0,1)$.
# 
# 
# 
# The exact value of P(0,1) is as follows:
# 
# $$ B(0,1)=\frac{1}{a}(1-e^{-a})      $$
# 
# $$ A(0,1)=exp[(b-\frac{\sigma^2}{2a^2})(B(0,1)-1)-\frac{\sigma^2}{4a}B^2(0,1)]      $$
# 
# $$ P(0,1) =A(0,1) e^{-B(0,1)r_0}   $$

# In[8]:


import numpy as np

r0=0.005
a=2.11
b=0.02
sigma=0.033

B=1.0/a*(1-1.0/np.exp(a))
A=np.exp((b-sigma**2/2.0/a**2)*(B-1)-sigma**2/4.0/a*B**2)

P=A*np.exp(-1*B*r0)
print("The bond price P(0,1)=",P)


# 
# 4.Run Euler, Milstein, exact simulation on P(0,1)P(0,1) with different stepsizes, and find the covergence rate for each using linear regression. (Hint: one shall approximate integral with finite sum)

# In[9]:


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# In[21]:


class SDE:
    
    
    #Initialize
    
    def __init__(self, Mu, Sigma, InitState):
        self.Mu = Mu
        self.Sigma = Sigma
        self.InitState = InitState
        
        self.MuP = lambda r, t: 0 
    
    #Euler method
    
    def Euler(self, T, N):
        r0 = self.InitState
        Mu = self.Mu
        Sigma = self.Sigma       
        t = np.linspace(0, T, N+1)
        DeltaT = T/N
        
        Wh = np.zeros(N+1) 
        Rh = r0 + np.zeros(N+1) 
        
        for i in range(N):         
            DeltaW = np.sqrt(t[i+1] - t[i]) * np.random.normal()
            Wh[i+1] = Wh[i] + DeltaW
            Rh[i+1] = Rh[i] + Mu(Rh[i], t[i]) * DeltaT + Sigma(Rh[i], t[i])* DeltaW
            
        return t, Rh, Wh
    
    #Milstein method
    
    
    def Milstein(self, T, N):
        r0 = self.InitState
        Mu = self.Mu
        MuP = self.MuP
        
        Sigma = self.Sigma
        t = np.linspace(0, T, N+1)
        DeltaT = T/N
        
        Wh = np.zeros(N+1) 
        Rh = r0 + np.zeros(N+1) 
        
        for i in range(N):
            DeltaW = np.sqrt(t[i+1] - t[i]) * np.random.normal()
            Wh[i+1] = Wh[i] + DeltaW
            Rh[i+1] = Rh[i+1] + 0.5 * Mu(Rh[i], t[i]) * MuP(Rh[i], t[i]) * (DeltaW**2 - DeltaT)
                
        return t, Rh, Wh
    
    #Exact simulation
    
    def Exact(self,a,b,T,sigma,N):
        
        r0 = self.InitState      
        t = np.linspace(0, T, N+1)
        Rh = r0 + np.zeros(N+1)
        
        for i in range(N):
            Rh[i+1]=r0*np.exp(-1*a*t[i+1])+b-b*np.exp(-1*a*t[i+1])+sigma*np.random.normal()       
        return Rh

    
    #Explict solution
    
    def Explicit(self,a,b, T, Wt):
        r0 = self.InitState
        Rh=r0*np.exp(-1*a*T)+b-b*np.exp(-1*a*T)+Wt
            
        return Rh
    
    def Bond_Exact(self,a,b,sigma):
        r0 = self.InitState
        B=1.0/a*(1-1.0/np.exp(a))
        A=np.exp((b-sigma**2/2.0/a**2)*(B-1)-sigma**2/4.0/a*B**2)
        P=A*np.exp(-1*B*r0)
        
        return P


# In[29]:


if __name__ == '__main__':
    
    a=2.11
    b=0.02
    mu = lambda r, t: a*(b-r)
    sigma = lambda r, t: 0.033
    sig=0.033
    r0 = 0.005
   
    iSDE = SDE(mu, sigma, r0)

    ASteps = np.arange(8)
    NSteps = 2 

    AE_Euler = np.zeros(ASteps.size)
    AE_Milstein = np.zeros(ASteps.size)
    AE_Exact = np.zeros(ASteps.size)
    T = 1.
    NumSimu = 200
    
    #Exact price of zero-coupon bond
    
    P=iSDE.Bond_Exact(a,b,sig)
    for n in ASteps:
        NumMesh = np.power(2, n + NSteps)
        ess_Euler = 0
        ess_Milstein = 0
        ess_Exact = 0
        
        deltaT=T/NumMesh
        
        for i in range(NumSimu):
            
           
            #Euler simulation
                        
            [t, Rh, Wh] = iSDE.Euler(T, NumMesh)
            RhT = Rh[-1]
            P_hat=np.exp(-1.0*deltaT*np.sum(Rh))
            ess_Euler = ess_Euler + np.abs(P_hat - P)
            
            #Milstein simulation
                        
            [t, Rh, Wh] = iSDE.Milstein(T, NumMesh)
            RhT = Rh[-1]
            P_hat=np.exp(-1.0*deltaT*np.sum(Rh))         
            ess_Milstein = ess_Milstein + np.abs(P_hat - P)
            
            #Exact simulation
                        
            
            Rh = iSDE.Exact(a,b,T,sig,NumMesh)
            RhT = Rh[-1]
            P_hat=np.exp(-1.0*deltaT*np.sum(Rh))
            ess_Exact = ess_Exact + np.abs(P_hat - P)

        AE_Euler[n] = ess_Euler/NumSimu
        AE_Milstein[n] = ess_Milstein/NumSimu
        AE_Exact[n] = ess_Exact/NumSimu        

    
    r_coordinate = ASteps+NSteps
    y_coordinate_Euler = np.log(AE_Euler)
    y_coordinate_Milstein = np.log(AE_Milstein)
    y_coordinate_Exact = np.log(AE_Exact)    
    plt.plot(r_coordinate, y_coordinate_Euler,label='Euler')
    plt.plot(r_coordinate, y_coordinate_Milstein,label='Milstein')    
    plt.plot(r_coordinate, y_coordinate_Exact,label='Exact simulation')
    plt.legend()   
    
    lg1 = stats.linregress(r_coordinate,y_coordinate_Euler)
    lg2 = stats.linregress(r_coordinate,y_coordinate_Milstein)    
    lg3 = stats.linregress(r_coordinate,y_coordinate_Exact)
    rate1 = -lg1[0]
    rate2=-lg2[0]
    rate3=-lg3[0]
    print("Euler method  covergence rate is " , rate1)
    print("Milstein method  covergence rate is ", rate2)        
    print("Exact simulation  covergence rate is ", rate3)   


# In[ ]:




