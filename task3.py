import random 
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings
from sympy import *
import pandas as pd
from scipy.optimize import minimize, rosen, rosen_der
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import cg

ab = np.random.rand(1,2)
w = ab[0][0]
z = ab[0][1]
x_cor = []
y_cor = []
s = np.random.normal(0,1,100)


def noise(s):
    return np.random.choice(s)

#funciton create numbers with noise 
def func(x):
    n = noise(s)
    cal = w*x + z +n

    return cal

#where i create 100 number from 0-1
def random():
    for k in range(101):

        x = k/100
        y = func(x)
        x_cor.append(x)
        y_cor.append(y)    
    return
# def function(x , parameter):
#     a = parameter[0]
#     b = parameter[1]
#     c = parameter[2]
#     return a*x**2 + b*x + c

def gradient_descent(x,y):
    
    m_curr = b_curr = 0  #initial guess
    limit_iteration = 1000
    iterations = 0
    n = len(x)
    learning_rate = 0.08
    last_cost = 0
    eva_fun = 0

    
    for i in range(limit_iteration):
        y_predicted = m_curr * x + b_curr
        cost = (1/n) * sum([val**2 for val in (y-y_predicted)])
        eva_fun += 1
        md = -(2/n)*sum(x*(y-y_predicted))
        bd = -(2/n)*sum(y-y_predicted)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        
        # print ("m {}, b {}, cost {} iteration {}".format(m_curr,b_curr,cost, i))
        if(abs(cost-last_cost) < 0.001):
            break
        last_cost = cost
        iterations = i 
        
    # print ("Ultimate : m {}, b {}, cost {} iteration {}".format(m_curr,b_curr,last_cost,iterations))     
    print ("gradient_descent : iteration={} ,cost={} ,evaluation of function={} ".format(iterations,last_cost,eva_fun))   
    num = np.linspace(0,1,50)
    plt.plot(num,m_curr*num+b_curr,label='gradient_descent')

def gradient_descent2(x,y):
    m_curr = b_curr = 0 #initial guess
    limit_iteration = 1000
    iterations = 0
    n = len(x)
    learning_rate = 0.08
    last_cost = 0
    eva_fun = 0

    for i in range(limit_iteration):
        y_predicted = m_curr * x + b_curr
        cost = (1/n) * sum([val**2 for val in (y-y_predicted)])
        eva_fun += 1
        md = -(2/n)*sum(x*(y-y_predicted))
        
        bd = -(2/n)*sum(y-y_predicted)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        # print ("m {}, b {}, cost {} iteration {}".format(m_curr,b_curr,cost, i))
        if(abs(cost-last_cost) < 0.001):
            break
        last_cost = cost
        iterations = i 
        
    # print ("Ultimate : m {}, b {}, cost {} iteration {}".format(m_curr,b_curr,last_cost, iterations))  
    print ("gradient_descent : iteration={} ,cost={} ,evaluation of function={} ".format(iterations,last_cost,eva_fun))    
    num = np.linspace(0,1,50)
    plt.plot(num,m_curr*num+b_curr,label='gradient_descent_2')

def conjugate_gradient(x,y):
   a = symbols('a')
   b = symbols('b')
   for_hessian = [a,b]
   limit_iteration = 1000
   iteration = 0
   aa = bb = 0
   current_node = [aa,bb]
   current_node = np.array(current_node)
   n = len(x)
   eva_func = 0
   
   x = np.array(x)
   y = np.array(y)   
   cost = (1/n)*sum([(val)**2 for val in (y -((a * x)+b))])
   diffa = diff(cost,a)
   eva_func += 1
   diffb = diff(cost,b)
   eva_func += 1
   hessianf = hessian(cost,for_hessian)
   eva_func += 1
#----------------------------------------
# gradient matrix
   f1 = []
   f1.append(diffa.subs([(a,current_node[0]),(b,current_node[1])]))
   f1.append(diffb.subs([(a,current_node[0]),(b,current_node[1])]))
#----------------------------------------
   s1 = np.array(f1) * -1
   s1 = np.array(s1)
   hessianf = np.array(hessianf)
   hessianf = hessianf.astype(float)
   # print(hessianf)
   # implementation from scipy     
   A = csc_matrix(hessianf)
   b = np.array(f1)
   b = b.astype(float)
   xx = cg(A, b)
   print('conjugate gradietn ---------------------------------------------------------')
   print('this is implementation from scipy {}\n ----------------------'.format(xx[0]))

# AX = B 
   def ConjGrad(a, b, x ,tol=0.001):
          r = b - A.dot(x)
          p = r.copy()
          for i in range(5):
               Ap = A.dot(p)
               alpha = np.dot(p,r)/np.dot(p,Ap)
               x = x + alpha*p
               r = b - A.dot(x)
 
               if np.sqrt(np.sum((r**2))) < tol:
                   iteration = i
                   break
               else:
                   beta = -np.dot(r,Ap)/np.dot(p,Ap)
                   p = r + beta*p
          return x , iteration
      
   xx , iteration = ConjGrad(A,b,current_node)
   # print(xx)
   num = np.linspace(0,1,101)
   num_y = xx[0]*num+xx[1]
   plt.plot(num,num_y,label='conjugate_gradient_descent')
   print('this is code from myself {}\n ----------------------'.format(xx))
   cost = sum(abs(num-y - y))
   print('conjugate_gradient :iteration={} , cost={} ,evaluation of function={} '.format(iteration+1,cost,eva_func))


        


def conjugate_gradient2(x,y):
   a = symbols('a')
   b = symbols('b')
   for_hessian = [a,b]
    
   aa = bb = 0
   current_node = [aa,bb]
   current_node = np.array(current_node)
   n = len(x)
   eva_func = 0
   
   x = np.array(x)
   y = np.array(y)   
   predict = a/1 + x * b
   cost = (1/n)*sum([(val)**2 for val in (y_cor-predict)])
   diffa = diff(cost,a)
   eva_func += 1
   diffb = diff(cost,b)
   eva_func += 1
   hessianf = hessian(cost,for_hessian)
   eva_func += 1
   f1 = []
   f1.append(diffa.subs([(a,current_node[0]),(b,current_node[1])]))
   f1.append(diffb.subs([(a,current_node[0]),(b,current_node[1])]))
   s1 = np.array(f1) * -1
   s1 = np.array(s1)
   hessianf = np.array(hessianf)
   hessianf = hessianf.astype(float)
   # print(hessianf)
   
   # implementation from scipy     
   A = csc_matrix(hessianf)
   b = np.array(f1)
   b = b.astype(float)
   xx, exit_code = cg(A, b)
   print('conjugate gradietn ---------------------------------------------------------')
   print('this is implementation from scipy {}\n ----------------------'.format(xx[0]))
   # AX = B 
   def ConjGrad(a, b, x ,tol=0.001):
          r = b - A.dot(x)
          p = r.copy()
          for i in range(5):
               Ap = A.dot(p)
               alpha = np.dot(p,r)/np.dot(p,Ap)
               x = x + alpha*p
               r = b - A.dot(x)
 
               if np.sqrt(np.sum((r**2))) < tol:
                   iteration = i
                   break
               else:
                   beta = -np.dot(r,Ap)/np.dot(p,Ap)
                   p = r + beta*p
          return x , iteration
      
   xx , iteration = ConjGrad(A,b,current_node)
   print(xx)
   num = np.linspace(0,1,101)
   num_y = xx[0]*num+xx[1]
   plt.plot(num,num_y,label='conjugate_gradient_descent')
   print('this is code from myself {}\n ----------------------'.format(xx))
   cost = sum(abs(num-y - y))
   print('conjugate_gradient :iteration={} , cost={} ,evaluation of function={} '.format(iteration+1,cost,eva_func))

  

    
def newton(x,y):
    a = symbols('a')
    b = symbols('b')
    learning_rate = 0.2      
    last_cost = 0
    tempa = tempb = 0
    current_node= [tempa,tempb]
    n = len(x_cor)
    iteration=100
    eva_func= 0
    predict = a*x + b
    cost = (1/n)*sum([(val)**2 for val in (y_cor-predict)])

    difb = diff(cost,b)
    dif2b = diff(cost,b,2)
    difbf = diff(difb,a)
    dif = diff(cost,a)
    dif2 = diff(cost,a,2)
    eva_func += 4
  

    for i in range(iteration):

        hessian = []
        hessian.append(dif2)
        hessian.append(difbf)
        hessian.append(difbf)
        hessian.append(dif2b)
        hessian = np.array(hessian)
         
        hessian =hessian.reshape(2,2)

        hessian = np.float64(hessian)
        hessian_inv = np.linalg.inv(hessian)

        hessian_inv = hessian_inv*learning_rate

        gradient = []
        gradient.append(dif.subs([(b,current_node[1]),(a,current_node[0])]))
        gradient.append(difb.subs([(b,current_node[1]),(a,current_node[0])]))
        
        #get new cost
        current_cost =cost.subs([(a,current_node[0]),(b,current_node[1])])

        
        current_node , gradient = np.array([current_node,gradient])
        
        #get new node
        current_node = current_node - np.dot(hessian_inv,gradient)
        if(abs(last_cost-current_cost) < 0.001):
            print('newton :iteration={} , cost={} , evaluation of function={}'.format(i,current_cost,eva_func))

            break
        last_cost = current_cost
    plt.scatter(x_cor,y_cor)
    x = np.linspace(0,1,100)
    y = current_node[0] *x + current_node[1]
    plt.plot(x,y,label='newton')

        
def newton2(x,y):
   a = symbols('a')
   b = symbols('b')
   learning_rate = 0.08     
   last_cost = 0
   tempa = tempb = 0
   current_node= [tempa,tempb]
   n = len(x_cor)
   iteration=100
   eva_func = 0
   predict = a/1 + x * b
   cost = (1/n)*sum([(val)**2 for val in (y_cor-predict)])

   difb = diff(cost,b)
   dif2b = diff(cost,b,2)
   difbf = diff(difb,a)
   dif = diff(cost,a)
   dif2 = diff(cost,a,2) 
   eva_func += 4

   for i in range(iteration):

       hessian = []
       hessian.append(dif2)
       hessian.append(difbf)
       hessian.append(difbf)
       hessian.append(dif2b)
       hessian = np.array(hessian)
        
       hessian =hessian.reshape(2,2)
       # print(hessian)
       hessian = np.float64(hessian)
       hessian_inv = np.linalg.inv(hessian)
       # print(hessian_inv)
       hessian_inv = hessian_inv*learning_rate
       # print(hessian_inv)
       gradient = []
       gradient.append(dif.subs([(b,current_node[1]),(a,current_node[0])]))
       gradient.append(difb.subs([(b,current_node[1]),(a,current_node[0])]))
       #get new cost
       current_cost =cost.subs([(a,current_node[0]),(b,current_node[1])])
       # print(gradient)
       
       current_node , gradient = np.array([current_node,gradient])
       #get new node
       current_node = current_node - np.dot(hessian_inv,gradient)
       if(abs(last_cost-current_cost) < 0.001):
           print('newton :iteration={} , cost={} , evaluation of function={}'.format(i,current_cost,eva_func))
           break
       last_cost = current_cost
       
   plt.scatter(x_cor,y_cor)
   x = np.linspace(0,1,10)
   y = current_node[0] / (1+current_node[1]*x)
   plt.plot(x,y,label='newton')   
   
   
def LMA(x,y):
   a = symbols('a')
   b = symbols('b')
   learning_rate = 0.08     
   last_cost = 0
   tempa = tempb = 0
   current_node= [tempa,tempb]
   n = len(x_cor)
   limit_iteration=100
   iterations = 0
   lamb = 5
   eva_func = 0
   predict = a* x + b
   cost = (1/n)*sum([(val)**2 for val in (y_cor-predict)])
   # print(current_node)

   difb = diff(cost,b)
   dif2b = diff(cost,b,2)
   diffb = difb/dif2b
   difbf = diff(difb,a)
   dif = diff(cost,a)
   dif2 = diff(cost,a,2)
   eva_func += 4
 

   for i in range(limit_iteration):

       hessian = []
       hessian.append(dif2)
       hessian.append(difbf)
       hessian.append(difbf)
       hessian.append(dif2b)
       hessian = np.array(hessian)
       identity = np.identity(2)
       identity = identity*lamb
       hessian =hessian.reshape(2,2)

       hessian = np.float64(hessian)
       hessian = hessian + identity
       hessian_inv = np.linalg.inv(hessian)


       gradient = []
       gradient.append(dif.subs([(b,current_node[1]),(a,current_node[0])]))
       gradient.append(difb.subs([(b,current_node[1]),(a,current_node[0])]))
       #get new cost
       current_cost =cost.subs([(a,current_node[0]),(b,current_node[1])])
       eva_func += 1

       
       current_node , gradient = np.array([current_node,gradient])
       
       #get new node
       current_node = current_node - np.dot(hessian_inv,gradient)
       #change lambda
       if(current_cost <= last_cost):
           #lambda down 
           lamb = lamb/10
           # print('change')
          
       else:
           #lambda up because need to rely on hessian more  
           lamb = lamb*10
           
           
       if(abs(last_cost-current_cost) < 0.001):
           iterations = i
           last_cost = current_cost
           # print('stop')
           break
       iterations = i
       last_cost = current_cost
   print('Levenberg- Marquardt algorithm :iteration={} , cost={} , evaluation of function={}'.format(iterations,last_cost,eva_func))        
   plt.scatter(x_cor,y_cor)
   x = np.linspace(0,1,10)
   y = current_node[0] *x + current_node[1]
   plt.plot(x,y,label='Levenberg- Marquardt algorithm')
   
   
def LMA2(x,y):
   a = symbols('a')
   b = symbols('b')
   learning_rate = 0.08     
   last_cost = 0
   tempa = tempb = 0
   current_node= [tempa,tempb]
   n = len(x_cor)
   limit_iteration = 1000
   iteration=0
   lamb = 10
   predict = a/1 + x * b
   eva_func = 0
   cost = (1/n)*sum([(val)**2 for val in (y_cor-predict)])
   # print(current_node)

   difb = diff(cost,b)
   dif2b = diff(cost,b,2)
   diffb = difb/dif2b
   difbf = diff(difb,a)

   dif = diff(cost,a)
   dif2 = diff(cost,a,2)
   eva_func += 4
 

   for i in range(limit_iteration):

       hessian = []
       hessian.append(dif2)
       hessian.append(difbf)
       hessian.append(difbf)
       hessian.append(dif2b)
       hessian = np.array(hessian)
       identity = np.identity(2)
       identity = identity*lamb
       hessian =hessian.reshape(2,2)
       # print(hessian)
       hessian = np.float64(hessian)
       hessian = hessian + identity
       hessian_inv = np.linalg.inv(hessian)
       # print(hessian_inv)
       hessian_inv = hessian_inv*learning_rate
       # print(hessian_inv)
       gradient = []
       gradient.append(dif.subs([(b,current_node[1]),(a,current_node[0])]))
       gradient.append(difb.subs([(b,current_node[1]),(a,current_node[0])]))
       
       current_cost =cost.subs([(a,current_node[0]),(b,current_node[1])])
       eva_func += 1
       # print(gradient)
       
       current_node , gradient = np.array([current_node,gradient])
       
       current_node = current_node - np.dot(hessian_inv,gradient)
       if(current_cost <= last_cost):
           lamb = lamb/10
           # print('change')
          
       else:
           lamb = lamb*10
           
           
       if(abs(last_cost-current_cost) < 0.001):
           iterations = i
           last_cost = current_cost
           # print('stop')
           break
       iterations = i
       last_cost = current_cost
   print('Levenberg- Marquardt algorithm :iteration={} , cost={} , evaluation of function={}'.format(iterations,last_cost,eva_func))        
   plt.scatter(x_cor,y_cor)
   x = np.linspace(0,1,10)
   y = current_node[0] *x + current_node[1]
   plt.plot(x,y,label='Levenberg- Marquardt algorithm')


    

def main():   
    
    


    random()
    x = np.array(x_cor)
    y = np.array(y_cor)
    plt.scatter(x_cor,y_cor)
    
    plt.subplot(121)
    plt.scatter(x_cor,y_cor)
    gradient_descent(x,y)   
    conjugate_gradient(x_cor,y_cor)
    newton(x,y) 
    LMA(x,y) 
    plt.legend()
    
    
    plt.subplot(122)
    gradient_descent2(x,y)
    conjugate_gradient2(x_cor,y_cor)
    newton2(x,y)
    LMA2(x, y)
    plt.legend()
        

if __name__ == "__main__":
    main()





                 
