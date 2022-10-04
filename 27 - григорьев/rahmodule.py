import numpy as np
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm
import random
import copy
import os

#------------------------------------------Runge–Kutta ------------------------------------------------------------------
def phi(t, y, h, f, **kwargs):
    #h = T/T_steps
    k1 = f(t, y, **kwargs)
    k2 = f(t + h/2, y + h/2*k1, **kwargs)
    k3 = f(t + h/2, y + h/2*k2, **kwargs)
    k4 = f(t + h, y + h*k3, **kwargs)
    
    return y + h/6*(k1 + 2*k2 + 2*k3 + k4)

#if nan stops
def solve(y0, T, f, h_eps=1e-8, T_steps=1000, **kwargs):
    h = T/T_steps
    ans_grid = []
    ans_grid.append(y0)
    
    left = 0
    right = T
    t_grid = [0]
    t = left;
    #u_next = u(y0[0], y0[2])
    h_len = [h]
    #u_vls = [u_next]
    
    
    while(True):
        if (t >= right):
            break
        
        #u_last = u_next
        y_new = phi(t, ans_grid[-1], h, f, **kwargs)
        #u_next = u(y_new[0], y_new[2])
        
        #print(u_last, y_new, u_next)
        
        ans_grid.append(y_new)
        t += h
        t_grid.append(t)
        h_len.append(h)
            
        if np.isnan(ans_grid[-1][0]):
            break
            
    return (t_grid, ans_grid, h_len)

# ---------------------------------------Newton Solutions Finder--------------------------------------------------------

def derivatives(func, p, dim=3, d=1e-3, **kwargs): #func should return np.array
    zero = np.array([0.0]*dim) 
    d_vect = [zero]
    function_values = [func(*p, **kwargs)]
    Jac = []
    for i in range(dim):
        a = np.copy(zero)
        a[i] = d
        d_vect.append(a)
        function_values.append(func(*(p+a), **kwargs))
        Jac.append((function_values[-1] - function_values[0])/d)
        
    #print("f vals:", function_values)
    #print("d_vect vals:", d_vect)
    
    return np.array(Jac, ndmin=2).T
#string - fixed function
#column - fixed variable


def newtonian_finder(func, p0, silent=1, steps=50, Jac_recount=1, d=1e-5, dim=3, **kwargs): # поиск начальных условий методом Ньютона 
    p = np.copy(p0)
    f_vec = func(*p, **kwargs)
    for k in tqdm(range(steps), disable=silent):
        if(k%Jac_recount == 0):
            Jac_f = derivatives(func, p, dim, d, **kwargs)
            J_f_inv = np.linalg.inv(Jac_f)
        
        alpha = 1
        while (alpha > 1e-8):
            p_new = p - alpha * np.dot(J_f_inv, f_vec)
            f_vec_new = func(*p_new, **kwargs)
            if (np.linalg.norm(f_vec_new) < np.linalg.norm(f_vec)):
                p = p_new
                f_vec = f_vec_new
                if not silent:
                    print(f" Step {k}, point {p}: {np.linalg.norm(f_vec)}")
                break
            else: 
                alpha /= 2
                if not silent:
                    print(f" Step {k}, alpha changed {alpha}")
        
        
        if (alpha < 1e-8):
            print ("terminated")
            break
            
        if np.linalg.norm(f_vec) < 1e-10:
            break
       
    return p

    if (np.linalg.norm(f_vec) > 1e-10):
        return np.nan
    else:
        return p



# ---------------------------------------Discrepancy------------------------------------------------------------------







# ------------------------------Sample of portraits----------------------------------------------------
def random_strike_portrait(p1_range, boundary_conditions, p20 = 1, T = 20, lines = 50, generate=0):
    
    portrait = []
    for p10 in tqdm(p1_range):
        x10, x20 = boundary_conditions[0]
        y0 = np.array([x10, x20, p10, p20])
        t_grid, ans, us, hs = solve(y0, T, T_steps = 500)
        ans_grid = np.array(ans).T[0:2]

        portrait.append(ans_grid)
    
    
    if lines < len(p1_range):
        l = random.sample(portrait, lines)
    else:
        l = portrait
    
    plt.figure(figsize=(10,8))
    
    plt.plot(boundary_conditions[0][0], boundary_conditions[0][1], 'ro')
    for line in l:
        plt.plot(line[0], line[1])
    plt.plot(boundary_conditions[1][0], boundary_conditions[1][1], 'ro')
    plt.axis([-1, 15, -1, 15])
    plt.show()


# -----------------------------Pretty output - all-imclusive---------------------------------------------------
#простро красивый код для вывода результата
def checksol(p10, p20, boundary_conditions, T=20, start = 0, end = -1,
            steplength_plot=0, u_plot = 0, p_plot = 0, x_plot = 1, phase_plot=1, save = ""):
    x10, x20 = boundary_conditions[0]
    y0 = np.array([x10, x20, p10, p20])
    t_grid, ans, us, hs = solve(y0, T, T_steps = 10000)
    ans = np.array(ans)
    ans_grid = ans.T[0:2]
    ans_grid_normalized = ans_grid - ans_grid.mean(axis=1).reshape(-1,1)
    
    
    Xc = X_alternative(boundary_conditions, ans, us)
    print("Данные невязки , невязка, правый конец решения):")
    print(f"Модуль невязки: {np.linalg.norm(Xc[0])}")
    print(f"Вектор невязки: {Xc[0]}")
    print(f"Правый конец решения: {ans[Xc[1]]}")
    
    if end == -1:
        end = Xc[1]
    
    if phase_plot:
        print("Plase plot of x1, x2")
        plt.figure(figsize=(8,8))
        plt.plot(boundary_conditions[0][0], boundary_conditions[0][1], 'ro')
        plt.plot(boundary_conditions[1][0], boundary_conditions[1][1], 'ro')
        plt.plot(ans_grid[0][start:end], ans_grid[1][start:end])
        arrow = [ans[int(np.round(Xc[1]/2, 0))][0:2], ans[int(np.round(Xc[1]/2, 0))+1][0:2] ]
        plt.arrow(arrow[0][0], arrow[0][1], 
                  arrow[1][0]-arrow[0][0], arrow[1][1] - arrow[0][1], 
                  shape='full', lw=0, length_includes_head=True, head_width=.2)
        plt.axis([-1, 10, -1, 10])
        plt.show()
        
    
    if x_plot:
        print("Plot of x1, x2")
        plt.figure(figsize=(7,6))
        plt.plot(t_grid[start:end], ans_grid.T[start:end])
        plt.show()
    
    #plt.plot(t_grid, ans_grid_normalized.T)
    #plt.show()
    
    if p_plot:
        print("Plot of p1, p2")
        plt.plot(t_grid, ans.T[2:].T)
        plt.show()
    
    if u_plot:
        print("Plot of u:")
        plt.plot(np.array(us)[start:end])
        plt.show()

        
    if steplength_plot:
        print("Plot of step length:")
        plt.plot(hs[start:end])
        plt.show()
        
    if len(save) > 0:
        with open(save, 'w') as X__:
            X__.write(f"p10: {p10}, p20: {p20}\n")
            X__.write(f"boundary_conditions: {boundary_conditions}\n")
            for i in range(Xc[1]+1):  
                X__.write(f"{t_grid[i]}, {ans[i][0]}, {ans[i][1]}, {ans[i][2]}, {ans[i][3]}, {us[i]} \n")
                #X__.write(f"trajectory: {ans[0:Xc[1]+1].tolist()}\n")
        