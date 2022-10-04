import numpy as np
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm
import random

#------------------------------------------System of ODE -------------------------------------------
def u(x1, p1):
        if (x1*p1 >= 0):
            return 1
        else:
            return -1

def f(t, y):
    x1, x2, p1, p2 = y
    u_ = u(x1, p1)
    #print(y, u_)
    
    xx1 = (u_ - 2/3)*x1 + x2/6
    xx2 = 2/3*x1 - x2/6
    pp1 = -p1*u_ + 2/3*(p1 - p2)
    pp2 = -1/6*(p1 - p2)
    return np.array([xx1, xx2, pp1, pp2])

#------------------------------------------Runge–Kutta ------------------------------------------------------------------
def phi(t, y, h):
    #h = T/T_steps
    #print(t, y)
    k1 = f(t, y)
    k2 = f(t + h/2, y + h/2*k1)
    k3 = f(t + h/2, y + h/2*k2)
    k4 = f(t + h, y + h*k3)
    
    return y + h/6*(k1 + 2*k2 + 2*k3 + k4)

#if nan stops
def solve(y0, T, h_eps=1e-8, T_steps=1000):
    h = T/T_steps
    ans_grid = []
    ans_grid.append(y0)
    
    left = 0
    right = T
    t_grid = [0]
    t = left;
    u_next = u(y0[0], y0[2])
    h_len = [h]
    u_vls = [u_next]
    
    
    while(True):
        if (t+h >= right):
            break
        
        u_last = u_next
        y_new = phi(t, ans_grid[-1], h)
        u_next = u(y_new[0], y_new[2])
        
        #print(u_last, y_new, u_next)
        
        if (u_last == u_next or h < h_eps):
            ans_grid.append(y_new)
            u_vls
            t += h
            t_grid.append(t)
            h_len.append(h)
            u_vls.append(u_next)
            if (h < h_eps):
                h = T/T_steps
        else:
            h = h/2
            u_next = u_last
        
        
        
        #if (right - t < h):
        #    ans_grid.append(phi(t, ans_grid[-1], right - t, alpha=alpha))
        #    t_grid.append(right)
        #    break
            
        if np.isnan(ans_grid[-1][0]):
            break
    return (t_grid, ans_grid, u_vls, h_len)


# ---------------------------------------Discrepancy------------------------------------------------------------------
def X_alternative(boundary_conditions, ans, us): 
    ans_grid = np.array(ans).T[0:2]    
    opt_id = np.argmin(np.linalg.norm((ans_grid - np.array(boundary_conditions[1]).reshape(-1,1)).T, axis=1))
    
    #print(ans_grid)
    #print((ans_grid - np.array(boundary_conditions[1]).reshape(-1,1)).T[1867])
    #print(np.linalg.norm((ans_grid - np.array(boundary_conditions[1]).reshape(-1,1)).T, axis=1)[1867])
    
    if (opt_id == len(ans)):
        print(f"Optimal id {opt_id} is last!")
    x1, x2, p1, p2 = ans[opt_id]
    X_ = boundary_conditions[1] - ans[opt_id][0:2]
    return [X_, opt_id]


# ----------------------------A bit slow and stupid grid-search for initial value----------------------------------------
def initial_search(boundary_conditions, T = 20, bc=0, step=0.001, T_steps = 1000, cur_range = []):
    X_arr1 = []
    X_arr2 = []
    if len(cur_range)==0:
        cur_range = np.arange(-2, 2, step)
    
    p20 = -1
    for p10 in tqdm(cur_range):
        x10, x20 = boundary_conditions[0]
        y0 = np.array([x10, x20, p10, p20])
        t_grid, ans, us, hs = solve(y0, T, T_steps = T_steps)
        ans_grid = np.array(ans).T[0:2]
        Xc = X_alternative(boundary_conditions, ans, us) #Xcomplicated = [X, opt_id]
        X = Xc[0]
        X_arr1.append(np.linalg.norm(X))

    plt.plot(cur_range, X_arr1)
    plt.show()
    
    p20 = 1
    # так как уже проверено, что на этом участке решений нет, для экономии времени:
    cur_range = np.arange(-2, 2, 0.1)
    for p10 in tqdm(cur_range):
        x10, x20 = boundary_conditions[0]
        y0 = np.array([x10, x20, p10, p20])
        t_grid, ans, us, hs = solve(y0, T, T_steps = T_steps)
        ans_grid = np.array(ans).T[0:2]
        Xc = X_alternative(boundary_conditions, ans, us) #Xcomplicated = [X, opt_id]
        X = Xc[0]
        X_arr2.append(np.linalg.norm(X))

    plt.plot(cur_range, X_arr2)
    plt.show()
    
    return X_arr1, X_arr2


# ------------------------------Loss and 1d-gradient descent----------------------------------------------------
def LOSS(p10, boundary_conditions, p20=-1, T=20):
    x10, x20 = boundary_conditions[0]
    y0 = np.array([x10, x20, p10, p20])
    t_grid, ans, us, hs = solve(y0, T, T_steps = 1000)
    ans_grid = np.array(ans).T[0:2]
    ans_grid_normalized = ans_grid - ans_grid.mean(axis=1).reshape(-1,1)
    
    #print(boundary_conditions, ans[-1], us[-1])
    Xc = X_alternative(boundary_conditions, ans, us)
    return(np.linalg.norm(Xc[0]))

def idiot_optimizer(func, p0, others=[], alpha=1e-5, silent=0, steps=50):
    p = np.copy(p0)
    f_val = func(*p,*others)
    for k in tqdm(range(steps)):
        f1 = func(*p+alpha,*others)
        f2 = func(*p-alpha,*others)
        while (np.abs(f1) >= np.abs(f_val) and np.abs(f2) >= np.abs(f_val)):
            if alpha < 1e-18:
                print("Interrupted! Step less than 1e-18!")
                return p
            alpha /= 2
            if not silent:
                print(f" Step {k}, alpha changed {alpha}")
            f1 = func(*p+alpha, *others)
            f2 = func(*p-alpha, *others)
        if np.abs(f1) < np.abs(f_val):
            f_val, p = f1, p+alpha
            if not silent:
                print(f" Step {k}, point {p}: {f_val}")
        else:
            f_val, p = f2, p-alpha
            if not silent: 
                print(f" Step {k}, point {p}: {f_val}")
        if np.abs(f_val) < 1e-12:
            print("Interrupted! Loss function value less then 1e-12!")
            break
    if k == steps-1:
        print("Interrupted! Max steps!")
    return p


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
        