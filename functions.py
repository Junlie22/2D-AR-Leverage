import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

def get_cov_loc(order=[1,1]):
    cov_loc = []
    p = order[0]
    q = order[1]
    for i in range(-p,p+1):
        for j in range(-q,q+1):
            if((i==0) & (j==0)): continue
            cov_loc.append([i,j])
    return cov_loc


def two_dim_AR_lev(image, order=[1,1]):
    p = order[0]
    q = order[1]
    [m,n] = image.shape
    
    lev = np.zeros([m,n])
    cov_loc = get_cov_loc(order)
    X = np.zeros([(m-2*p)*(n-2*q),len(cov_loc)])

    for i in range(p,m-p):
        for j in range(q,n-q):
            row = (i-p)*(n-2*q)+j-q
            col = 0
            for loc in cov_loc:
                k = i+loc[0]
                l = j+loc[1]
                X[row,col] = image[k,l]
                col = col + 1
    
    U = np.linalg.svd(X, full_matrices=False)[0]
    for i in range(p,m-p):
        for j in range(q,n-q):
            row = (i-p)*(n-2*q)+j-q
            lev[i,j] = np.sum(U[row,:] * U[row,:])
                
    #lev = lev / np.max(lev)

    lev_min = np.min(lev[p:-p,q:-q])
    lev_max = np.max(lev[p:-p,q:-q])
    lev = (lev - lev_min) / (lev_max - lev_min)
    lev[np.where(lev < 0)] = 0

    return lev



def two_dim_AR_lev_adjust(image, order=[1,1],method = 'min'):
    p = order[0]
    q = order[1]
    [m,n] = image.shape
    
    lev = np.zeros([m,n])
    lev_adjust = np.zeros([m,n])
    cov_loc = get_cov_loc(order)
    X = np.zeros([(m-2*p)*(n-2*q),len(cov_loc)])

    for i in range(p,m-p):
        for j in range(q,n-q):
            row = (i-p)*(n-2*q)+j-q
            col = 0
            for loc in cov_loc:
                k = i+loc[0]
                l = j+loc[1]
                X[row,col] = image[k,l]
                col = col + 1
    
    U = np.linalg.svd(X, full_matrices=False)[0]
    for i in range(p,m-p):
        for j in range(q,n-q):
            row = (i-p)*(n-2*q)+j-q
            lev[i,j] = np.sum(U[row,:] * U[row,:])
                
    if(method == 'min'):
        for i in range(p,m-p):
            for j in range(q,n-q):
                lev_temp_list = []
                for loc in cov_loc:
                    k = i+loc[0]
                    l = j+loc[1]
                    lev_temp_list.append(lev[k,l])
                lev_adjust[i,j] = np.min(lev_temp_list)

    #lev_adjust = lev_adjust / np.max(lev_adjust)

    lev_min = np.min(lev_adjust[2*p:-2*p,2*q:-2*q])
    lev_max = np.max(lev_adjust[2*p:-2*p,2*q:-2*q])
    lev_adjust = (lev_adjust - lev_min) / (lev_max - lev_min)
    lev_adjust[np.where(lev_adjust < 0)] = 0

    return lev_adjust



def ROC_curve(anomaly_score, GT, tau_list = np.linspace(0.01,1,100)):
    TPR_list = []
    FPR_list = []
    [m,n] = GT.shape
    indices = np.where(GT == 1)
    P_total = len(indices[0])
    for tau in tau_list:
        TP = 0
        P = len(np.where(anomaly_score > tau)[0])
        for i in range(P_total):
            row = indices[0][i]
            col = indices[1][i]
            if(anomaly_score[row,col] > tau):
                TP = TP + 1
        TPR = TP / P_total
        FP = P - TP
        FPR = FP / (m*n - P_total)
        TPR_list.append(TPR)
        FPR_list.append(FPR)

    return {'TPR_list': TPR_list, 'FPR_list':FPR_list}


def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)


def lev_adjust(lev, order=[1,1], method = 'min'):
    p = order[0]
    q = order[1]
    [m,n] = lev.shape
    lev_adjust = np.zeros([m,n])

    cov_loc = get_cov_loc(order)
    if(method == 'min'):
        for i in range(p,m-p):
            for j in range(q,n-q):
                lev_temp_list = []
                for loc in cov_loc:
                    k = i+loc[0]
                    l = j+loc[1]
                    lev_temp_list.append(lev[k,l])
                lev_adjust[i,j] = np.min(lev_temp_list)

    if(method == 'median'):
        for i in range(p,m-p):
            for j in range(q,n-q):
                lev_temp_list = []
                for loc in cov_loc:
                    k = i+loc[0]
                    l = j+loc[1]
                    lev_temp_list.append(lev[k,l])
                lev_adjust[i,j] = np.median(lev_temp_list)


    #lev_adjust = lev_adjust / np.max(lev_adjust)
    lev_min = np.min(lev_adjust[2*p:-2*p,2*q:-2*q])
    lev_max = np.max(lev_adjust[2*p:-2*p,2*q:-2*q])
    lev_adjust = (lev_adjust - lev_min) / (lev_max - lev_min)
    lev_adjust[np.where(lev_adjust < 0)] = 0

    return lev_adjust


def plt_target_frame(ax, target, is_HSI = False, scaling_ratio = 224 / 1600):
    shapes = target['shapes']
    for item in shapes:
        [x0,y0] = item['points'][0]
        [x1,y1] = item['points'][1]
        w = x1 - x0
        h = y1 - y0
        if(is_HSI) :
            x0 = x0 * scaling_ratio
            y0 = y0 * scaling_ratio
            w = w * scaling_ratio
            h = h * scaling_ratio
            
        rect = patches.Rectangle((x0,y0),w,h, linewidth=0.1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)


def map_from_frame(target, is_HSI = False, img_shape = [1600,1600], scaling_ratio = 224 / 1600):
    [m,n] = img_shape
    if(is_HSI):
        m = int(m * scaling_ratio)
        n = int(n * scaling_ratio)
    map = np.zeros([m,n])
    shapes = target['shapes']
    for item in shapes:
        [x0,y0] = item['points'][0]
        [x1,y1] = item['points'][1]
        if(is_HSI) :
            x0 = x0 * scaling_ratio
            y0 = y0 * scaling_ratio
            x1 = x1 * scaling_ratio
            y1 = y1 * scaling_ratio
        xl = min(x0,x1)
        xr = max(x0,x1)
        yl = min(y0,y1)
        yr = max(y0,y1)
        for i in range(max(0,round(xl)), min(round(xr),m)):
            for j in range(max(0,round(yl)), min(round(yr),n)):
                map[j,i] = 1

    return map


def HSI_average(HSI, stride = 10):
    [m,n,c] = HSI.shape
    l = len(np.array(range(1,c,stride)))
    HSI_avg = np.zeros([m,n,l])
    for i in range(l-1):
        HSI_avg[:,:,i] = np.mean(HSI[:,:,(i*stride):(i*stride+10)] ,axis = 2)
    HSI_avg[:,:,l-1] = np.mean(HSI[:,:,((l-1)*stride):] ,axis = 2)
    return HSI_avg


def tensor_multiply(A, B):
    if(len(A.shape) == 2):
        [i,j] = A.shape
        A = A.reshape(i,j,1)
    if(len(B.shape) == 2):
        [i,j] = B.shape
        B = B.reshape(i,j,1)
    m = A.shape[2]
    n = B.shape[2]
    res = np.zeros([m,n])
    for i in range(m):
        for j in range(n):
            res[i,j] = np.sum(A[:,:,i] * B[:,:,j])
    return res


def tensor_vec_multiply(A, b):
    [m,n] = A.shape[0:2]
    h = A.shape[2]
    res = np.zeros([m,n])
    for i in range(h):
        res = res + A[:,:,i] * b[i]
    return res


def two_dim_param_estimate_leverage(XX, X, lev_score = [], subsample_size = 0):
    if(subsample_size == 0 or len(lev_score) == 0):
        A = tensor_multiply(XX,XX)
        b = tensor_multiply(XX,X)
        return np.linalg.solve(A, b)
    else:
        prob = lev_score / np.sum(lev_score)
        index = np.random.choice(prob.size, size=subsample_size, replace=False , p=prob.flatten())
        [m,n] = X.shape[0:2]
        i = index // n
        j = index % n
        XX_sub = XX[i,j,:]
        X_sub = X[i,j]
        A = XX_sub.T @ XX_sub
        b = XX_sub.T @ X_sub
        return np.linalg.solve(A, b)
    

def two_dim_AR_lev_iteration(image, order=[1,1]):
    p = order[0]
    q = order[1]
    [m,n] = image.shape

    lev = np.zeros([m,n])
    lev_temp = np.zeros([m-2*p,n-2*q])
    cov_loc = get_cov_loc(order)
    XX = np.zeros([(m-2*p), (n-2*q),len(cov_loc)])
    i = 0
    for loc in cov_loc:
        k = loc[0]
        l = loc[1]
        XX[:,:,i] = image[(p+k):(m-p+k), (q+l):(n-q+l)]
        i = i + 1
    
    lev_temp = XX[:,:,0]**2 / np.sum(XX[:,:,0]**2)
    phi = np.sum(XX[:,:,0] * XX[:,:,1]) / np.sum(XX[:,:,0]**2)
    residual = XX[:,:,1] - XX[:,:,0] * phi
    lev_temp = lev_temp + residual**2 / np.sum(residual**2)

    for s in range(2,len(cov_loc)):
        phi = two_dim_param_estimate_leverage(XX[:,:,:s], XX[:,:,s])
        residual = XX[:,:,s] - tensor_vec_multiply(XX[:,:,:s], phi)
        lev_temp = lev_temp + residual**2 / np.sum(residual**2)

    lev[p:-p,q:-q] = lev_temp
    return lev


def fast_2d_AR_leverage_score(image, order=[1,1], subsample_size = 1000):
    p = order[0]
    q = order[1]
    [m,n] = image.shape

    lev = np.zeros([m,n])
    lev_temp = np.zeros([m-2*p,n-2*q])
    cov_loc = get_cov_loc(order)
    XX = np.zeros([(m-2*p), (n-2*q),len(cov_loc)])
    i = 0
    for loc in cov_loc:
        k = loc[0]
        l = loc[1]
        XX[:,:,i] = image[(p+k):(m-p+k), (q+l):(n-q+l)]
        i = i + 1
    
    lev_temp = XX[:,:,0]**2 / np.sum(XX[:,:,0]**2)
    phi = np.sum(XX[:,:,0] * XX[:,:,1]) / np.sum(XX[:,:,0]**2)
    residual = XX[:,:,1] - XX[:,:,0] * phi
    lev_temp = lev_temp + residual**2 / np.sum(residual**2)

    for s in range(2,len(cov_loc)):
        phi = two_dim_param_estimate_leverage(XX[:,:,:s], XX[:,:,s], lev_score = lev_temp, subsample_size = subsample_size)
        residual = XX[:,:,s] - tensor_vec_multiply(XX[:,:,:s], phi)
        lev_temp = lev_temp + residual**2 / np.sum(residual**2)

    lev[p:-p,q:-q] = lev_temp
    return lev


def MSE_calculate(XX, Y, phi):
    Yhat = tensor_vec_multiply(XX, phi)
    MSE = np.mean((Y - Yhat)**2)
    return MSE


def causal_get_cov_loc(order=[2,2]):
    cov_loc = []
    p = order[0]
    q = order[1]
    for i in range(-p,1):
        for j in range(-q,1):
            if((i==0) & (j==0)): continue
            cov_loc.append([i,j])
    return cov_loc


def causal_fast_2d_AR_leverage_score(image, order=[2,2], subsample_size = 1000):
    p = order[0]
    q = order[1]
    [m,n] = image.shape

    lev = np.zeros([m,n])
    lev_temp = np.zeros([m-p,n-q])
    cov_loc = causal_get_cov_loc(order)
    XX = np.zeros([(m-p), (n-q),len(cov_loc)])
    i = 0
    for loc in cov_loc:
        k = loc[0]
        l = loc[1]
        XX[:,:,i] = image[(p+k):(m+k), (q+l):(n+l)]
        i = i + 1
    
    start_time = time.time()
    lev_temp = XX[:,:,0]**2 / np.sum(XX[:,:,0]**2)
    phi = np.sum(XX[:,:,0] * XX[:,:,1]) / np.sum(XX[:,:,0]**2)
    residual = XX[:,:,1] - XX[:,:,0] * phi
    lev_temp = lev_temp + residual**2 / np.sum(residual**2)
    time_fast = time.time() - start_time

    for s in range(2,len(cov_loc)):
        phi = two_dim_param_estimate_leverage(XX[:,:,:s], XX[:,:,s], lev_score = lev_temp, subsample_size = subsample_size)
        residual = XX[:,:,s] - tensor_vec_multiply(XX[:,:,:s], phi)
        lev_temp = lev_temp + residual**2 / np.sum(residual**2)

    lev[p:,q:] = lev_temp
    return lev


def beta_generate(order, method = '1'):
    p = order[0]
    q = order[1]
    beta = np.zeros([p+1,q+1])

    if(method == '1'):
        for i in range(p+1):
            for j in range(q+1):
                beta[i,j] = np.power(-1,i+j) * (i+j) * 0.02 / (p+q)
        beta[p,q] = 0
    
    if(method == '2'):
        for i in range(p+1):
            for j in range(q+1):
                b = np.sum(np.random.randn(1) / p / q / 5)
                b = np.min([b, 1/(p+1)/(q+1)/2])
                b = np.max([b, -1/(p+1)/(q+1)/2])
                beta[i,j] = b
        beta[p,q] = 0

    if(method == '3'):
        for i in range(p+1):
            for j in range(q+1):
                b = np.sum(0.5 * (np.random.rand(1) - 0.5))
                beta[i,j] = b / (p+1)/(q+1)
        beta[p,q] = 0

    return beta


def tensor_multiply(A, B):
    if(len(A.shape) == 2):
        [i,j] = A.shape
        A = A.reshape(i,j,1)
    if(len(B.shape) == 2):
        [i,j] = B.shape
        B = B.reshape(i,j,1)
    m = A.shape[2]
    n = B.shape[2]
    res = np.zeros([m,n])
    for i in range(m):
        for j in range(n):
            res[i,j] = np.sum(A[:,:,i] * B[:,:,j])
    return res


def tensor_vec_multiply(A, b):
    [m,n] = A.shape[0:2]
    h = A.shape[2]
    res = np.zeros([m,n])
    for i in range(h):
        res = res + A[:,:,i] * b[i]
    return res


def two_dim_param_estiamte_leverage(XX, X, lev_score = [], subsample_size = 0):
    if(subsample_size == 0 or len(lev_score) == 0):
        A = tensor_multiply(XX,XX)
        b = tensor_multiply(XX,X)
        return np.linalg.solve(A, b)
    else:
        prob = lev_score / np.sum(lev_score)
        index = np.random.choice(prob.size, size=subsample_size, replace=False , p=prob.flatten())
        [m,n] = X.shape[0:2]
        i = index // n
        j = index % n
        XX_sub = XX[i,j,:]
        X_sub = X[i,j]
        A = XX_sub.T @ XX_sub
        b = XX_sub.T @ X_sub
        return np.linalg.solve(A, b)
    

def two_dim_AR_lev_iteration(image, order=[1,1]):
    p = order[0]
    q = order[1]
    [m,n] = image.shape

    lev = np.zeros([m,n])
    lev_temp = np.zeros([m-2*p,n-2*q])
    cov_loc = get_cov_loc(order)
    XX = np.zeros([(m-2*p), (n-2*q),len(cov_loc)])
    i = 0
    for loc in cov_loc:
        k = loc[0]
        l = loc[1]
        XX[:,:,i] = image[(p+k):(m-p+k), (q+l):(n-q+l)]
        i = i + 1
    
    lev_temp = XX[:,:,0]**2 / np.sum(XX[:,:,0]**2)
    phi = np.sum(XX[:,:,0] * XX[:,:,1]) / np.sum(XX[:,:,0]**2)
    residual = XX[:,:,1] - XX[:,:,0] * phi
    lev_temp = lev_temp + residual**2 / np.sum(residual**2)

    for s in range(2,len(cov_loc)):
        phi = two_dim_param_estiamte_leverage(XX[:,:,:s], XX[:,:,s])
        residual = XX[:,:,s] - tensor_vec_multiply(XX[:,:,:s], phi)
        lev_temp = lev_temp + residual**2 / np.sum(residual**2)

    lev[p:-p,q:-q] = lev_temp
    return lev


def fast_2d_AR_leverage_score(image, order=[1,1], subsample_size = 1000):
    p = order[0]
    q = order[1]
    [m,n] = image.shape

    lev = np.zeros([m,n])
    lev_temp = np.zeros([m-2*p,n-2*q])
    cov_loc = get_cov_loc(order)
    XX = np.zeros([(m-2*p), (n-2*q),len(cov_loc)])
    i = 0
    for loc in cov_loc:
        k = loc[0]
        l = loc[1]
        XX[:,:,i] = image[(p+k):(m-p+k), (q+l):(n-q+l)]
        i = i + 1
    
    lev_temp = XX[:,:,0]**2 / np.sum(XX[:,:,0]**2)
    phi = np.sum(XX[:,:,0] * XX[:,:,1]) / np.sum(XX[:,:,0]**2)
    residual = XX[:,:,1] - XX[:,:,0] * phi
    lev_temp = lev_temp + residual**2 / np.sum(residual**2)

    for s in range(2,len(cov_loc)):
        phi = two_dim_param_estiamte_leverage(XX[:,:,:s], XX[:,:,s], lev_score = lev_temp, subsample_size = subsample_size)
        residual = XX[:,:,s] - tensor_vec_multiply(XX[:,:,:s], phi)
        lev_temp = lev_temp + residual**2 / np.sum(residual**2)

    lev[p:-p,q:-q] = lev_temp
    return lev