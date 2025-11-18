from scipy.optimize import root_scalar
from scipy.stats import sem
from scipy.special import logsumexp
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

def pull_sample(data, n):
    if type(data) is list:
        length = [len(item) for item in data]
        L_test = [L==length[0] for L in  length]
        assert all(L_test), 'data list elements needs to have the same length'
        data_length = length[0]
    else:
        data_length = len(data)
                   
    decider = np.random.choice(range(data_length),n, replace=False)
    if type(data) is list:
        return [item[decider] for item in data]
    else:
        return data[decider]


def calculate_delta_F(system, times=None, resolution=1000, beta=1, domain=[[-5],[5]]):
    if times is None:
        try:
            times = system.protocol.t_i, system.protocol.t_f
        except:
            print('no times given, and system has no protocol')
            return

    U_0, X = system.lattice(times[0], resolution, manual_domain=domain)
    U_f, X = system.lattice(times[-1], resolution, manual_domain=domain)

    BF_0 = -np.log(np.trapz(np.exp(-beta*U_0), x=X[0]))
    BF_f = -np.log(np.trapz(np.exp(-beta*U_f), x=X[0]))
    return BF_f-BF_0

def get_mean_error(sample):
    return np.mean(sample), sem(sample)**2

def get_proportion_error(prob_bool):
    N = len(prob_bool)
    p = sum(prob_bool)/N
    var = p*(1-p)/N
    return p, var

def class_meta_f(class_work, probability_bools=None, beta=1, sample_error=True, z=1):
    if sample_error:
        sample_var = sem(np.exp(-beta * class_work))**2
    else:
        sample_var = 0

    estimate = -logsumexp(-beta*class_work)+np.log(len(class_work))

    if probability_bools is not None:
        tcft_mean, tcft_var = tcft_correction(*probability_bools)
        sample_var = sample_var/np.exp(estimate)**2 + tcft_var
        estimate += tcft_mean
        
    if sample_error:
        return estimate, z*np.sqrt(sample_var)
    else:
        return estimate, None

def tcft_correction(probs, rev_probs):
    p, var = get_proportion_error(probs)
    p_rev, var_rev = get_proportion_error(rev_probs)

    return -np.log(p/p_rev), var/p**2 + var_rev/p_rev**2

def bennet_delta_F(f_works, r_works, probability_bools=None, beta=1, z=1):
    nf = len(f_works)
    nr = len(r_works)
    nt = nr+nf
    M = np.log(nf/nr)/beta

    def get_diff(Delta_F):
        S1 = (1/(1+np.exp(beta*(M+f_works-Delta_F)))).sum()
        S2 = (1/(1+np.exp(-beta*(M-r_works-Delta_F)))).sum()
        return S1-S2
    
    Delta_F = ((nf/nt)*np.log(np.mean(np.exp(beta*f_works))) + (nr/nt)*np.log(np.mean(np.exp(-beta*r_works))))/beta
    
    guesses = [Delta_F*(1+p) for p in [-.1,.1]]
    sol = root_scalar(get_diff, x0=guesses[0], x1=guesses[1]);
    

    Delta_F = sol.root
    
    var = (1/nt)*(1/np.mean(1/(2+2*np.cosh(beta*(M+np.append(f_works,-r_works)-Delta_F))))-nt**2/(nr*nf))
    if not probability_bools == None:
        tcft_mean, tcft_var = tcft_correction(*probability_bools)
        Delta_F += tcft_mean
        var += tcft_var

    return beta*Delta_F, z*np.sqrt(var)
'''
def check_JAR_variance(final_W, forward_prob, reverse_prob, sample_size, repetitions,  b=1, TCFT=True):
    metastable_f = np.zeros(2,repetitions)
    bennet_f = np.zeros(2,repetitions)
    i=0
    while i < repetitions:
        if TCFT:
            W, P = pull_sample([final_W, forward_prob], sample_size)
            R = pull_sample(reverse_prob, sample_size)
            metastable_f[:,i] = class_meta_f(W[P], probability_bools=[P, R], beta=b, sample_error=True)
        else:
            W = pull_sample(final_W, sample_size)
            metastable_f[:,i] = class_meta_f(W, probability_bools=None, beta=b, sample_error=False)
        i+=1
        print("done {} samples out of {}".format(i, repetitions), end="\r")
    return metastable_f

def check_BAR_variance(final_W, reverse_final_W, forward_prob, reverse_prob, sample_size, repetitions,  b=1, TCFT=True):
    bennet_f = np.zeros(2,repetitions)
    i=0
    while i < repetitions:
        if TCFT:
            W, P = pull_sample([final_W, forward_prob], sample_size)
            WR, R = pull_sample([reverse_final_W, reverse_prob], sample_size)
            metastable_f[:,i] =
        else:
            W = pull_sample(final_W, sample_size)
            metastable_f[:,i] = class_meta_f(W, probability_bools=None, beta=b, sample_error=False)
        i+=1
        print("done {} samples out of {}".format(i, repetitions), end="\r")
    return metastable_f
'''
def variance_plot(samples, parameter = None, ax = None, z_score=1.96, asym=False):
    # s is list of N [estimate, estimated_error]
    # ax is an ax to plot on
    # the actual values of the parameter, if None it will populate with [<s[0]>]
    # z_score is for setting the size of the confidence interval we are tying to make
    import scipy.stats as st
    confidence = 2*(1-st.norm.cdf(z_score))
    
    s = np.array(samples)
    s[:,1] *= z_score

    if parameter is None:
        parameter = np.mean(s[:,0])

    return_ax = False
    if ax is None:
        return_ax = True
        ax = plt.subplots()
        
    colors=['b','r']
    fail = ~((s[:,0]-s[:,1] <= parameter) & (parameter <= s[:,0]+s[:,1] ))
    
    c = np.array([ colors[1] if f else colors[0] for f in fail ])
    title_string = f'CI fail:{sum(fail)/len(fail):.1%}; expected {confidence:.1%}'
    if asym:
        asym = np.sum(np.sign(s[:,0]-parameter))/len(s)
        title_string += f'symm:{(1+asym)/(1-asym):.3f}'
    ax.set_title(title_string)
    ax.errorbar(range(len(s)), s[:,0], yerr=s[:,1], linestyle='none', marker='o', ecolor=c, c='k')
    ax.axhline(parameter, linestyle='--', c='k', zorder=10_000, label='true_avg')
    ax.axhline(s[:,0].mean(), linestyle='-', c='g', zorder=9_999, label='sampling_avg')
    ax.legend()
    if return_ax:
        return ax
    else:
        return

        
    