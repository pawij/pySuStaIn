###
# pySuStaIn: SuStaIn algorithm in Python (https://www.nature.com/articles/s41468-018-05892-0)
# Author: Peter Wijeratne (p.wijeratne@ucl.ac.uk)
# Contributors: Leon Aksman (l.aksman@ucl.ac.uk), Arman Eshaghi (a.eshaghi@ucl.ac.uk)
###
import numpy as np
import multiprocessing
from matplotlib import pyplot as plt
from simfuncs import generate_random_sustain_model, generate_data_sustain
from funcs import run_sustain_algorithm, cross_validate_sustain_model
import os
from sklearn.model_selection import KFold, StratifiedKFold
from multiprocessing import Pool, cpu_count
import functools

import sys
sys.path.insert(0, "/home/paw/code/GP_progression_model_V2/src/GP_progression_model/")
import DataGenerator

def main():
    # cross-validation
    validate = False    
    #    num_cores = multiprocessing.cpu_count()
    num_cores = 1

    # if same variance set for every distribution...
    # to capture first changes need to either have a z-score between 0 < z < 1, or scale absoluate value by normal distribution (like EBM)
    # otherwise distribution with more measurements near max abnormality (i.e. higher variance) will always come first
    
    N_bio = 3  # number of biomarkers
    N_sub = 200 # number of people
    N_tps = 2 # number of measurements / person
    dt = 3 # time between measurements
        
    def f(x, L, k, x0):
        return L / (1 + np.exp(-k * (x - x0)))
    time = []
    for i in range(N_sub):
        temp = []
        for j in range(N_tps):
            temp.append( (np.random.rand()*12. - 6.) + j*dt )
        temp1 = []
        for j in range(N_bio):
            temp1.append(temp)
        time.append(temp1)        
    time = np.array(time)
    L = 1.
    # [k, x0, noise]
    pars = []
    """
    for i in range(N_bio):
        pars.append([np.random.rand(), 0, 0])
        #        pars.append([np.random.rand(), np.random.rand()*3-6, 0])
        #        pars.append([np.random.rand()*.5+.5, np.random.rand()*3, 0])    
    """
    pars = [[0.85, 0, 0.05], [0.15, 0, 0.05], [0.43, 0, 0.05]]
    data = []
    for i in range(N_sub):
        temp = []
        for j in range(N_bio):
            noise = np.random.normal(0, 0.05, 1)#(np.random.rand()*2-1)*pars[j][2]
            temp.append(f(time[i][j], L, pars[j][0], pars[j][1]) + noise - f(-6, L, pars[j][0], pars[j][1]))
        data.append(temp)
    data = np.array(data)
    bio_mean_std = []
    for i in range(N_bio):
        temp_mean = []
        temp_std = []
        for j in range(N_tps):
            temp_mean.append(np.mean(data[:,i,j]))
            temp_std.append(np.std(data[:,i,j]))
        bio_mean_std.append([temp_mean, temp_std])
    bio_mean_std = np.array(bio_mean_std)
    print (bio_mean_std)
    # plot
    numY, numX = (int(np.ceil(np.sqrt(data.shape[1]))),
                  int(round(np.sqrt(data.shape[1]))))
    if numX < 2:
        numX = 2
    #    fig, ax = plt.subplots(numX, numY)
    fig, ax = plt.subplots()
    col = [['r','yellow'],
           ['b','purple'],
           ['g','darkgreen'],
           ['grey','black']]
    style = ['-','--','.','-.-']
    Xrange = np.linspace(-6, 6+dt, 100)
    for i in range(data.shape[1]):
        for k in range(data.shape[2]):
            ax.scatter(time[:,i,k], data[:,i,k], color=col[i][k], s=.5)
            X_t = np.array([x+k*dt for x in Xrange])
            Y_t = np.subtract(f(X_t, L, pars[i][0], pars[i][1]), f(-6, L, pars[i][0], pars[i][1]))
            ax.plot(X_t, Y_t, color=col[i][k])
            ax.plot([-7,8], [np.mean(data[:,i,k]), np.mean(data[:,i,k])], linestyle=style[k], color=col[i][k], label='biom '+str(i))
        handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    fig, ax = plt.subplots()
    for i in range(data.shape[1]):
        for k in range(data.shape[2]):
            ax.hist(data[:,i,k], bins=np.arange(0, L+.5, 0.1), color=col[i][k], alpha=.5, label='biom '+str(i))
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    #
    #    plt.show()

    #    data = data[:,:,0]
    
    Z_vals = np.array([[1,2]]*N_bio) # Z-scores for each biomarker
    #    Z_vals = np.array([[1]]*N_bio) # Z-scores for each biomarker
    IX_vals = np.array([[x for x in range(N_bio)]]*2).T
    #    IX_vals = np.array([[x for x in range(N_bio)]]).T
    Z_max = np.array([2]*N_bio) # maximum z-score
    #    Z_max = np.array([1]*N_bio) # maximum z-score
    stage_zscore = np.array([y for x in Z_vals.T for y in x])
    stage_zscore = stage_zscore.reshape(1,len(stage_zscore))
    stage_biomarker_index = np.array([y for x in IX_vals.T for y in x])
    stage_biomarker_index = stage_biomarker_index.reshape(1,len(stage_biomarker_index))
    min_biomarker_zscore = [0]*N_bio;
    max_biomarker_zscore = Z_max;
    std_biomarker_zscore = [1]*N_bio;

    print ('stage_zscore',stage_zscore)
    print ('stage_biomarker_index',stage_biomarker_index)
    
    """
    N_S_gt = 1 # number of ground truth subtypes
    SuStaInLabels = []
    SuStaInStageLabels = []
    # ['Biomarker 0', 'Biomarker 1', ..., 'Biomarker N' ]
    for i in range(N):
        SuStaInLabels.append( 'Biomarker '+str(i))
    for i in range(len(stage_zscore)):
        SuStaInStageLabels.append('B'+str(stage_biomarker_index[i])+' - Z'+str(stage_zscore[i]))

    gt_f = [1+0.5*x for x in range(N_S_gt)]
    gt_f = [x/sum(gt_f) for x in gt_f][::-1]
    # ground truth sequence for each subtype
    gt_sequence = generate_random_sustain_model(stage_zscore,stage_biomarker_index,N_S_gt)
    
    N_k_gt = np.array(stage_zscore).shape[1]+1
    subtypes = np.random.choice(range(N_S_gt),N_sub,replace=True,p=gt_f)
    stages = np.ceil(np.random.rand(N_sub,1)*(N_k_gt+1))-1
    data, data_denoised, stage_value = generate_data_sustain(subtypes,
                                                             stages,
                                                             gt_sequence,
                                                             min_biomarker_zscore,
                                                             max_biomarker_zscore,
                                                             std_biomarker_zscore,
                                                             stage_zscore,
                                                             stage_biomarker_index)
    """
    # number of starting points 
    N_startpoints = 25
    # maximum number of subtypes 
    N_S_max = 1
    N_iterations_MCMC = int(1e4)
    N_iterations_MCMC_opt = int(1e3)
    
    likelihood_flag = 'Approx'
    output_folder = 'test'
    dataset_name = 'test'
    
    samples_sequence, samples_f = run_sustain_algorithm(data,
                                                        min_biomarker_zscore,
                                                        max_biomarker_zscore,
                                                        std_biomarker_zscore,
                                                        stage_zscore,
                                                        stage_biomarker_index,
                                                        N_startpoints,
                                                        N_S_max,
                                                        N_iterations_MCMC,
                                                        likelihood_flag,
                                                        output_folder,
                                                        dataset_name,
                                                        N_iterations_MCMC_opt, 
                                                        num_cores,
                                                        bio_mean_std)

    if validate:
        ### USER DEFINED INPUT: START
        # test_idxs: indices corresponding to 'data' for test set, with shape (N_folds, data.shape[0]/N_folds)
        # select_fold: index of a single fold from 'test_idxs'. For use if this code was to be run on multiple processors
        # target: stratification is done based on the labels provided here. For use with sklearn method 'StratifiedKFold'
        test_idxs = []
        select_fold = []
        target = []
        ### USER DEFINED INPUT: END
        
        if not test_idxs:
            print(
                '!!!CAUTION!!! No user input for cross-validation fold selection - using automated stratification. Only do this if you know what you are doing!')
            N_folds = 10
            if target:
                cv = StratifiedKFold(n_splits=N_folds, shuffle=True)
                cv_it = cv.split(data, target)
            else:
                cv = KFold(n_splits=N_folds, shuffle=True)
                cv_it = cv.split(data)
            for train, test in cv_it:
                test_idxs.append(test)
            test_idxs = np.array(test_idxs)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        if select_fold:
            test_idxs = test_idxs[select_fold]
        Nfolds = len(test_idxs)

        pool = Pool(num_cores)
        copier = functools.partial(cross_validate_sustain_model,
                                   data=data,
                                   test_idxs=test_idxs,
                                   min_biomarker_zscore=min_biomarker_zscore,
                                   max_biomarker_zscore=max_biomarker_zscore,
                                   std_biomarker_zscore=std_biomarker_zscore,
                                   stage_zscore=stage_zscore,
                                   stage_biomarker_index=stage_biomarker_index,
                                   N_startpoints=N_startpoints,
                                   N_S_max=N_S_max,
                                   N_iterations_MCMC=N_iterations_MCMC,
                                   likelihood_flag=likelihood_flag,
                                   output_folder=output_folder,
                                   dataset_name=dataset_name,
                                   select_fold=select_fold,
                                   target=target,
                                   n_mcmc_opt_its=N_iterations_MCMC_opt)
        pool.map(copier, range(Nfolds))

    plt.show()

if __name__ == '__main__':
    np.random.seed(42)
    main()
