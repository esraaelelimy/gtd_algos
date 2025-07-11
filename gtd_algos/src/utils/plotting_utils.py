import matplotlib as mpl
import numpy as np
from collections.abc import MutableMapping
import pandas as pd 
import wandb
api = wandb.Api()
import matplotlib.pyplot as plt
import json
import collections
from collections.abc import MutableMapping
import numpy as np
import pickle
import tqdm 
import os


class coloring():
    def __init__(self):
        self.t_ = 0
        self.color_list = ['#4477AA','#EE6677','#228833','#CCBB44','#66CCEE','#AA3377','#BBBBBB', '#EE3377', '#009988','#6699CC', '#EECC66', '#994455', '#997700', '#EE99AA',
                           '#3498db','#9b59b6','#95a5a6']
        # https://personal.sron.nl/~pault/
        self.bright = {'blue':'#4477AA', 'cyan': '#66CCEE', 'green': '#228833', 'yellow':'#CCBB44', 'red':'#EE6677', 'purple':'#AA3377', 'grey':'#BBBBBB'}
        self.high_contrast = {'yellow':'#DDAA33','red': '#BB5566', 'blue':'#004488'}
        self.vibrant = {'blue':'#0077BB', 'cyan':'#33BBEE', 'teal':'#009988', 'orange':'#EE3377', 'red':'#CC3311', 'magenta':'#EE3377', 'grey':'#BBBBBB'}
    def get_color(self,t=None):
        if t is not None:
            return self.color_list[t]    
        else:
            idx = self.t_ % len(self.color_list)
            self.t_ += 1
            return self.color_list[idx]
    def get_color_by_name(self,color,palette='bright'):
        return getattr(self, palette)[color]
        

    

def flatten(d, parent_key='', sep='__'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def confidence_interval(mean, stderr):
    return (mean - stderr, mean + stderr)

def confidence_interval_delta_05(mean, stderr):
    epsilon = stderr * 1.95
    return (mean - epsilon, mean + epsilon)

def compute_mean_stderr(data, n_runs):
    # data is a numpy array of shape (n_runs, n_steps)
    mean_metric = np.mean(data, axis=0)
    stderr_metric = np.std(data, axis=0) / np.sqrt(n_runs)
    return mean_metric, stderr_metric

def compute_mean_stderr_over_steps(data, n_runs):
    # data is a numpy array of shape (n_runs, n_steps)
    mean_over_steps = np.mean(data, axis=1)
    mean_metric = np.mean(mean_over_steps)
    stderr_metric = np.std(mean_over_steps, axis=0) / np.sqrt(n_runs)
    return mean_metric, stderr_metric

def plot_mean_and_ci(ax,plot_data,color,label,alpha=0.4,ci=confidence_interval, marker=None):
    '''
    mean_x: x-axis values
    mean_y: y-axis values
    stderr: standard error
    bin_size: bin size for averaging
    '''
    mean_x, mean_y, stderr = plot_data
    assert len(mean_x) == len(mean_y) == len(stderr)
    (cl,ch) = ci(mean_y, stderr)
    if marker is not None:
        ax.plot(mean_x,mean_y,color = color,label=label, marker = marker)    
    else:
        ax.plot(mean_x,mean_y,color = color,label=label)
    ax.fill_between(mean_x,
                cl,
                ch,
                alpha=alpha,color=color)

def boolean_indexing(v, fillval=np.nan):
    lens = np.array([len(item) for item in v])
    mask = lens[:,None] > np.arange(lens.max())
    out = np.full(mask.shape,fillval)
    out[mask] = np.concatenate(v)
    return out

def get_wandb_runs_w_keys(df,proj_name,run_keys,results_keys,diagnose=True):
    get_run = True
    results = {}
    for result_key in results_keys:
        results[result_key] = []

    i = 0
    for run in df[proj_name].values():
            target_cfg = flatten(run['config'])
            for k,v in run_keys.items():
                if k in target_cfg.keys():
                    if target_cfg[k] == v:
                        get_run = get_run and True
                    else:
                        get_run = get_run and False
                else:
                    get_run = False
            if get_run:
                if diagnose:
                    print(f"Run {i}, steps: {len(run['history']['env_steps'])}, return: {len(run['history']['undiscounted_return'])}") 
                for result_key in results_keys:
                    if result_key not in run['history'].keys():
                        print(f"Key {result_key} not found in run {i}")
                        get_run = False
                    else:
                        results[result_key].append(np.array(run['history'][result_key]))        
                i += 1
            else:
                get_run = True
    return results


def making_all_runs_same_length(env_steps, return_env, n_runs, diagnose=False,steps=None):
    '''
    env_steps: list of arrays
    return_env: list of arrays of the same length as env_steps
    n_runs: number of runs
    '''
    # TODO (Esraa): At the moment, if a run is not completetd yet, all other runs will be truncated to the length of the shortest run.
    # Ideally, we should also report which runs are not completed yet (and maybe just ignore the uncompleted ones)
    min_len = len(env_steps[0])
    truncated_env_steps = []
    truncated_return_env = []
     
    for i in range(n_runs):
        if len(env_steps[i]) < min_len:
            min_len = len(env_steps[i])
    if diagnose:
        print(f"Min length of runs: {min_len}")
    
    if steps is not None:
        for i in range(n_runs):
            if len(env_steps[i]) >= steps:
                if diagnose:
                    print(len(env_steps[i]))
                    print(len(env_steps[i][:steps])) 
                truncated_env_steps.append(env_steps[i][:steps])
                truncated_return_env.append(return_env[i][:steps])
    else:
        for i in range(n_runs):
            if diagnose:
                print(len(env_steps[i]))
                print(len(env_steps[i][:min_len])) 
            truncated_env_steps.append(env_steps[i][:min_len])
            truncated_return_env.append(return_env[i][:min_len])
        
    return np.array(truncated_env_steps), np.array(truncated_return_env)


def filter_non_completed_runs(results, results_keys, diagnose=False):
    max_length = 0
    for result_key in results_keys:
        for arr in results[result_key]:
            if len(arr) >= max_length:
                max_length = len(arr)
    if diagnose:
        print(f"Max length of runs: {max_length}")
    new_results = {}
    for result_key in results_keys:
        new_results[result_key] = []
        for arr in results[result_key]:
            if len(arr) == max_length:
                new_results[result_key].append(arr)
    return new_results
    
    
def filter_nan(values):
    return values[np.logical_not(np.isnan(values))]

def arange_steps(steps,mean):
    max_step = np.max(steps)
    steps = np.linspace(start=0, stop=max_step, num=len(mean))
    return steps
    
def get_results_from_wandb(df,proj_name,run_keys,results_keys,diagnose=False,get_individual_seeds=False):
    '''
    proj_name: name of the wandb project
    run_keys: dictionary with keys to filter runs
    results_keys: list of recorded results to extract from the runs
    '''
    results = get_wandb_runs_w_keys(df,proj_name,run_keys,results_keys,diagnose)
    
    all_runs_have_same_length = True
    for result_key in results_keys:
        all_runs_have_same_length = all_runs_have_same_length and all(arr.shape == results[result_key][0].shape for arr in results[result_key])
        if diagnose:
            print(arr.shape == results['env_steps'][0].shape for arr in results[result_key])
    if not all_runs_have_same_length:
        print("All runs should have the same length, Non completed runs detected.")
        results = filter_non_completed_runs(results, results_keys, diagnose)
        
        #env_steps, return_env = making_all_runs_same_length(results['env_steps'], results['undiscounted_return'], n_runs, diagnose,steps)
    
    if diagnose:
        for key in results.keys():
            print([arr.shape for arr in results[key]])
        for i in range(len(results['env_steps'])):
            print(f"env steps max: {np.max(results['env_steps'][i])}")
            print(f"env steps shape: {(results['env_steps'][i].shape)}")    
    n_runs = len(results['env_steps'])
    results_mean = {}
    for result_key in results_keys:
        if diagnose:
            print(result_key)
        mean, stderr = compute_mean_stderr(results[result_key], n_runs)
        ### wandb counts every call to log as a step, so if a metric is not logged for a step, it will be nan and we need to filter out the nans
        mean = filter_nan(mean)
        stderr = filter_nan(stderr)
        steps = arange_steps(results['env_steps'],mean)
        results_mean[result_key] = (steps, mean, stderr)
        if get_individual_seeds:
            results_mean[result_key] = (steps, mean, stderr, results[result_key])
            
    return results_mean, n_runs

    
def sliding_window_smoothing(data, window_size, step_size):
    '''
    data: numpy array
    window_size: size of the window to average over
    step_size: step size to move the window (step_size < window_size gives a sliding average, step_size = window_size gives a non-overlapping average)
    '''
    assert step_size >= 1
    assert window_size >= 1
    assert window_size <= len(data)
    raise NotImplementedError
    return 

def fixed_window_smoothing(data, window_size, padding= False):
    assert window_size >= 1
    assert data.shape[0] >= window_size
    if not data.shape[0] % window_size == 0: 
        if not padding:
            #print("Data length is not divisible by window size, and padding is not enabled: data will be truncated.")
            data = data[:-(data.shape[0] % window_size)]
    reshaped_data = data.reshape(len(data)//window_size, window_size)
    smoothed_data = np.mean(reshaped_data, axis=1)
    return smoothed_data
    

def plot_smoothed_results(ax, results,results_key,color,label,smoothing_methods=fixed_window_smoothing,window_size=10,alpha=0.4,ci=confidence_interval,marker=None):
    (steps, mean, stderr) = results[results_key]
    if smoothing_methods is None:
        plot_mean_and_ci(ax, (steps, mean, stderr),color,label)
    else:
        # TODO: needs to be more generic if other smoothing methods are added
        steps = smoothing_methods(steps, window_size)
        mean = smoothing_methods(mean, window_size)
        stderr = smoothing_methods(stderr, window_size)
        plot_mean_and_ci(ax, (steps, mean, stderr),color,label,alpha,ci,marker)
    

def plot_individual_seeds_and_mean(ax, results,results_key,color,label,smoothing_methods=fixed_window_smoothing,window_size=10,alpha=0.4):
    (steps, mean, _, runs) = results[results_key]
    steps = smoothing_methods(steps, window_size)
    for seed in range(len(runs)):
        run_data = runs[seed]
        run_data = smoothing_methods(run_data, window_size)
        ax.plot(steps,run_data,color = color,alpha=alpha)
    mean = smoothing_methods(mean, window_size)
    ax.plot(steps,mean,color = color, label = label)
    
    
def filter_non_completed_runs_based_on_env_steps(results, min_steps=4_500_000):
    filtered_results = {k: [] for k in results}
    for i in range(len(results['env_steps'])):
        final_step = results['env_steps'][i][-1]
        if final_step >= min_steps:
            for key in results:
                filtered_results[key].append(results[key][i])
    num_runs_used = len(filtered_results['env_steps'])
    return filtered_results, num_runs_used


def bin_and_average_runs(results, step_bins, value_key='undiscounted_return', min_valid=1, get_all_seeds=False):
    num_bins = len(step_bins) - 1
    num_runs = len(results['env_steps'])

    # Pre-allocate list of lists to collect per-bin values across runs
    binned_values_per_run = np.full((num_runs, num_bins), np.nan)

    for i in range(num_runs):
        steps = np.array(results['env_steps'][i])
        values = np.array(results[value_key][i])
        
        # Find bin indices for all steps in vectorized way
        bin_indices = np.searchsorted(step_bins, steps, side='right') - 1
        
        # Clip to valid range [0, num_bins - 1]
        valid_mask = (bin_indices >= 0) & (bin_indices < num_bins)
        bin_indices = bin_indices[valid_mask]
        values = values[valid_mask]

        # Aggregate by bin using efficient grouping
        for b in range(num_bins):
            bin_mask = (bin_indices == b)
            if np.any(bin_mask):
                binned_values_per_run[i, b] = np.mean(values[bin_mask])

    # Compute mean, stderr, and counts
    valid_counts = np.sum(~np.isnan(binned_values_per_run), axis=0)
    mean = np.nanmean(binned_values_per_run, axis=0)
    stderr = np.nanstd(binned_values_per_run, axis=0) / np.sqrt(np.maximum(valid_counts, 1))

    # Mask bins with insufficient runs
    mean[valid_counts < min_valid] = np.nan
    stderr[valid_counts < min_valid] = np.nan

    # Bin centers for plotting
    bin_centers = 0.5 * (step_bins[:-1] + step_bins[1:])
    if get_all_seeds:
        return bin_centers, mean, stderr, binned_values_per_run    
    return bin_centers, mean, stderr

def get_mean_stderr_results_from_wandb(df,proj_name,run_keys,
                                       min_steps,total_steps, window_size,
                                       results_keys,metric_key='undiscounted_return',diagnose=False,get_individual_seeds=False):
    '''
    proj_name: name of the wandb project
    run_keys: dictionary with keys to filter runs
    results_keys: list of recorded results to extract from the runs
    '''
    results = get_wandb_runs_w_keys(df,proj_name,run_keys,results_keys,diagnose)
    filtered_results, num_runs_used = filter_non_completed_runs_based_on_env_steps(results, min_steps)
    step_bins = np.arange(0, total_steps+1, window_size)
    if get_individual_seeds:
        bin_centers, mean_returns, stderr_returns,binned_values_per_run = bin_and_average_runs(filtered_results, step_bins, value_key=metric_key,get_all_seeds=True)
        return bin_centers, mean_returns, stderr_returns, num_runs_used, binned_values_per_run
    
    bin_centers, mean_returns, stderr_returns = bin_and_average_runs(filtered_results, step_bins, value_key=metric_key)
    return bin_centers, mean_returns, stderr_returns, num_runs_used

    
def fig_names(ax, x_label, y_label,title, font_size):
    ax.legend(fontsize=font_size,frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel(x_label, fontsize=font_size)
    ax.set_ylabel(y_label, fontsize=font_size, rotation=0, loc='top')
    ax.set_title(title, fontsize=font_size)
    