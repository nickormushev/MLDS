import pandas as pd
import numpy as np
import arviz as az
import pymc as pm
import matplotlib.pyplot as plt
from scipy.optimize import fmin_bfgs
from scipy.stats import gaussian_kde
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches

import seaborn as sns

from warnings import filterwarnings
filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool` is a deprecated alias')


def plot_scatter(samples_coeff_angle, samples_coeff_distance, sample_coeff_angle50, sample_coeff_distance50):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    sns.kdeplot(x=samples_coeff_angle, y=samples_coeff_distance, ax=axs[0], fill=True, color='blue')
    axs[0].set_title('All shots')
    axs[0].set_xlabel('Coeff_Angle')
    axs[0].set_ylabel('Coeff_Distance')
    axs[0].grid(True) 

    sns.kdeplot(x=sample_coeff_angle50, y=sample_coeff_distance50, ax=axs[1], fill=True, color='red')
    sns.kdeplot(x=samples_coeff_angle, y=samples_coeff_distance, ax=axs[1], fill=True, color='blue', alpha=0.8)
    axs[1].set_title('Subset of 50 shots')
    axs[1].set_xlabel('Coeff_Angle')
    axs[1].set_ylabel('Coeff_Distance')
    axs[1].grid(True) 

    red_patch = mpatches.Patch(color='red', label='Subset of 50 shots')
    blue_patch = mpatches.Patch(color='blue', label='All shots')
    axs[1].legend(handles=[red_patch, blue_patch])

    plt.tight_layout()
    plt.savefig('scatterplots.pdf')

df = pd.read_csv("./dataset.csv")

X = df.drop('Made', axis=1)
y = df['Made']

mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
X = (X - mean) / std

def run_model(X, y, start_sd=10):
    start_sd = 10
    with pm.Model() as model:
        intercept = pm.Normal("Intercept", 0, sigma=start_sd)
        coeff_angle = pm.Normal("Coeff_Angle", 0, sigma=start_sd)
        coeff_distance = pm.Normal("Coeff_Distance", 0, sigma=start_sd)

        p = pm.invlogit(intercept + coeff_angle*X.iloc[:,0] + coeff_distance*X.iloc[:,1])
        likelihood = pm.Binomial('y', n=len(y), p=p, observed=y)

        res = pm.sample(10000, chains=1)
    return res

# Usage
res = run_model(X, y)

pm.plot_posterior(res, var_names=['Coeff_Distance'])
plt.savefig("Coeff_Distance.pdf")

subset_indices = np.random.choice(range(len(X)), 50, replace=False)
res50 = run_model(X.iloc[subset_indices], y.iloc[subset_indices])

def log_likelihood(theta, X, y):
    intercept, coeff_angle, coeff_distance = theta
    p = 1 / (1 + np.exp(-(intercept + coeff_angle*X.iloc[:,0] + coeff_distance*X.iloc[:,1])))
    return np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)) / len(y)

def optimizer(parameters, *args):
    X = args[0]
    y = args[1]
    std = args[2]

    posterior = -(log_likelihood(parameters, X, y) \
        - np.sum(0.5 * (parameters/std)**2))

    return posterior

initial_values = np.array([0, 0.1, 0.1])
initial_std = 2
b_opt, _, _, InvHes, _, _, _ = fmin_bfgs(optimizer, x0=initial_values, args=(X, y, initial_std), full_output=True)

mean_intercept, mean_coeff_angle, mean_coeff_distance = b_opt
std_intercept, std_coeff_angle, std_coeff_distance = np.sqrt(InvHes.diagonal() / len(y)) 


samples_coeff_angle = res.posterior['Coeff_Angle'].values
samples_coeff_distance = res.posterior['Coeff_Distance'].values

diff = np.sum(np.abs(samples_coeff_distance) > np.abs(samples_coeff_angle))
print("Probability: {}".format(diff / len(samples_coeff_distance)))
diff = np.sum(samples_coeff_angle <= 0)
print("Probability: {}".format(diff / len(samples_coeff_angle)))

#samples_coeff_angle50 = res50.posterior['Coeff_Angle'].values
#samples_coeff_distance50 = res50.posterior['Coeff_Distance'].values
#
##plot_scatter(samples_coeff_angle[0], samples_coeff_distance[0], samples_coeff_angle50[0], samples_coeff_distance50[0])
#
#points = 3000
#samples_coeff_angle = np.random.normal(mean_coeff_angle, std_coeff_angle, points)
#samples_coeff_distance = np.random.normal(mean_coeff_distance, std_coeff_distance, points)
#
#print(samples_coeff_angle)
#
#xy = np.vstack([samples_coeff_angle, samples_coeff_distance])
#z = gaussian_kde(xy)(xy)
#
#fig, axs = plt.subplots(1, 2, figsize=(14, 8), subplot_kw={'projection': '3d'})
#
## Plot a 3D scatter plot for Laplace approximation
#axs[0].scatter(samples_coeff_angle, samples_coeff_distance, z, c=z)
#axs[0].set_title('Laplace approximation')
#axs[0].set_xlabel('Coeff_Angle')
#axs[0].set_ylabel('Coeff_Distance')
#axs[0].set_zlabel('Density')
#
#
## Generate samples for MCMC
#samples_coeff_angle = res.posterior['Coeff_Angle'].values[:,:points]
#samples_coeff_distance = res.posterior['Coeff_Distance'].values[:,:points]
#print(len(samples_coeff_angle))
#xy = np.vstack([samples_coeff_angle, samples_coeff_distance])
#z = gaussian_kde(xy[:points])(xy[:points])
#
## Plot a 3D scatter plot for MCMC
#axs[1].scatter(samples_coeff_angle[:, :points], samples_coeff_distance[:, :points], z, c=z)
#axs[1].set_title('MCMC')
#axs[1].set_xlabel('Coeff_Angle')
#axs[1].set_ylabel('Coeff_Distance')
#axs[1].set_zlabel('Density')
#
#
#plt.savefig('3d.pdf')
#plt.show()