#!/usr/bin/env python
# coding: utf-8

# # Assignment 8
# 
# Plotting with `matplotlib`
# 
# ## Problem 1
# 
# Use the data in [poro_perm.csv](poro_perm.csv) to reproduce the following plot with `matplotlib` in Python.
# 
# <img src="./images/poro_perm.png" width=700>
# 
# Since you've already developed fitting routines in [Assignment 7](https://github.com/PGE323M-Students/assignment7/) you should use them to perform the analysis on the data.  To avoid having to reproduce or copy the code from Assignment 7, you can load the class directly.  First, from the Terminal command line, run the following command to convert the Jupyter notebook into a regular Python file
# 
# ```bash
# jupyter nbconvert assignment7.ipynb --to python
# ```
# 
# then move the newly created `assignment7.py` into this repository, i.e. the same location as `assignment8.ipynb` and execute the following line in this notebook
# 
# ```python
# from assignment7 import KozenyCarmen
# ```
# 
# This will load the `KozenyCarmen` class directly into the namespace of the notebook and it will be available for use.  If you use this approach, don't forget to add `assignment7.py` to this repository when you commit your final solution for testing.
# 
# Please note that the plot must be **exactly the same** in order for the tests to pass, so take care to note the small details.  Here are a couple of tips:
# 
#  * For plotting the fit lines, use a Numpy `linspace` that goes from 0 to 0.012 with 50 points.
#  
#  * The $\LaTeX$ source for the $x$-axis label is `\frac{\phi^3}{(1-\phi)^2}`.  It shouldn't be too difficult for you to figure out the $y$-axis label.

# In[8]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from assignment7 import KozenyCarmen


# In[30]:


def kozeny_carmen_plot(filename, **kwargs):   
    kc_model = KozenyCarmen(filename)

    kappa0, m = kc_model.fit()
    m_zero = kc_model.fit_through_zero()[0]

    phi = kc_model.df['porosity'].values
    k = kc_model.df['permeability'].values

    x_values = (phi ** 3) / ((1 - phi) ** 2)

    def fit(phi, kappa0, m):
        return kappa0 + m * x_values
    
    def fit_through_zero(phi, m):
        return m * x_values
    
    x_fit_transformed = x_values
    y_fit = fit(x_fit_transformed, kappa0, m)
    y_fit_zero = fit_through_zero(x_fit_transformed, m_zero)
    
    fig, ax = plt.subplots(**kwargs)
    
    ax.scatter(x_values, k, color='black', label='Data')
    ax.plot(x_fit_transformed, y_fit, color='red', label='Fit', linewidth=2)
    ax.plot(x_fit_transformed, y_fit_zero, color='blue', label='Fit Through Zero', linewidth=2)

    ax.set_xlabel(r'$\frac{\phi^3}{(1-\phi)^2}$')
    ax.set_ylabel(r'Permeability (mD)')

    ax.legend()
    # ax.set_title('Kozeny-Carmen Fit')

    plt.show()
    return fig


# In[31]:


kozeny_carmen_plot('poro_perm.csv')


# ## Problem 2
# 
# Complete the function below to create the following contour plot.
# 
# <img src='./images/Nechelik.png' width=800>
# 
# Read in the [Nechelik.dat](Nechelik.dat) file which contains actual, estimated porosity of a field at equally spaced $x$,$y$ positions in the reservoir. Note that there are $54$ grid blocks/porosity values in the $x$ direction and $44$ in the $y$ direction i.e. you need a $44 \times 54$ porosity matrix. Each grid block is a square with sides $130.75$ ft.
# 
# As in Problem 1, the plot must be **exactly the same** for the tests to pass.  Refer to the tips above, and be sure to set the aspect ratio of the plot to `'equal'`.

# In[46]:


def contour_plot(filename, **kwargs):
    
    data = np.loadtxt(filename)

    x_blocks = 54
    y_blocks = 44

    porosity_matrix = data.reshape(y_blocks, x_blocks)
    #porosity_matrix = porosity_matrix.T
    
    x = np.linspace(0, x_blocks * 130.75, x_blocks)
    y = np.linspace(0, y_blocks * 130.75, y_blocks)

    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots(**kwargs)
    
    c = ax.contourf(X, Y, porosity_matrix, cmap='viridis')
    
    ax.set_aspect('equal')
    ax.invert_yaxis()

    fig.colorbar(c, ax=ax)

    # ax.set_xlabel('x (ft)')
    # ax.set_ylabel('y (ft)')

    ax.set_title('Porosity')

    plt.show()
    return fig


# In[47]:


contour_plot('Nechelik.dat')


# In[ ]:




