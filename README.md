---
jupyter:
  interpreter:
    hash: edc2edf56ea13f1e22424e20afa6e48f5488415ce9d428406e2569475df17409
  kernelspec:
    display_name: Python 3.8.8 (\'base\')
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.8.8
  nbformat: 4
  nbformat_minor: 2
  orig_nbformat: 4
---

::: {.cell .markdown}
```{=html}
<h1 align="center">Examples of Double Descent</h1>
```
```{=html}
<hr style="border:2px solid gray">
```
:::

::: {.cell .markdown}
## \#\#\# Import python packages {#-import-python-packages}
:::

::: {.cell .code execution_count="3"}
``` {.python}
import numpy as np
import matplotlib.pyplot as plt

from numpy.random import default_rng
from sklearn.model_selection import LeaveOneOut, GridSearchCV
from sklearn.linear_model import LinearRegression
```
:::

::: {.cell .markdown}

------------------------------------------------------------------------

## \# Intuition behind double descent {#-intuition-behind-double-descent}
:::

::: {.cell .code execution_count="67"}
``` {.python}
import matplotlib.pyplot as plt
%matplotlib inline
plt.axis('off')
plt.arrow(0, 0, 1.0,  0.0, head_width=0.05, head_length=0.1, fc='k', ec='k'); plt.figtext(0.90, 0.25, r'$f_1$', fontsize=26)
plt.arrow(0, 0, 0.4,  0.6, head_width=0.05, head_length=0.1, fc='k', ec='k'); plt.figtext(0.50, 0.70, r'$f_2$', fontsize=26)
plt.arrow(0, 0, 0.9, -0.2, head_width=0.05, head_length=0.1, fc='tab:blue'  , ec='tab:blue'  ); plt.figtext(0.83, 0.10, r'$f_3$', fontsize=26)
plt.arrow(0, 0, 0.0,  0.8, head_width=0.05, head_length=0.1, fc='tab:blue'  , ec='tab:blue'  ); plt.figtext(0.10, 0.90, r'$f_4$', fontsize=26)
plt.arrow(0, 0, 0.7,  0.4, head_width=0.05, head_length=0.1, fc='tab:orange', ec='tab:orange'); plt.figtext(0.70, 0.55, r'$y$', fontsize=26)
plt.show()
```

::: {.output .display_data}
![](https://raw.githubusercontent.com/pbenner/double-descent/master/README_files/7073d3ea7bf231aa5215cb205aa9b641d5e1ab5b.png)
:::
:::

::: {.cell .markdown}
According to the above image, we define four feature vectors
![f_1 = (1, 0, 0)\^\\top](https://latex.codecogs.com/png.latex?f_1%20%3D%20%281%2C%200%2C%200%29%5E%5Ctop "f_1 = (1, 0, 0)^\top"),
![f_2 = (0, 1, 0)\^\\top](https://latex.codecogs.com/png.latex?f_2%20%3D%20%280%2C%201%2C%200%29%5E%5Ctop "f_2 = (0, 1, 0)^\top"),
![f_3 = (1, -0.2, 0)\^\\top](https://latex.codecogs.com/png.latex?f_3%20%3D%20%281%2C%20-0.2%2C%200%29%5E%5Ctop "f_3 = (1, -0.2, 0)^\top"),
and
![f_4 = (0, 0, 1)\^\\top](https://latex.codecogs.com/png.latex?f_4%20%3D%20%280%2C%200%2C%201%29%5E%5Ctop "f_4 = (0, 0, 1)^\top").
The response
![y = (1, 1, 0)\^\\top](https://latex.codecogs.com/png.latex?y%20%3D%20%281%2C%201%2C%200%29%5E%5Ctop "y = (1, 1, 0)^\top")
lies in the plane defined by the first two feature vectors
![f_1](https://latex.codecogs.com/png.latex?f_1 "f_1") and
![f_2](https://latex.codecogs.com/png.latex?f_2 "f_2").
:::

::: {.cell .code execution_count="80"}
``` {.python}
f1 = np.array([1.0,  0.0, 0.0])
f2 = np.array([0.0,  1.0, 0.0])
f3 = np.array([1.0, -0.2, 0.0])
f4 = np.array([0.0,  0.0, 1.0])
y  = np.array([1.0,  1.0, 0.0])
```
:::

::: {.cell .markdown}
We study the length of the OLS solution, which for
![n \> p](https://latex.codecogs.com/png.latex?n%20%3E%20p "n > p") is
given by the parameter vector with minimum
![\\ell_2](https://latex.codecogs.com/png.latex?%5Cell_2 "\ell_2")-norm.
:::

::: {.cell .code execution_count="76"}
``` {.python}
def OLSnorm(X, y):
    return np.linalg.norm(np.linalg.pinv(X.T@X)@X.T@y)
```
:::

::: {.cell .markdown}
As baseline, we begin with the OLS solution if we are only given
![f_1](https://latex.codecogs.com/png.latex?f_1 "f_1") and
![f_2](https://latex.codecogs.com/png.latex?f_2 "f_2"):
:::

::: {.cell .code execution_count="81"}
``` {.python}
OLSnorm(np.array([f1, f2]).T, y)
```

::: {.output .execute_result execution_count="81"}
    1.4142135623730951
:::
:::

::: {.cell .markdown}
The length of the solution decreases if we add another feature vector to
X:
:::

::: {.cell .code execution_count="82"}
``` {.python}
OLSnorm(np.array([f1, f2, f3]).T, y)
```

::: {.output .execute_result execution_count="82"}
    1.2985663286116427
:::
:::

::: {.cell .markdown}
However, this is only the case if the feature vector is correlated with
![y](https://latex.codecogs.com/png.latex?y "y"):
:::

::: {.cell .code execution_count="83"}
``` {.python}
OLSnorm(np.array([f1, f2, f4]).T, y)
```

::: {.output .execute_result execution_count="83"}
    1.4142135623730951
:::
:::

::: {.cell .markdown}

------------------------------------------------------------------------

## \#\#\# Data set {#-data-set}
:::

::: {.cell .code execution_count="4"}
``` {.python}
data = np.array([
    0.001399613, -0.23436656,
    0.971629779,  0.64689524,
    0.579119475, -0.92635765,
    0.335693937,  0.13000706,
    0.736736086, -0.89294863,
    0.492572335,  0.33854780,
    0.737133774, -1.24171910,
    0.563693769, -0.22523318,
    0.877603280, -0.12962722,
    0.141426545,  0.37632006,
    0.307203910,  0.30299077,
    0.024509308, -0.21162739,
    0.843665029, -0.76468719,
    0.771206067, -0.90455412,
    0.149670258,  0.77097952,
    0.359605608,  0.56466366,
    0.049612895,  0.18897607,
    0.409898906,  0.32531750,
    0.935457898, -0.78703491,
    0.149476207,  0.80585375,
    0.234315216,  0.62944986,
    0.455297119,  0.02353327,
    0.102696671,  0.27621694,
    0.715372314, -1.20379729,
    0.681745393, -0.83059624 ]).reshape(25,2)
y = data[:,1]
X = data[:,0:1]

plt.scatter(X[:,0], y)
plt.show()
```

::: {.output .display_data}
![](https://raw.githubusercontent.com/pbenner/double-descent/master/README_files/8cc01adbe18ab36d6896cea353ce1220b4f3fdc0.png)
:::
:::

::: {.cell .markdown}

------------------------------------------------------------------------

## \#\#\# Linear regressor class {#-linear-regressor-class}
:::

::: {.cell .markdown}
We use the following linear regressor class for all double descent
examples. It takes only the first
![p](https://latex.codecogs.com/png.latex?p "p") columns from the
feature matrix ![F](https://latex.codecogs.com/png.latex?F "F") and
computes the minimum
![\\ell_2](https://latex.codecogs.com/png.latex?%5Cell_2 "\ell_2")-norm
solution when
![n \< p](https://latex.codecogs.com/png.latex?n%20%3C%20p "n < p").
:::

::: {.cell .code execution_count="3"}
``` {.python}
class MyRidgeRegressor:
    def __init__(self, p=3, alpha=0.0):
        self.p     = p
        self.theta = None
        self.alpha = alpha
    
    def fit(self, F, y):
        F = F[:, 0:self.p]
        self.theta = np.linalg.pinv(F.transpose()@F + self.alpha*np.identity(F.shape[1]))@F.transpose()@y

    def predict(self, F):
        F = F[:, 0:self.p]
        return F@self.theta

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep=True):
        return {"p" : self.p, "alpha" : self.alpha}
```
:::

::: {.cell .markdown}

------------------------------------------------------------------------

## \#\#\# Model evaluation {#-model-evaluation}
:::

::: {.cell .code execution_count="4"}
``` {.python}
def evaluate_model(fg, X, y, n, ps, runs=10):
    estimator = MyRidgeRegressor()
    result = None
    for i in range(runs):
        F, y = fg(X, y, n, np.max(ps), random_state=i)

        clf = GridSearchCV(estimator=estimator,
                            param_grid=[{ 'p': list(ps) }],
                            cv=LeaveOneOut(),
                            scoring="neg_mean_squared_error")
        clf.fit(F, y)

        if result is None:
            result  = -clf.cv_results_['mean_test_score']
        else:
            result += -clf.cv_results_['mean_test_score']
    
    return result / runs
```
:::

::: {.cell .markdown}

------------------------------------------------------------------------

## \# 1 Random features {#-1-random-features}
:::

::: {.cell .markdown}
This example of double descent uses covariates generated from a
multivariate normal distribution, i.e. the
![j](https://latex.codecogs.com/png.latex?j "j")-th column of X is given
by

![
    f_j \~ \\sim \~ N(0, I_n)
](https://latex.codecogs.com/png.latex?%0A%20%20%20%20f_j%20~%20%5Csim%20~%20N%280%2C%20I_n%29%0A "
    f_j ~ \sim ~ N(0, I_n)
")

In order to obtain a double descent phenomenon, we need that some
![f_j](https://latex.codecogs.com/png.latex?f_j "f_j") are highly
correlated with ![y](https://latex.codecogs.com/png.latex?y "y") and the
remaining features are uncorrelated. Hence, we define
![\\theta_j = 1/j](https://latex.codecogs.com/png.latex?%5Ctheta_j%20%3D%201%2Fj "\theta_j = 1/j")
and generate observations
![y](https://latex.codecogs.com/png.latex?y "y") according to the linear
model

![
    y = X \\theta + \\epsilon
](https://latex.codecogs.com/png.latex?%0A%20%20%20%20y%20%3D%20X%20%5Ctheta%20%2B%20%5Cepsilon%0A "
    y = X \theta + \epsilon
")

where
![\\epsilon \\sim N(0, \\sigma\^2 I_n)](https://latex.codecogs.com/png.latex?%5Cepsilon%20%5Csim%20N%280%2C%20%5Csigma%5E2%20I_n%29 "\epsilon \sim N(0, \sigma^2 I_n)").
:::

::: {.cell .code execution_count="22"}
``` {.python}
class RandomFeatures():
    def __init__(self, scale = 1):
        self.scale = scale

    def __call__(self, X, y, n, p, random_state=42):
        rng = default_rng(seed=random_state)
        mu = np.repeat(0, n)
        sigma = np.identity(n)
        F = rng.multivariate_normal(mu, sigma, size=p).T
        theta = np.array([ 1/(j+1) for j in range(p) ])
        y = F@theta + rng.normal(0, self.scale, size=n)
        return F, y
```
:::

::: {.cell .markdown}

------------------------------------------------------------------------

## \#\#\# Results {#-results}
:::

::: {.cell .code execution_count="23"}
``` {.python}
ps = range(3, 151)
scales = [0.1, 1, 10]
result = [ evaluate_model(RandomFeatures(scale=scale), None, None, 20, ps, runs=100) for scale in scales ]
result = np.array(result)
```
:::

::: {.cell .code execution_count="31"}
``` {.python}
p = plt.plot(ps, result.T)
[ plt.axvline(x=ps[np.argmin(result[i])], color=p[i].get_color(), alpha=0.5, linestyle='--') for i in range(result.shape[0]) ]
plt.legend(scales)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("p")
plt.ylabel("average mse")
plt.show()
```

::: {.output .display_data}
![](https://raw.githubusercontent.com/pbenner/double-descent/master/README_files/30caccee17aa08ef0dc22a7499e15f28c324c949.png)
:::
:::

::: {.cell .markdown}

------------------------------------------------------------------------

## \# 2 Noisy polynomial features {#-2-noisy-polynomial-features}
:::

::: {.cell .markdown}
Increasing the number of features not always leads to an increase in
model complexity. There are several cases where adding more features
actually constraints the model, which we call here *implicit
regularization*. For instance, adding features to
![F](https://latex.codecogs.com/png.latex?F "F") that are uncorrelated
with ![y](https://latex.codecogs.com/png.latex?y "y") will generally
lead to a stronger regularization (e.g. when adding columns to
![F](https://latex.codecogs.com/png.latex?F "F") that are drawn from a
normal distribution). Here, we test a different strategy to increase
implicit regularization. We implement a function called
*compute_noisy_polynomial_features* that computes noisy polynomial
features ![F](https://latex.codecogs.com/png.latex?F "F") from
![X = (x)](https://latex.codecogs.com/png.latex?X%20%3D%20%28x%29 "X = (x)").
The ![j](https://latex.codecogs.com/png.latex?j "j")-th column of
![F \\in \\mathbb{R}\^{n \\times p}](https://latex.codecogs.com/png.latex?F%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bn%20%5Ctimes%20p%7D "F \in \mathbb{R}^{n \times p}")
is given by

![
    f_j
    =
    \\begin{cases}
        (1, \\dots, 1)\^\\top & \\text{if \$j = 1\$}\\\\
        x\^{k(j-2)} + \\epsilon_j & \\text{if \$j \> 1\$}
    \\end{cases}
](https://latex.codecogs.com/png.latex?%0A%20%20%20%20f_j%0A%20%20%20%20%3D%0A%20%20%20%20%5Cbegin%7Bcases%7D%0A%20%20%20%20%20%20%20%20%281%2C%20%5Cdots%2C%201%29%5E%5Ctop%20%26%20%5Ctext%7Bif%20%24j%20%3D%201%24%7D%5C%5C%0A%20%20%20%20%20%20%20%20x%5E%7Bk%28j-2%29%7D%20%2B%20%5Cepsilon_j%20%26%20%5Ctext%7Bif%20%24j%20%3E%201%24%7D%0A%20%20%20%20%5Cend%7Bcases%7D%0A "
    f_j
    =
    \begin{cases}
        (1, \dots, 1)^\top & \text{if $j = 1$}\\
        x^{k(j-2)} + \epsilon_j & \text{if $j > 1$}
    \end{cases}
")

where
![k(j) = (j\\mod m) + 1](https://latex.codecogs.com/png.latex?k%28j%29%20%3D%20%28j%5Cmod%20m%29%20%2B%201 "k(j) = (j\mod m) + 1"),
![m \\le p](https://latex.codecogs.com/png.latex?m%20%5Cle%20p "m \le p")
denotes the maximum degree (*max_degree* parameter) and
![\\epsilon_j](https://latex.codecogs.com/png.latex?%5Cepsilon_j "\epsilon_j")
is a vector of ![n](https://latex.codecogs.com/png.latex?n "n")
independent draws from a normal distribution with mean
![\\mu = 0](https://latex.codecogs.com/png.latex?%5Cmu%20%3D%200 "\mu = 0")
and standard deviation
![\\sigma](https://latex.codecogs.com/png.latex?%5Csigma "\sigma"). With
![x\^k](https://latex.codecogs.com/png.latex?x%5Ek "x^k") we denote the
![k](https://latex.codecogs.com/png.latex?k "k")-th power of each
element in ![x](https://latex.codecogs.com/png.latex?x "x").
:::

::: {.cell .code execution_count="26"}
``` {.python}
class NoisyPolynomialFeatures():
    def __init__(self, max_degree = 15, scale = 0.1):
        self.max_degree = max_degree
        self.scale      = scale

    def __call__(self, X, y, n, p, random_state=42):
        x = X if len(X.shape) == 1 else X[:,0]
        rng = default_rng(seed=random_state)
        F = np.array([]).reshape(x.shape[0], 0)
        F = np.insert(F, 0, np.repeat(1, len(x)), axis=1)
        for k in range(p):
            d = (k % self.max_degree)+1
            f = x**d + rng.normal(size=len(x), scale=self.scale)
            F = np.insert(F, k+1, f, axis=1)
        return F, y
```
:::

::: {.cell .markdown}

------------------------------------------------------------------------

## \#\#\# Results {#-results}
:::

::: {.cell .markdown}
Evaluate the performance for
![p = 3, 4, \\dots, 200](https://latex.codecogs.com/png.latex?p%20%3D%203%2C%204%2C%20%5Cdots%2C%20200 "p = 3, 4, \dots, 200"),
and
![\\sigma \\in \\{0.01, 0.02, 0.05\\}](https://latex.codecogs.com/png.latex?%5Csigma%20%5Cin%20%5C%7B0.01%2C%200.02%2C%200.05%5C%7D "\sigma \in \{0.01, 0.02, 0.05\}").
:::

::: {.cell .code execution_count="397"}
``` {.python}
ps = range(3, 201)
scales = [0.01, 0.02, 0.05]
result = [ evaluate_model(NoisyPolynomialFeatures(scale=scale), X, y, len(y), ps, runs=100) for scale in scales ]
result = np.array(result)
```
:::

::: {.cell .code execution_count="406"}
``` {.python}
p = plt.plot(ps, result.T)
[ plt.axvline(x=ps[np.argmin(result[i])], color=p[i].get_color(), alpha=0.5, linestyle='--') for i in range(result.shape[0]) ]
plt.legend(scales)
plt.xscale("log")
plt.xlabel("p")
plt.yscale("log")
plt.ylim(0.1,10)
plt.ylabel("average mse")
plt.show()
```

::: {.output .display_data}
![](https://raw.githubusercontent.com/pbenner/double-descent/master/README_files/a7e914acc29f369cf6bbbfaaa4e246dd88ea744a.png)
:::
:::

::: {.cell .markdown}

------------------------------------------------------------------------

## \# 3 Polynomial features combined with random features {#-3-polynomial-features-combined-with-random-features}
:::

::: {.cell .markdown}
Another possibility to obtain double descent curves is to start off with
a standard polynomial regression task and to add random (uncorrelated)
features. The ![j](https://latex.codecogs.com/png.latex?j "j")-th column
of
![F \\in \\mathbb{R}\^{n \\times p}](https://latex.codecogs.com/png.latex?F%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bn%20%5Ctimes%20p%7D "F \in \mathbb{R}^{n \times p}")
is given by

![
    f_j
    =
    \\begin{cases}
        x\^{j-1} & \\text{if \$j \\le m+1\$}\\\\
        \\epsilon_j & \\text{if \$j \> m+1\$}
    \\end{cases}
](https://latex.codecogs.com/png.latex?%0A%20%20%20%20f_j%0A%20%20%20%20%3D%0A%20%20%20%20%5Cbegin%7Bcases%7D%0A%20%20%20%20%20%20%20%20x%5E%7Bj-1%7D%20%26%20%5Ctext%7Bif%20%24j%20%5Cle%20m%2B1%24%7D%5C%5C%0A%20%20%20%20%20%20%20%20%5Cepsilon_j%20%26%20%5Ctext%7Bif%20%24j%20%3E%20m%2B1%24%7D%0A%20%20%20%20%5Cend%7Bcases%7D%0A "
    f_j
    =
    \begin{cases}
        x^{j-1} & \text{if $j \le m+1$}\\
        \epsilon_j & \text{if $j > m+1$}
    \end{cases}
")

where ![m](https://latex.codecogs.com/png.latex?m "m") denotes the
maximum degree of the polynomial features and
![\\epsilon_j \~ \\sim \~ N(0, \\sigma\^2 I_n)](https://latex.codecogs.com/png.latex?%5Cepsilon_j%20~%20%5Csim%20~%20N%280%2C%20%5Csigma%5E2%20I_n%29 "\epsilon_j ~ \sim ~ N(0, \sigma^2 I_n)").
:::

::: {.cell .code execution_count="31"}
``` {.python}
class PolynomialWithRandomFeatures():
    def __init__(self, max_degree = 15, scale = 0.1):
        self.max_degree = max_degree
        self.scale      = scale

    def __call__(self, X, y, n, p, random_state=42):
        x = X if len(X.shape) == 1 else X[:,0]
        rng = default_rng(seed=random_state)
        F = np.array([]).reshape(x.shape[0], 0)
        # Generate polynomial features
        for deg in range(np.min([p, self.max_degree+1])):
            F = np.insert(F, deg, x**deg, axis=1)
        if p <= self.max_degree+1:
            return F, y
        # Generate random features
        for j in range(p - self.max_degree - 1):
            f = rng.normal(size=F.shape[0], scale=self.scale)
            F = np.insert(F, F.shape[1], f, axis=1)
        return F, y
```
:::

::: {.cell .markdown}

------------------------------------------------------------------------

## \#\#\# Results {#-results}
:::

::: {.cell .code execution_count="61"}
``` {.python}
ps = list(range(3, 20)) + [30, 50, 100, 150, 200, 300, 400, 500, 1000]
scales = [0.01, 0.1, 1.0]
result = [ evaluate_model(PolynomialWithRandomFeatures(scale=scale), X, y, len(y), ps, runs=100) for scale in scales ]
result = np.array(result)
```
:::

::: {.cell .code execution_count="62"}
``` {.python}
p = plt.plot(ps, result.T)
[ plt.axvline(x=ps[np.argmin(result[i])], color=p[i].get_color(), alpha=0.5, linestyle='--') for i in range(result.shape[0]) ]
plt.legend(scales)
plt.xscale("log")
plt.xlabel("p")
plt.yscale("log")
#plt.ylim(0.1,10)
plt.ylabel("average mse")
plt.show()
```

::: {.output .display_data}
![](https://raw.githubusercontent.com/pbenner/double-descent/master/README_files/4f8963836a2352107cc7e0ffaa89f18e29527ebe.png)
:::
:::

::: {.cell .markdown}

------------------------------------------------------------------------

## \# 4 Legendre polynomial {#-4-legendre-polynomial}
:::

::: {.cell .code execution_count="6"}
``` {.python}
class LegendrePolynomialFeatures():
    def __call__(self, X, y, n, p, random_state=42):
        x = X if len(X.shape) == 1 else X[:,0]
        F = np.array([]).reshape(x.shape[0], 0)
        # Generate polynomial features
        for deg in range(p):
            l = np.polynomial.legendre.Legendre([0]*deg + [1], domain=[0,1])
            F = np.insert(F, deg, l(x), axis=1)
        return F, y
```
:::

::: {.cell .markdown}

------------------------------------------------------------------------

## \#\#\# Demonstration {#-demonstration}
:::

::: {.cell .code execution_count="101"}
``` {.python}
F, _ = LegendrePolynomialFeatures()(X, y, len(y), 1000)
g = np.linspace(0, 1, 10000)
G, _ = LegendrePolynomialFeatures()(g, y, len(y), 1000)
```
:::

::: {.cell .code execution_count="107"}
``` {.python}
clf = MyRidgeRegressor(p=8)
clf.fit(F, y)
plt.plot(g, clf.predict(G))
plt.scatter(X, y)
plt.title("p = 8")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
```

::: {.output .display_data}
![](https://raw.githubusercontent.com/pbenner/double-descent/master/README_files/15d9abf8bea424141547921f56c959835df26df8.png)
:::
:::

::: {.cell .code execution_count="114"}
``` {.python}
clf = MyRidgeRegressor(p=50)
clf.fit(F, y)
plt.plot(g, clf.predict(G))
plt.scatter(X, y)
plt.title("p = 100")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
```

::: {.output .display_data}
![](https://raw.githubusercontent.com/pbenner/double-descent/master/README_files/845a7b46fa2ef8f5c4694544ed26e772dc1c770c.png)
:::
:::

::: {.cell .code execution_count="108"}
``` {.python}
clf = MyRidgeRegressor(p=1000)
clf.fit(F, y)
plt.plot(g, clf.predict(G))
plt.scatter(X, y)
plt.title("p = 1000")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
```

::: {.output .display_data}
![](https://raw.githubusercontent.com/pbenner/double-descent/master/README_files/aab1daf89cfcc9db85e1405a643d0626ab3c4fb7.png)
:::
:::

::: {.cell .markdown}

------------------------------------------------------------------------

## \#\#\# Results {#-results}
:::

::: {.cell .code execution_count="115"}
``` {.python}
ps = list(range(3, 20)) + [30, 50, 100, 150, 200, 300, 400, 500, 1000]
result = evaluate_model(LegendrePolynomialFeatures(), X, y, len(y), ps, runs=1)
```
:::

::: {.cell .code execution_count="119"}
``` {.python}
p = plt.plot(ps, result)
plt.axvline(x=ps[np.argmin(result)], color=p[0].get_color(), alpha=0.5, linestyle='--')
plt.legend(scales)
plt.xscale("log")
plt.xlabel("p")
plt.yscale("log")
plt.ylim(0.1,10)
plt.ylabel("mse")
plt.show()
```

::: {.output .display_data}
![](https://raw.githubusercontent.com/pbenner/double-descent/master/README_files/66e43fdad7cb52784d8273bce359a220bb6c30b0.png)
:::
:::

::: {.cell .markdown}

------------------------------------------------------------------------

## \#\#\# Correlation analysis {#-correlation-analysis}
:::

::: {.cell .markdown}
To understand why we are seeing a double descent curve for Legendre
polynomials, we compute the correlation between features
![f_j](https://latex.codecogs.com/png.latex?f_j "f_j") and response
![y](https://latex.codecogs.com/png.latex?y "y"):
:::

::: {.cell .code execution_count="10"}
``` {.python}
cor = []
for i in range(F.shape[1]):
    cor.append(np.abs(np.correlate(F[:,i], y)))
```
:::

::: {.cell .markdown}
A plot of the result shows that the correlation decreases exponentially,
whereby we are essentially adding noise to the feature matrix
![F](https://latex.codecogs.com/png.latex?F "F").
:::

::: {.cell .code execution_count="38"}
``` {.python}
cor_x = np.log(list(range(1, len(cor)+1)))
cor_x = np.array(cor_x).reshape(-1, 1)
cor_y = np.log(cor)
cor_z = np.linspace(1, len(cor), 100).reshape(-1, 1)

clf = LinearRegression()
clf.fit(cor_x, cor_y)

plt.plot(cor)
plt.plot(cor_z, np.exp(clf.predict(np.log(cor_z))))
plt.xscale('log')
plt.xlabel('degree')
plt.yscale('log')
plt.ylabel('absolute correlation')
plt.show()
```

::: {.output .display_data}
![](https://raw.githubusercontent.com/pbenner/double-descent/master/README_files/28a48289a3cd6b3edbf6f113cff9c4abda3d6883.png)
:::
:::

::: {.cell .code}
``` {.python}
```
:::
