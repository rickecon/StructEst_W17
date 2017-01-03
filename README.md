# MACS 40200: Structural Estimation (Winter 2017) #

|  | [Dr. Richard Evans](https://sites.google.com/site/rickecon/) |
|--------------|--------------------------------------------------------------|
| Email | rwevans@uchicago.edu |
| Office | 250 Saieh Hall |
| Office Hours | W 2:30-4:30pm |
| GitHub | [rickecon](https://github.com/rickecon) |

* **Meeting day/time**: T,Th 12:00-1:20pm, Saieh Hall, Room 247
* Office hours also available by appointment

## Prerequisites ##

Advanced undergraduate or first-year graduate microeconomic theory, statistics, linear algebra, multivariable calculus, recommended coding experience.


## Recommended Texts (not required) ##

* Davidson, Russell and James G. MacKinnon, _Econometric Theory and Methods_, Oxford University Press (2004).
* Hansen, Lars Peter and Thomas J. Sargent, _Robustness_, Princeton University Press (2008).
* Scott, David W., _Multivariate Density Estimation: Theory, Practice, and Visualization_, 2nd edition, John Wiley & Sons (2015).
* Wolpin, Kenneth I., The Limits of Inference without Theory, MIT Press (2013).


## Course description ##

The purpose of this course is to give students experience estimating parameters of structural models. We will define the respective differences, strengths, and weaknesses of structural modeling and estimation versus reduced form modeling and estimation. We will focus on structural estimation. Methods will include taking parameters from other studies (weak calibration), estimating parameters to match moments from the data (GMM, strong calibration), simulating the model to match moments from the data (SMM, indirect inference), maximum likelihood estimation of parameters, and questions of model uncertainty and robustness. We will focus on both obtaining point estimates as well as getting an estimate of the variance-covariance matrix of the point estimates.

Most of the examples in the course will come from economics, but the material will be presented in a general way in order to allow students to apply the methods to estimating structural model parameters in any field. We will focus on computing solutions to estimation problems. Students can use whatever programming language they want, but I highly recommend you use Python 3.x ([Anaconda distribution](https://www.continuum.io/downloads)). I will be most helpful with code debugging and suggestions in Python. We will also study results and uses from recent papers listed in the "References" section below. The dates on which we will be covering those references are listed in the "Daily Course Outline" section below.


## Course Objectives and Learning Outcomes ##

* You will learn the difference between and the strengths and weaknesses of:
	* Structural vs. reduced form models
	* Linear vs. nonlinear models
	* Deterministic vs. stochastic models
	* Parametric vs. nonparametric models
* You will learn multiple ways to estimate parameters of structural models.
	* Calibration
	* Maximum likelihood estimation
	* Generalized method of moments
	* Simulated method of moments
* You will learn how to compute the variance-covariance matrix for your estimates.
* You will learn coding and collaboration techniques such as:
	* Best practices for Python coding ([PEP 8](https://www.python.org/dev/peps/pep-0008/))
	* Writing modular code with functions and objects
	* Creating clear docstrings for functions
	* Collaboration tools for writing code using [Git](https://git-scm.com/) and [GitHub.com](https://github.com/).


## Grades ##

Grades will be based on the four categories listed below with the corresponding weights.

Assignment   | Points | Percent |
-------------|--------|---------|
Problem Sets |    50  |   62.5% |
Project      |    10  |   12.5% |
Final Exam   |    20  |   25.0% |
**Total points** | **80** | **100%** |

* **Homework:** I will assign 6 problem sets throughout the term, and I will drop your one lowest problem set score.
	* You must write and submit your own computer code, although I encourage you to collaborate with your fellow students. I **DO NOT** want to see a bunch of copies of identical code. I **DO** want to see each of you learning how to code these problems so that you could do it on your own.
	* Problem set solutions, both written and code portions, will be turned in via a pull request from your private [GitHub.com](https://git-scm.com/) repository which is a fork of the class master repository on my account. (You will need to set up a GitHub account if you do not already have one.)
	* Problem sets will be due on the day listed in the Daily Course Outline section of this syllabus (see below) unless otherwise specified. Late homework will not be graded.
* **Project:** The project will be a replication of an estimation study that I assign. It will be worth 10 point, which is equivalent to one homework assignment. But the project scor cannot be dropped.
* **Final Exam:** The final exam will be given on Thursday, March 16, from 10:30am to 12:30pm in our classroom (Saieh 247). The final exam will be a comprehensive, in-class, real-time computational exercise. It will be easier than any of the problem sets you completed. Its purpose it to make sure that each of you can do the operations of structural estimation on your own, in contrast to your problem sets on which I encourage you to work together. The final exam is worth 20 points, which is equivalent to 2 problem sets. The final exam grade cannot be dropped.


## Daily Course Schedule ##

|  Date   | Day|            Topic            | Readings | Homework |
|---------|----|-------------------------------------|-------|-----|
| Jan.  3 |  T | Introduction                        |       |     |
| Jan.  5 | Th | Structural vs. reduced form disc.   | K2010 | PS1 |
|         |    |                                     | R2010 |     |
| Jan. 10 |  T | Maximum likelihood estimation (ML)  | Notes |     |
| Jan. 12 | Th |                                     |       |     |
| Jan. 17 |  T |                                     |       | PS2 |
| Jan. 19 | Th |                                     |       |     |
| Jan. 24 |  T | Generalized method of moments (GMM) | Notes | PS3 |
| Jan. 26 | Th |                                     | H1982 |     |
| Jan. 31 |  T |                                     |       | PS4 |
| Feb.  2 | Th |                                     |       |     |
| Feb.  7 |  T | Compare ML and GMM                | FMS1995 | PS5 |
| Feb.  9 | Th | Simulated Method of Moments (SMM) | Notes   |     |
| Feb. 14 |  T |                                   | S2008   |     |
| Feb. 16 | Th |                                   | DRM2004, |    |
| Feb. 21 |  T | SMM papers                        | Chap. 9.6 |   |
|         |    |                                   | ASV2013 |     |
| Feb. 23 | Th | Nonlinear Estimation Project      |         | PS6 |
| Feb. 28 |  T |                                   |         |     |
| Mar.  2 | Th |                                   |         |     |
| Mar.  7 |  T | Review, Advances in structural est. |   | Project |
|         |    | *Exam preparation (reading) days, Mar. 9-10* |  | |
| **Mar. 16** | **Th** | **Final Exam (comprehensive)** |  | Final |
|         |     | **10:30a.m.-12:30p.m. in Saieh 247** |     |     |


## References ##

* Altonji, Joseph G., Anthony A. Smith, Jr., and Ivan Vidangos, "Modeling Earnings Dynamics," *Econometrica*, 84:4, pp. 1395-1454 (July 2013)
* Davidson, Russell and James G. MacKinnon, "Section 9.6: The Method of Simulated Moments," *Econometric Theory and Methods*, Oxford University Press, (2004).
* Fuhrer, Jeffrey C. and George R. Moore, and Scott D. Schuh, "Estimating the Linear-quadratic Inventory Model: Maximum Likelihood versus Generalized Method of Moments," *Journal of Monetary Economics*, 35:1, pp. 115-157 (Feb. 1995).
* Hansen, Lars Peter, "Large Sample Properties of Generalized Method of Moments Estimators," *Econometrica*, 50:4, pp.1029-1054 (July 1982).
* Keane, Michael P., "Structural vs. Atheoretic Approaches to Econometrics," *Journal of Econometrics*, 156:1, pp. 3-20 (May 2010).
* Rust, John, "Comments on: 'Structural vs. Atheoretic Approaches to Econometrics' by Michael Keane," *Journal of Econometrics*, 156:1, pp. 21-24 (May 2010).
* Smith, Anthony A. Jr., "[Indirect Inference](http://www.econ.yale.edu/smith/palgrave7.pdf)," *New Palgrave Dictionary of Economics*, 2nd edition, (2008).


## Disability services ##

If you need any special accommodations, please provide us with a copy of your Accommodation Determination Letter (provided to you by the Student Disability Services office) as soon as possible so that you may discuss with me how your accommodations may be implemented in this course.
