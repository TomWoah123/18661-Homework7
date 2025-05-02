<div align="center">
<h1>18661 Homework 7: PCA and Reinforcement Learning</h1>

</div>
<div align="center">
<b>Timothy Wu</b><sup>1*</sup>
<br>
</div>
<div align="center">
<sup>1</sup>Carnegie Mellon University
</div>

## PCA
The script `pca.py` produces the eigenfaces from the `face_data.mat` file using Principal Component Analysis. The data matrix gets zero-centered to calculate the co-variance matrix $C_x = \frac{1}{n} X^T X$. The eigenvectors and their
corresponding eigenvalues of the co-variance matrix get sorted based on their eigenvalue. The top $k$ eigenvectors are the principal components used for dimensionality reduction.

## Reinforcement Learning
The script `VI_SARSA.py` simulates an agent traversing through a Frozen Lake environment. The agent learns an optimal policy using value iteration, policy iteration, Q-learning, and SARSA algorithms.
