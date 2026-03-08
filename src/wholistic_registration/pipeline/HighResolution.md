## WHOLISTIC Registration Pipeline(High resolution)

# Remodel
Instead of searching for a motion field to warp the moving image, we could find a mapping from moving image to reference image. We could regard the reference as world coordinates and regard it as a dense space by interpolating. We could know the intensity of each point even if it's not on the grid.

$$
\begin{aligned}
\phi: \Omega_{move} &\to \Omega_{ref} \\
X &\to \phi(X) \in \Omega_{ref}
\end{aligned}
$$

So the loss function is changed to:

$$
\begin{aligned}
L(\phi)
&=[I_{ref}(\phi(X)) - I_{mov}(X)]^2 + \lambda \nabla(\phi)^2 
\end{aligned}
$$

Just like the way we did before, we can regard the mapping as a linear model:

$$
\begin{aligned}
\phi: \Omega_{move} &\to \Omega_{ref} \\
X &\to \phi(X) = X_0' + \Delta X' \\
&= X_0' + \Delta X_a + \Delta X_s \\
\text{s.t. } X_0' &\in \Omega_{ref}
\end{aligned}
$$

where \(X_0'\) is a initial mapping
And we could rewrite the loss function as:

$$
\begin{aligned}
    L(\phi)&=[I_{ref}(\phi(X))-I_{mov}(X)]^2+\lambda\nabla(\phi)^2\\
    &=[I_{ref}(X'+\Delta X_a'+\Delta X_s')-I_{mov}(X)]+\nabla(\Delta X_a+\Delta X_s-\overline{\Delta X_a})^2\\
    &=[I_{ref}(X'+\Delta X_a')+\frac{\partial I_{\text{ref}}(X'+\Delta X_a')}{\partial X} \Delta X_a'-I_{mov}(X)]^2\\
    &+\nabla(\Delta X_a+\Delta X_s-\overline{\Delta X_a})^2
\end{aligned}
$$

Then we have

$$
\begin{aligned}
   \frac {\partial L(\phi)}{\partial \Delta X_a'}&=2\frac{\partial I_{\text{ref}}(X'+\Delta X_a')}{\partial X}[I_{ref}(X'+\Delta X_a')-I_{mov}(X)+\frac{\partial I_{\text{ref}}(X'+\Delta X_a')}{\partial X} \Delta X_a']\\
   &+2\lambda (\Delta X_a+\Delta X_s-\overline{\Delta X_a})\\
   &=0\
\end{aligned}
$$

The format is the same as before, so we needn't to change the solution.


# Simulation
We use a method to build motion field in a more continuous and biophysically motivated way.It first generates dense random fields in the lateral directions (\(x\)and \(y\)), and then smooths them with anisotropic Gaussian filtering to obtain spatially coherent lateral deformations. The axial motion (\(z\)) is not generated independently; instead, it is derived from the depth-wise gradients of the lateral motion fields. In this way, the z-direction displacement is coupled to the variation of lateral deformation across depth, which makes the simulated motion more consistent with realistic volumetric tissue deformation. The amplitudes of lateral and axial motion are normalized separately, allowing independent control of in-plane and out-of-plane motion strength.

Previously, we just randomly select some control poins and generate motion for them, and then smooth the motion field. It will cause some extreme motion in some places, which can't be seen in real bio-images.

We did 10 repeats with various motion smoothness scale, deformation amplitude, and noise level. Here's the result:
![result](images/Simulation.png)

# Remained Question
1.How to get the initial \(X'\) (the initial mapping, I think it's the most important thing)
Now we just pick the corresponding plane as the initial phase

2.The distribition is quite different, and I'm not sure what the exact reason is.
Now we use the histgram mapping to correct the pixels, but it depends on if we have the two sample shape images. Or we just use a simple transform from the sampe plane to correct all the pixels
![result](images/Histgram.png)

3.Shall we still use the anisotropic pyramid
