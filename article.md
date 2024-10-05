# Some Custimization Based On Wang's FGB Method
## Contact Constraint
Instead of volume preserving method, we use nearest point pair's distance to fomulate the energy.
$$
E_{c}= \frac{1}{2}max(d- \frac{(x_a-x_b)_1(x_a-x_b)_2}{||x_a-x_b||},0)^2\\
$$
- $x_i$ is the position of nearest point of convex shape $i$ to another

fix any other terms except $(x_a-x_b)_1$ we got:
$$
E_c'(x_a)= -\frac{x_a-x_b}{||x_a-x_b||}(d- \frac{(x_a-x_b)_1(x_a-x_b)_2}{||x_a-x_b||}),if ...
$$
keep fix except $(x_a-x_b)_1$ we got:
$$
E_c''(x_a)=\frac{x_a-x_b}{||x_a-x_b||}\times \frac{x_a-x_b}{||x_a-x_b||},if ...
$$

---
Instead of linearize contact constraint then solve LCP problem, we ultilize the projected gradiant and hession of the energy to fomulate the problem.
$$
E_m'(x)+\lambda E_c'(x_p)=0\\
E_c'(x_p)+E_c''(x_p)(x-x_p)=0
$$
- $E_m(x)=\frac{1}{2}(x-x_t)^2$ is the energy to minimize 
  
we got:
$$
(x-x_t)+\lambda E_c'(x_p)=0\\
\rightarrow x=x_t-\lambda E_c'(x_p)\\
$$
we got:
$$
E_c'(x_p)+E_c''(x_p)(x_t-\lambda E_c'(x_p)-x_p)=0\\
$$
let $n=\frac{E_c'(x_p)}{||E_c'(x_p)||},l= ||E_c'(x_p)||,h=E_c''(x_p)$, we project the second term on $n$:
$$
ln+h(x_t-\lambda ln-x_p)nn=0\\
\rightarrow l+h(x_t-\lambda ln-x_p)n=0\\
\rightarrow \lambda=\frac{l+h(x_t-x_p)n
}{lhnn}\\$$
