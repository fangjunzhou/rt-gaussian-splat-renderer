# Spherical Harmonics

## Resources

- [Spherical Harmonics Blog](https://patapom.com/blog/SHPortal/)
- [Spherical Harmonic Lighting: the Gritty Details](https://3dvar.com/Green2003Spherical.pdf)

## Details of Spherical Harmonics Computation

A spherical harmonics of degree $l$ and order $m$ is defined as

$$
    Y_{l}^{m} (\theta, \phi) = K_{l}^{m} P_{l}^{m}(cos(\theta)) e^{im \phi}
$$

Where $K_{l}^{m}$ is a normalization constant for each $l$ and $m$, and $P_{l}^{m}$ is the [associated Legendre polynomial](https://en.wikipedia.org/wiki/Associated_Legendre_polynomials) of degree $l$ and order $m$.

Notice that the spherical coordinate $(r, \theta, \phi)$ follows the physics convention here.

$$
    v(\theta, \phi) = (sin(\theta) cos(\phi), sin(\theta) sin(\phi), cos(\theta))
$$

### Associated Legendre Polynomial

The associated Legendre polynomials are a set of orthogonal polynomials that satisfies

$$
\int_{-1}^{1} F_{m}(x) F_{n}(x) dx = \begin{cases}
    0, \text{ if } m \ne n \\
    c, \text{ if } m = n
\end{cases}
$$

For $P_{l}^{m}$, the polynomials are orthogonal w.r.t. a constant between degrees, and they are orthogonal w.r.t. another constant between orders. The details of orthogonality is not discussed in this notes, please read [spherical harmonic lighting: the gritty details](https://3dvar.com/green2003spherical.pdf) for details.

To derive $P_{l}^{m}$, we need to follow three rules:

1. $(l-m) P_{l}^{m} = x (2l - 1) P_{l-1}^{m} - (l + m - 1)P_{l-2}^{m}$
2. $P_{m}^{m} = (-1)^{m} (2m - 1)!! (1 - x^{2})^{m/2}$
3. $P_{m+1}^{m} = x (2m + 1) P_{m}^{m}$

Rule 1 can be used recursively to evaluate $P_{l}^{m}$ for any $l$ and $m$ as long as $l \ge m + 2$. And to get the first two orders of polynomials, we need to use rule 2 and rule 3 to bootstrap the algorithm.

In other words, for any $P_{l}^{m}$,

- If $l = m$, use rule 2.
- If $l = m + 1$, use rule 3.
- If $l \ge m+2$, use rule 1.

![](/_static/image/2025-03-03-15-46-59.png)

#### Example

We can derived the 1st **order** ($m=1$) associated Legendre polynomial with the aforementioned algorithm

$$
\begin{align*}
    P_{1}^{1} &= (-1) \cdot 1 \cdot (1 - x^{2})^{\frac{1}{2}}, \quad (\text{Rule 2}) \\
    &= -(1 - x^{2})^{\frac{1}{2}} \\
    P_{2}^{1} &= x \cdot 3 \cdot P_{1}^{1}, \quad (\text{Rule 3}) \\
    &= -3x (1 - x^{2})^{\frac{1}{2}} \\
    P_{3}^{1} &= \frac{x \cdot 5 \cdot P_{2}^{1} - 3 \cdot P_{1}^{1}}{2}, \quad (\text{Rule 1}) \\
    &= \frac{3}{2} (1 - 5x^{2})(1 - x^{2})^{\frac{1}{2}}
\end{align*}
$$

This checks out with the common associated Legendre functions listed on [Wikipedia](https://en.wikipedia.org/wiki/Associated_Legendre_polynomials#The_first_few_associated_Legendre_functions).

#### Analytical Solutions to Common Associated Legendre Functions

We also checked and use other associated Legendre functions listed on [Wikipedia](https://en.wikipedia.org/wiki/Associated_Legendre_polynomials#The_first_few_associated_Legendre_functions) up to **degree** 3, since 3 degrees of spherical harmonics is accurate enough as mentioned in many literature.


$$
\begin{align*}
    P_{0}^{0} &= 1 \\
    P_{1}^{0} &= x \\
    P_{1}^{1} &= -(1 - x^{2})^{\frac{1}{2}} \\
    P_{2}^{0} &= \frac{1}{2} (3x^{2} - 1) \\
    P_{2}^{1} &= -3x (1 - x^{2})^{\frac{1}{2}} \\
    P_{2}^{2} &= 3 (1 - x^{2}) \\
    P_{3}^{0} &= \frac{1}{2} (5x^{3} - 3x) \\
    P_{3}^{1} &= \frac{3}{2} (1 - 5x^{2}) (1 - x^{2})^{\frac{1}{2}} \\
    P_{3}^{2} &= 15x (1 - x^{2}) \\
    P_{3}^{2} &= -15 (1 - x^{2})^{\frac{3}{2}}
\end{align*}
$$

### Evaluation of Spherical Harmonics

With the normalization factor

$$
    K_{l}^{m} = \sqrt{\frac{2l + 1}{4 \pi} \frac{(l - m)!}{(l + m)!}}
$$

We evaluate the spherical harmonics equation

$$
    Y_{l}^{m} (\theta, \phi) &= K_{l}^{|m|} P_{l}^{|m|}(cos(\theta)) e^{im \phi} \\
    &= K_{l}^{|m|} P_{l}^{|m|}(cos(\theta)) (cos(m \phi) + sin(m \phi) i) \\
$$

According to the [Spherical Harmonics Blog](https://patapom.com/blog/SHPortal/) and [this post on StackExchange](https://math.stackexchange.com/questions/145080/how-are-the-real-spherical-harmonics-derived), the real form of spherical harmonics is given by:

$$
    Y_{l}^{m}(\theta, \phi) = \begin{cases}
        (-1)^{m} \sqrt{2} K_{l}^{-m} P_{l}^{-m}(cos(\theta)) sin(-m\phi), & m < 0 \\
        K_{0}^{0} P_{0}^{0}(cos(\theta)), & m = 0 \\
        (-1)^{m} \sqrt{2} K_{l}^{m} P_{l}^{m}(cos(\theta)) cos(m\phi), & m > 0 \\
    \end{cases}
$$

### Spherical Harmonics in Cartesian Coordinate

It'll be convenient to convert spherical coordinate to cartesian coordinate to evaluate spherical harmonics for normalized direction vector.

We start with

$$
\begin{align*}
    cos(\theta) &= z \\
    sin(\theta) &= \sqrt{1 - z^{2}} \\
    cos(\phi) &= \frac{x}{sin(\theta)} \\
    &= \frac{x}{\sqrt{1 - z^{2}}} \\
    sin(\phi) &= \frac{y}{sin(\theta)} \\
    &= \frac{y}{\sqrt{1 - z^{2}}}
\end{align*}
$$

For all positive integer $m$, we can expand $cos(m \phi)$ and $sin(m \phi)$ using Chebyshev polynomial

$$
\begin{align*}
    cos(m \phi) &= \sum_{k=0}^{\lfloor {m/2} \rfloor} (-1)^{k} \binom{m}{2k} cos^{m-2k}(\phi) sin^{2k}(\phi) \\
    sin(m \phi) &= \sum_{k=0}^{\lfloor {m-1/2} \rfloor} (-1)^{k} \binom{m}{2k+1} cos^{m-2k-1}(\phi) sin^{2k+1}(\phi)
\end{align*}
$$

#### Analytical Solutions to Common Spherical Harmonics

$$
\begin{align*}
    Y_{0}^{0} &= \frac{1}{2} \sqrt{\frac{1}{\pi}} \\
    Y_{1}^{-1} &= \frac{1}{2} \sqrt{\frac{3}{\pi}} y \\
    Y_{1}^{0} &= \frac{1}{2} \sqrt{\frac{3}{\pi}} z \\
    Y_{1}^{1} &= \frac{1}{2} \sqrt{\frac{3}{\pi}} x \\
    Y_{2}^{-2} &= \frac{1}{2} \sqrt{\frac{15}{\pi}} xy \\
    Y_{2}^{-1} &= \frac{1}{2} \sqrt{\frac{15}{\pi}} yz \\
    Y_{2}^{0} &= \frac{1}{4} \sqrt{\frac{5}{\pi}} (3z^{2} - 1) \\
    Y_{2}^{1} &= \frac{1}{2} \sqrt{\frac{15}{\pi}} xz \\
    Y_{2}^{2} &= \frac{1}{4} \sqrt{\frac{15}{\pi}} (x^{2} - y^{2}) \\
    Y_{3}^{-3} &= \frac{1}{4} \sqrt{\frac{35}{2\pi}} y(3x^{2} - y^{2}) \\
    Y_{3}^{-2} &= \frac{1}{2} \sqrt{\frac{105}{\pi}} xyz \\
    Y_{3}^{-1} &= \frac{1}{4} \sqrt{\frac{21}{2\pi}} y(5z^{2} - 1) \\
    Y_{3}^{0} &= \frac{1}{4} \sqrt{\frac{7}{\pi}} (5z^{3} - 3z) \\
    Y_{3}^{1} &= \frac{1}{4} \sqrt{\frac{21}{2\pi}} x(5z^{2} - 1) \\
    Y_{3}^{2} &= \frac{1}{4} \sqrt{\frac{105}{\pi}} (x^{2} - y^{2})z \\
    Y_{3}^{3} &= \frac{1}{4} \sqrt{\frac{35}{2\pi}} x(x^{2} - 3y^{2})
\end{align*}
$$
