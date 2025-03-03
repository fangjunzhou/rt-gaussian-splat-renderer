# Spherical Harmonics

## Resources

- [Spherical Harmonics Blog](https://patapom.com/blog/SHPortal/)
- [spherical harmonic lighting: the gritty details](https://3dvar.com/green2003spherical.pdf)

## Details of Spherical Harmonics Computation

A spherical harmonics of degree $l$ and order $m$ is defined as

$$
Y_{l}^{m} (\theta, \phi) = K_{l}^{m} P_{l}^{m}(cos(\theta)) e^{im \phi}
$$

Where $K_{l}^{m}$ is a normalization constant for each $l$ and $m$, and $P_{l}^{m}$ is the [associated Legendre polynomial](https://en.wikipedia.org/wiki/Associated_Legendre_polynomials) of degree $l$ and order $m$.

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
