# A rigorous classical-input separation with restrictive structure

**Primary result:** Liu, Arunachalam & Temme, Nature Physics 17, 1013–1017
(2021), [paper](https://www.nature.com/articles/s41567-021-01287-z).

The paper constructs classical-input classification tasks with labels defined
through discrete-log structure. Under the stated hardness assumption, efficient
classical learners cannot obtain the stipulated predictive advantage, whereas
quantum kernel estimation supports accurate classification. Feature preparation
requires fault-tolerant quantum arithmetic.

## Implementation/use boundary

Preserve the exact concept class, distribution, kernel construction, precision
and assumption. Account for arithmetic, state preparation, training kernel calls
and inference queries before proposing a deployment.

This is useful for understanding what a *computationally hard target structure*
can buy, and for planning fault-tolerant learning research. It is not evidence
that low sample count, noisy labels or many features make a generic CSV suitable
for QML. A small-modulus simulation permits classical enumeration and cannot
demonstrate the asymptotic separation. No such toy win is included here.
