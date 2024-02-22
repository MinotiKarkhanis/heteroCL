# heteroCL
Designing a small naive implementation of a compute kernel

Task: 
1) Design a small naive implementation of a compute kernel that will accept five matrices A (P X Q), B (Q x R), C (R x S), D (S x T), E (P, T), and F (P, T) and will compute F = alpha * A.B * beta * C.D + gamma * E, where alpha, beta, and gamma are three floating point constants. Note the dimensions of each of the matrices are given in the parentheses.
2) Write a small test bench in the HeteroCL to show that your implementation is actually right and generating the expected results. Consider testbench where the matrix elements are either integers or floating point numbers.
3) Apply certain optimizations, also known as customizations, to optimize your design. Compare the results with the result of the naive implementation.
