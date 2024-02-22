# heteroCL
Designing a small naive implementation of a compute kernel

Task: 
1) Design a small naive implementation of a compute kernel that will accept five matrices A (P X Q), B (Q x R), C (R x S), D (S x T), E (P, T), and F (P, T) and will compute F = alpha * A.B * beta * C.D + gamma * E, where alpha, beta, and gamma are three floating point constants. Note the dimensions of each of the matrices are given in the parentheses.
2) Write a small test bench in the HeteroCL to show that your implementation is actually right and generating the expected results. Consider testbench where the matrix elements are either integers or floating point numbers.
3) Apply certain optimizations, also known as customizations, to optimize your design. Compare the results with the result of the naive implementation.


Installing HeteroCL: 

git clone https://github.com/cornell-zhang/heterocl.git heterocl-mlir
cd heterocl-mlir
git submodule update --init --recursive
pip install . -v --user 

#cd to hcl-dialect/build and expot
export PYTHONPATH=$(pwd)/tools/hcl/python_packages/hcl_core:${PYTHONPATH}

#Go back to the root directory
#cd to root directory heterocl-mlir
export LLVM_BUILD_DIR=$(pwd)/hcl-dialect/externals/llvm-project/build
export PATH=${LLVM_BUILD_DIR}/bin:${PATH}

#To make sure you have successfully installed HeteroCL run the tests
python3 -m pytest tests
