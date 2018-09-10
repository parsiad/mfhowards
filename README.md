# mfhowards
A matrix free implementation of policy iteration (a.k.a. Howard's algorithm) in C++ to solve Markov decision processes (MDPs).

## Basic usage

Howard's algorithm is used to find a vector _v_ which satisfies the so-called Bellman equation:

![](https://latex.codecogs.com/gif.latex?\min_{c&space;\\in&space;C}&space;\\left\\{&space;A(c)&space;v&space;-&space;b(c)&space;\\right\\}=0)

where _C_ is a finite set, _A(c)_ is a real square matrix, _b(c)_ is a real vector.

**Remark:** _For conditions on the matrix A(c) so that Howard's algorithm returns a valid solution v, see [this paper](https://arxiv.org/pdf/1510.03928.pdf) or [this one](https://hal.inria.fr/file/index/docid/179549/filename/RR-zidani.pdf))._

### Boiler-plate code

```cpp
#include <mfhowards>
#include <iostream>

using namespace mfhowards;
using namespace std;

enum MyControlType { c1, c2, c3 };

const auto A = [&](int i, int j, MyControlType c) {
	// TODO: Return a double corresponding to the
	//       (i, j)-th entry of A(c)
};

const auto b = [&](int i, MyControlType c) {
	// TODO: Return a double corresponding to the
	//       i-th entry of b(c)
};

auto results = howards_alg(
	bellman_eq_from_lambdas<MyControlType>(
		num_states,     // Number of states
		{ c1, c2, c3 }, // Control list
		A, b            // Matrix and vector
	)
);

cout << results;
```

**Remark:** While the example above uses an enum for ```MyControlType```, you can use anything you like (e.g., ```typedef double MyControlType```).

### Example

See the [examples/maze.cpp](https://github.com/parsiad/mfhowards/blob/master/examples/maze.cpp) for an implementation of the MDP in Chapter 17.1 of [the Stuart and Norvig book](http://thuvien.thanglong.edu.vn:8081/dspace/handle/DHTL_123456789/4010).
Alternatively, you can refer to [a blog post by Massimiliano Patacchiola](https://mpatacchiola.github.io/blog/2016/12/09/dissecting-reinforcement-learning.html#the-bellman-equation) which summarizes the chapter.

## Advanced usage

The more general interface to the code is as follows:

```cpp
template <typename Beq>
results howards_alg(Beq &&beq);
```

The type ```Beq``` should implement various methods.
Boiler plate code is given below:

```cpp
class MyBeq {
private:

	MatrixType A_c;
	VectorType b_c;
	
public:
	
	int rows() const; // Number of states in MDP
	
	/*
	 * This implements the policy improvement step
	 * of Howard's algorithm:
	 * 
	 * 1. Look for a control that c* that minimizes
	 *    A(c)x - b(c).
	 * 2. Store A(c*) and b(c*) in A_c and b_c.
	 */
	void improve(const Eigen::VectorXd &x);
	
	/*
	 * This implements the policy evaluation step
	 * of Howard's algorithm:
	 * 
	 * 1. Solve the linear system A_c x = b_c.
	 * 2. Return the solution x.
	 * 
	 * The size of the returned vector should equal
	 * the integer returned by rows().
	 */
	Eigen::VectorXd rhs() const;

};
```
