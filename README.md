# mfhowards
A matrix free implementation of policy iteration (a.k.a. Howard's algorithm) in C++.

## Description of Howard's algorithm

Let _C_ be a finite set.
For each _c_ in _C_, let _A(c)_ and _b(c)_ be a real square matrix and real vector.
Howard's algorithm is used to find a vector _v_ which satisfies the [Bellman equation](https://en.wikipedia.org/wiki/Bellman_equation#The_Bellman_equation):

![](https://latex.codecogs.com/gif.latex?\min_{c&space;\\in&space;C}&space;\\left\\{&space;A(c)&space;v&space;-&space;b(c)&space;\\right\\}=0)

**Remark:** _For conditions on A(c) to ensure that the algorithm returns a valid solution v, see [this paper](https://arxiv.org/pdf/1510.03928.pdf) or [this one](https://hal.inria.fr/file/index/docid/179549/filename/RR-zidani.pdf))._

## Boiler-plate code

```cpp
#include <mfhowards>
#include <iostream>

using namespace mfhowards;
using namespace std;

int main() {
	// Example with three controls
	enum MyControlType { c1, c2, c3 };

	const auto A = [&](int i, int j, MyControlType c) {
		// FILL THIS IN: Return a double corresponding to the (i, j)-th entry of A(c)
		return 0.;
	};

	const auto b = [&](int i, MyControlType c) {
		// FILL THIS IN: Return a double corresponding to the i-th entry of b(c)
		return 0.;
	};

	auto results = howards_alg(
		bellman_eq_from_lambdas<MyControlType>(
			num_states,     // Number of states
			{ c1, c2, c3 }, // Control list
			A, b            // Matrix and vector
		)
	);

	// Print results and return status
	cout << results;
	return results.status;
}
```

**Remark:** While the example above uses an enum for ```MyControlType```, you can use anything you like (e.g., ```typedef double MyControlType```).

## Example

See the [examples/maze.cpp](https://github.com/parsiad/mfhowards/blob/master/examples/maze.cpp) for an implementation of the Markov Decision Process (MDP) in Chapter 17.1 of [the Stuart and Norvig book](http://thuvien.thanglong.edu.vn:8081/dspace/handle/DHTL_123456789/4010).
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
	
	int rows() const; // Number of rows in A(c) and b(c)
	
	/*
	 * This implements the policy improvement step of Howard's algorithm:
	 * 
	 * 1. Look for a control that c* that minimizes A(c)x - b(c).
	 * 2. Store A(c*) and b(c*) in A_c and b_c.
	 */
	void improve(const Eigen::VectorXd &x);
	
	/*
	 * This implements the policy evaluation step of Howard's algorithm:
	 * 
	 * 1. Solve the linear system A_c x = b_c.
	 * 2. Return the solution x.
	 * 
	 * The size of the returned vector should equal the integer returned by rows().
	 */
	Eigen::VectorXd rhs() const;

};
```
