# mfhowards
A matrix free implementation of policy iteration (a.k.a. Howard's algorithm) in C++.

## Description of Howard's algorithm

Let _C = C1 x C2 x ... x Cn_ be a finite set.
For each _c = (c1, c2, ..., cn)_ in _C_, let _A(c)_ and _b(c)_ be an _n x n_ matrix and an _n x 1_ real vector (it is understood that the _i_-th rows of _A(c)_ and _b(c)_ depend only on _ci_).

Howard's algorithm is used to find a vector _v_ which satisfies the [Bellman equation](https://en.wikipedia.org/wiki/Bellman_equation#The_Bellman_equation):

![](https://latex.codecogs.com/gif.latex?\min_{c&space;\\in&space;C}&space;\\left\\{&space;A(c)&space;v&space;-&space;b(c)&space;\\right\\}=0)

**Remark:** _A sufficient condition for Howard's algorithm to return the unique solution is for A(c) to be a monotone matrix for all c; see [this paper](https://arxiv.org/pdf/1510.03928.pdf) or [this one](https://hal.inria.fr/file/index/docid/179549/filename/RR-zidani.pdf)) for details and further results._

## Boiler-plate code

Before trying to fill in the boiler-plate code below, I recommend also looking through the [example](#example).

```cpp
#include <mfhowards>
#include <iostream>

using namespace mfhowards;
using namespace std;

int main() {
	// Example with control set C = {eat, pray, code} x ... x {eat, pray, code}
	// i.e., in each state, the controller can eat, pray, or code.
	enum MyControlType { eat, pray, code };
	
	// You are not restricted to using an enum for your control type!
	// Any typename (e.g., double) works.

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
			num_states,          // Number of states
			{ eat, pray, code }, // List of all controls
			A, b                 // Matrix and vector
		)
	);

	// Print results and return status
	cout << results;
	return results.status;
}
```

## Example

![](https://mpatacchiola.github.io/blog/images/reinforcement_learning_simple_world.png)

See [examples/maze.cpp](https://github.com/parsiad/mfhowards/blob/master/examples/maze.cpp) for an implementation of the robot navigation problem pictured above.

The problem is described in detail in
* [a blog post by Massimiliano Patacchiola](https://mpatacchiola.github.io/blog/2016/12/09/dissecting-reinforcement-learning.html#the-bellman-equation)
* Chapter 17.1 of [the Stuart and Norvig book](http://aima.cs.berkeley.edu/)

## Advanced usage

The more general interface to the code is as follows:

```cpp
template <typename Beq>
results howards_alg(Beq &&beq);
```

You should creat your own ```Beq``` type to implement various methods called by howards_alg.
Some boiler-plate for a ``Beq`` type is given below:

```cpp
class MyBeqType {

private:
	MyMatrixType A_c;
	MyVectorType b_c;
	
public:
	/*
	   FILL THIS IN: This method should...
	   - Return the number of rows in A(c) and b(c).
	 */
	int rows() const;
	
	/*
	   FILL THIS IN: This method should...
	   - Look for a control that c* = (c*_1, c*_2, ..., c*_rows()) that minimizes A(c)x - b(c).
	   - Store A(c*) and b(c*) in A_c and b_c.
	 */
	void improve(const Eigen::VectorXd &x);
	
	/*
	   FILL THIS IN: This method should...
	   - Solve the linear system A_c x = b_c.
	   - Return the solution x. The size of x should coincide with rows().
	 */
	Eigen::VectorXd rhs() const;

};
```
