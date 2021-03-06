/*
 * maze.cpp
 * Copyright (c) 2018 Parsiad Azimzadeh
 *
 * Description
 * -----------
 *
 * Code to compute the value of the Markov decision process in Chapter 17.1 of
 * [Russell, Stuart J., and Peter Norvig. Artificial intelligence: a modern
 * approach. Malaysia; Pearson Education Limited,, 2016.] whose solution is
 * +0.812 +0.868 +0.918 +1.000
 * +0.762 +0.000 +0.660 -1.000
 * +0.705 +0.655 +0.611 +0.388
 *
 * You can also find a description of this problem at
 * https://mpatacchiola.github.io/blog/2016/12/09/dissecting-reinforcement-learning.html#the-bellman-equation
 *
 * The querying of the transition matrix (which is never explicitly stored)
 * in this example is not efficient, and as such, should be used only as a
 * reference for how to use the library.
 */

#include <iomanip>
#include <iostream>

#include <mfhowards>

using namespace mfhowards;
using namespace std;

///////////////////
// Problem setup //
///////////////////

constexpr double p_works = .8;  // Probability that robot works correctly

constexpr double discount = 1.; // Discount factor (1 means no discount)

// The map consists of tiles of type O and X, signifying walkable and
// obstructed tiles.
// There is also a special tile of type E, which signifies that the robot has
// reached an exit.
enum tile { E, O, X };

constexpr int m = 3, n = 4; // Size of the map

constexpr tile maze[m][n] = { // The map
  {O, O, O, E},
  {O, X, O, E},
  {O, O, O, O},
};

constexpr double rewards[m][n] = { // Rewards corresponding to each map tile
  {-0.04, -0.04, -0.04, +1.00},
  {-0.04, +0.00, -0.04, -1.00},
  {-0.04, -0.04, -0.04, -0.04},
};

enum direction { north, east, south, west }; // Cardinal directions

// Each state is represented as i = x + n * y, where (x, y) is a map index.
// There is also an additional state, i = m * n, which the robot is sent to
// after landing on a tile of type E.
// The transition from state i to state j, given that the robot is facing in
// direction c, is given by P(i, j, c).
double P(int i, int j, direction c);

int main() {

  ////////////////////////////
  // Run Howard's algorithm //
  ////////////////////////////

  // The value of this Markov decision process is the unique solution v of
  // min{ A(c)v - b(c) } == 0 where A(c) and b(c) are a real square matrix and
  // real vector whose entries are given below.

  const auto A = [&](int i, int j, direction c) {
    // A = I - discount * P
    const double kronecker_ij = (i == j ? 1. : 0.);
    return kronecker_ij - discount * P(i, j, c);
  };

  const auto b = [&](int i, direction) {
    if (i == m * n) { return 0.; }
    const int x = i % n, y = i / n;
    return rewards[y][x];
  };

  auto results = howards_alg(bellman_eq_from_lambdas<direction>(
      m * n + 1, // Number of states (+1 for state i = m * n)
      {north, east, south, west}, // Controls are the cardinal directions
      A, b));

  ///////////////////
  // Print results //
  ///////////////////

  int i = 0;
  for (int y = 0; y < m; ++y) {
    for (int x = 0; x < n; ++x) {
      cout << std::fixed << std::setw(11) << std::setprecision(6)
           << results.value(i++);
    }
    cout << endl;
  }

  return results.status;

}

///////////////////////
// Transition matrix //
///////////////////////

double P(int i, int j, direction c) {

  // Convert the node i to its x and y coordinates
  const int xi = i % n;
  const int yi = i / n;

  // Transition directly to the sink state
  if (i == m * n || maze[yi][xi] == E || maze[yi][xi] == X) {
    const double p_ij = (j == m * n) ? 1. : 0.;
    return p_ij;
  }

  // Probability that the robot malfunctions is p_fails * 2
  constexpr double p_fails = (1. - p_works) / 2.;

  // p_north, p_east, p_south, p_west, and p_0 are the probability of the
  // robot moving north, east, south, west, or staying in the same location.
  double p_north = 0., p_east = 0., p_south = 0., p_west = 0., p_0 = 0.;

  if (yi == 0 || maze[yi - 1][xi] == X) {
    // CANNOT move north
    if (c == north) {
      p_0 += p_works;
    } else if (c == east || c == west) {
      p_0 += p_fails;
    }
  } else {
    // CAN move north
    if (c == north) {
      p_north = p_works;
    } else if (c == east || c == west) {
      p_north = p_fails;
    }
  }

  if (xi == n - 1 || maze[yi][xi + 1] == X) {
    // CANNOT move east
    if (c == east) {
      p_0 += p_works;
    } else if (c == north || c == south) {
      p_0 += p_fails;
    }
  } else {
    // CAN move east
    if (c == east) {
      p_east = p_works;
    } else if (c == north || c == south) {
      p_east = p_fails;
    }
  }

  if (yi == m - 1 || maze[yi + 1][xi] == X) {
    // CANNOT move south
    if (c == south) {
      p_0 += p_works;
    } else if (c == east || c == west) {
      p_0 += p_fails;
    }
  } else {
    // CAN move south
    if (c == south) {
      p_south = p_works;
    } else if (c == east || c == west) {
      p_south = p_fails;
    }
  }

  if (xi == 0 || maze[yi][xi - 1] == X) {
    // CANNOT move west
    if (c == west) {
      p_0 += p_works;
    } else if (c == north || c == south) {
      p_0 += p_fails;
    }
  } else {
    // CAN move west
    if (c == west) {
      p_west = p_works;
    } else if (c == north || c == south) {
      p_west = p_fails;
    }
  }

  // Return p_ij, the probability of transitioning from i to j
  double p_ij;
  const int xj = j % n, yj = j / n;
  if (xj == xi && yj == yi - 1) {
    p_ij = p_north;
  } else if (xj == xi + 1 && yj == yi) {
    p_ij = p_east;
  } else if (xj == xi && yj == yi + 1) {
    p_ij = p_south;
  } else if (xj == xi - 1 && yj == yi) {
    p_ij = p_west;
  } else if (xj == xi && yj == yi) {
    p_ij = p_0;
  } else {
    p_ij = 0.;
  }
  return p_ij;

}

