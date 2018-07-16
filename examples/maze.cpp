#include <iostream>
#include <mfhowards>

using namespace mfhowards;
using namespace std;

int main() {

  ///////////////////
  // Problem setup //
  ///////////////////

  constexpr double p_works = .8;  // Probability that robot works correctly
  constexpr double discount = .9; // Discount factor

  enum direction { north, east, south, west }; // Cardinal directions
  enum tile { E /* End */, O /* Walkable */, X /* Obstructed */ };

  constexpr int m = 3, n = 4;
  constexpr tile map[m][n] = {
    {O, O, O, E},
    {O, X, O, E},
    {O, O, O, O},
  };
  constexpr double rewards[m][n] = {
    {0., 0., 0., +1.},
    {0., 0., 0., -1.},
    {0., 0., 0.,  0.}
  };

  ///////////////////////
  // Transition matrix //
  ///////////////////////

  const auto P = [&](int i, int j, direction c) {
    // We have one state per node in the map along with an additional "sink"
    // state, i == m * n. The robot is sent to the sink state when it reaches an
    // end tile (denoted tile::E) signfying the end of the game.

    // Convert the node i to its x and y coordinates
    const int xi = i % n;
    const int yi = i / n;

    // Transition directly to the sink state
    if (i == m * n || map[yi][xi] == E || map[yi][xi] == X) {
      const double p_ij = (j == m * n) ? 1. : 0.;
      return p_ij;
    }

    // Probability that the robot malfunctions is p_fails * 2
    constexpr double p_fails = (1. - p_works) / 2.;

    // p_north, p_east, p_south, p_west, and p_0 will hold the probability of
    // the robot moving north, east, south, west, or staying in the same
    // location.
    double p_north = 0., p_east = 0., p_south = 0., p_west = 0., p_0 = 0.;

    if (yi == 0 || map[yi - 1][xi] == X) {
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

    if (xi == n - 1 || map[yi][xi + 1] == X) {
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

    if (yi == m - 1 || map[yi + 1][xi] == X) {
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

    if (xi == 0 || map[yi][xi - 1] == X) {
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
  };

  ////////////////////////////
  // Run Howard's algorithm //
  ////////////////////////////

  auto results = howards_alg(bellman_eq_from_lambdas<direction>(
    m * n + 1, // Number of states (+1 for sink state i == m * n)
    {north, east, south, west}, // Controls are the cardinal directions
    [&](int i, int j, direction c) {
      // A = I - discount * P
      const double kronecker_ij = (i == j ? 1. : 0.);
      return kronecker_ij - discount * P(i, j, c);
    },
    [&](int i, direction) {
      if (i == m * n) {
        return 0.;
      }
      const int x = i % n, y = i / n;
      return rewards[y][x];
    }
  ));

  cout << results;
  return results.status;
}
