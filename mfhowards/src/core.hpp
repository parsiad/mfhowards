/**
 * MIT License
 *
 * Copyright (c) 2018 Parsiad Azimzadeh
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef MFHOWARDS_CORE_HPP
#define MFHOWARDS_CORE_HPP

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>

#include <chrono>   // std::chrono
#include <cstddef>  // std::size_t
#include <iostream> // std::ostream
#include <numeric>  // std::accumulate
#include <vector>   // std::vector

namespace mfhowards {

namespace {

struct results {
  Eigen::VectorXd value;
  int outer_iters;
  std::vector<int> inner_iters;
  int status;
  double time_ms;

  results(const Eigen::VectorXd &value) : value(value) {}

  results(results &&r) {
    value = std::move(r.value);
    outer_iters = r.outer_iters;
    inner_iters = std::move(r.inner_iters);
    status = r.status;
  }

  results(const results &) = delete;
  results &operator=(const results &) = delete;
};

std::ostream &operator<<(std::ostream &os, const results &r) {
  const double avg_inner_iters =
      accumulate(r.inner_iters.begin(), r.inner_iters.end(), 0.) /
      r.inner_iters.size();
  os << "value = " << r.value.transpose() << std::endl
     << "[status = " << r.status << ", "
     << "outer iterations = " << r.outer_iters << ", "
     << "average inner iterations = " << avg_inner_iters << ", "
     << "time = " << r.time_ms << " ms]" << std::endl;
  return os;
}

} // namespace

template <typename T,
          typename S = Eigen::BiCGSTAB<T, Eigen::IdentityPreconditioner>>
results howards_alg(T &&beq, Eigen::VectorXd x0 = Eigen::VectorXd(0),
                    const double rtol = 1e-5, const double atol = 1e-8,
                    const int max_iters = 1000) {
  S solver;

  if (x0.size() == 0) {
    x0 = Eigen::VectorXd::Zero(beq.rows());
  }
  results r(x0);

  auto t0 = std::chrono::steady_clock::now();

  while (true) {
    // If maximum number of iterations reached, return with bad status
    if (r.outer_iters == max_iters) {
      r.status = 1;
      break;
    }

    // Policy improvement
    beq.improve(r.value);

    // Policy evaluation
    solver.compute(beq);
    x0 = r.value;
    r.value = solver.solve(beq.rhs());
    r.outer_iters += 1;
    r.inner_iters.push_back(solver.iterations());

    // If the solver fails, return with bad status
    if (solver.info() != Eigen::Success) {
      r.status = 255;
      break;
    }

    // If error tolerance met, return with good status
    const Eigen::VectorXd delta =
        (x0 - r.value).cwiseAbs() - rtol * r.value.cwiseAbs();
    const bool converged = (delta.array() < atol).all();
    if (converged) {
      r.status = 0;
      break;
    }
  }

  auto t1 = std::chrono::steady_clock::now();
  auto delta = t1 - t0;
  r.time_ms = std::chrono::duration<double, std::milli>(delta).count();

  return r;
}

} // namespace mfhowards

#endif
