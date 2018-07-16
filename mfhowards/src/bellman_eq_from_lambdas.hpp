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

#ifndef MFHOWARDS_BELLMAN_EQ_FROM_LAMBDAS_HPP
#define MFHOWARDS_BELLMAN_EQ_FROM_LAMBDAS_HPP

#include <functional>       // std::function
#include <initializer_list> // std::initializer_list
#include <limits>           // std::numeric_limits
#include <utility>          // std::forward

namespace mfhowards {
namespace {
template <typename CtrlT> class bellman_eq_from_lambdas;
}
} // namespace mfhowards

namespace Eigen {
namespace internal {
template <typename CtrlT>
struct traits<mfhowards::bellman_eq_from_lambdas<CtrlT>>
    : public Eigen::internal::traits<Eigen::SparseMatrix<double>> {};
} // namespace internal
} // namespace Eigen

namespace mfhowards {
namespace {
template <typename CtrlT>
class bellman_eq_from_lambdas
    : public Eigen::EigenBase<bellman_eq_from_lambdas<CtrlT>> {

public:
  typedef double Scalar;
  typedef double RealScalar;
  typedef int StorageIndex;

  enum {
    ColsAtCompileTime = Eigen::Dynamic,
    MaxColsAtCompileTime = Eigen::Dynamic,
    IsRowMajor = false
  };

private:
  typedef std::function<Scalar(int, int, CtrlT)> FA;
  typedef std::function<Scalar(int, CtrlT)> Fb;

  int n;
  std::vector<CtrlT> ctrl_set;
  FA A;
  Fb b;
  CtrlT *curr_ctrls;

  template <typename Rhs>
  using Helper = Eigen::Product<bellman_eq_from_lambdas<CtrlT>, Rhs,
                                Eigen::AliasFreeProduct>;

public:
  int rows() const { return n; }
  int cols() const { return n; }

  template <typename Rhs>
  Helper<Rhs> operator*(const Eigen::MatrixBase<Rhs> &x) const {
    return Helper<Rhs>(*this, x.derived());
  }

  template <typename T>
  bellman_eq_from_lambdas(const int n, T &&ctrl_set, FA A, Fb b)
      : ctrl_set(std::forward<T>(ctrl_set)) {
    this->n = n;
    this->A = A;
    this->b = b;
    curr_ctrls = new CtrlT[n];
  }

  bellman_eq_from_lambdas(const int n, std::initializer_list<CtrlT> list, FA A,
                          Fb b)
      : bellman_eq_from_lambdas(n, std::vector<CtrlT>(list), A, b) {}

  ~bellman_eq_from_lambdas() { delete[] curr_ctrls; }

  bellman_eq_from_lambdas(const bellman_eq_from_lambdas &) = delete;
  bellman_eq_from_lambdas &operator=(const bellman_eq_from_lambdas &) = delete;

  void improve(const Eigen::VectorXd &x) {
    Eigen::VectorXd best =
        Eigen::VectorXd::Ones(rows()) * std::numeric_limits<double>::infinity();
    for (auto c : ctrl_set) {
      for (int i = 0; i < rows(); ++i) {
        double tmp = 0;
        for (int j = 0; j < cols(); ++j) {
          tmp += A(i, j, c) * x(j) - b(i, c);
        }
        if (tmp < best(i)) {
          best(i) = tmp;
          curr_ctrls[i] = c;
        }
      }
    }
  }

  Eigen::VectorXd rhs() const {
    Eigen::VectorXd rhs(rows());
    for (int i = 0; i < rows(); ++i) {
      rhs(i) = b(i, curr_ctrls[i]);
    }
    return rhs;
  }

  template <typename, typename, typename, typename, int>
  friend class Eigen::internal::generic_product_impl;
};
} // namespace
} // namespace mfhowards

namespace Eigen {
namespace internal {
template <typename CtrlT, typename Rhs>
struct generic_product_impl<mfhowards::bellman_eq_from_lambdas<CtrlT>, Rhs,
                            SparseShape, DenseShape, GemvProduct>
    : generic_product_impl_base<
          mfhowards::bellman_eq_from_lambdas<CtrlT>, Rhs,
          generic_product_impl<mfhowards::bellman_eq_from_lambdas<CtrlT>,
                               Rhs>> {

  typedef
      typename Product<mfhowards::bellman_eq_from_lambdas<CtrlT>, Rhs>::Scalar
          Scalar;

  template <typename Dest>
  static void
  scaleAndAddTo(Dest &dst, const mfhowards::bellman_eq_from_lambdas<CtrlT> &lhs,
                const Rhs &rhs, const Scalar &) {
    for (int i = 0; i < lhs.rows(); ++i) {
      for (int j = 0; j < lhs.cols(); ++j) {
        dst(i) += lhs.A(i, j, lhs.curr_ctrls[i]) * rhs(j);
      }
    }
  }
};
} // namespace internal
} // namespace Eigen

#endif
