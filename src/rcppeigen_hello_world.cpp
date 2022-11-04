// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppEigen.h which pulls Rcpp.h in for us
#include <RcppEigen.h>
using namespace Rcpp;


// typedef Eigen::Map<Eigen::MatrixXd> MapMatd;


// via the depends attribute we tell Rcpp to create hooks for
// RcppEigen so that the build process will know what to do
//
// [[Rcpp::depends(RcppEigen)]]

// simple example of creating two matrices and
// returning the result of an operatioon on them
//
// via the exports attribute we tell Rcpp to make this function
// available from R
//
// [[Rcpp::export]]
Eigen::MatrixXd rcppeigen_hello_world() {
  Eigen::MatrixXd m1 = Eigen::MatrixXd::Identity(3, 3);
  // Eigen::MatrixXd m2 = Eigen::MatrixXd::Random(3, 3);
  // Do not use Random() here to not promote use of a non-R RNG
  Eigen::MatrixXd m2 = Eigen::MatrixXd::Zero(3, 3);
  for (auto i=0; i<m2.rows(); i++)
    for (auto j=0; j<m2.cols(); j++)
      m2(i,j) = R::rnorm(0, 1);
  
  return m1 + 3 * (m1 + m2);
}


// another simple example: outer product of a vector,
// returning a matrix
//
// [[Rcpp::export]]
Eigen::MatrixXd rcppeigen_outerproduct(const Eigen::VectorXd & x) {
  Eigen::MatrixXd m = x * x.transpose();
  return m;
}

// and the inner product returns a scalar
//
// [[Rcpp::export]]
double rcppeigen_innerproduct(const Eigen::VectorXd & x) {
  double v = x.transpose() * x;
  return v;
}

// and we can use Rcpp::List to return both at the same time
//
// [[Rcpp::export]]
Rcpp::List rcppeigen_bothproducts(const Eigen::VectorXd & x) {
  Eigen::MatrixXd op = x * x.transpose();
  double          ip = x.transpose() * x;
  return Rcpp::List::create(Rcpp::Named("outer")=op,
                            Rcpp::Named("inner")=ip);
}


extern "C" SEXP Cdqrls(SEXP x, SEXP y, SEXP tol, SEXP chk);

Eigen::VectorXd convert2Eigen(Eigen::VectorXd x){
  return x;
}

Eigen::VectorXd LogisticFunction(Eigen::MatrixXd x, Eigen::VectorXd beta){
  return (1 + (-x*beta).array().exp()).pow(-1);
}

Eigen::MatrixXd WeightedLS(Eigen::MatrixXd x, Eigen::VectorXd y, Eigen::VectorXd w){
  Eigen::VectorXd w_sqrt = w.array().sqrt();
  Eigen::VectorXd w_inv = w.array().pow(-1);
  
  Eigen::DiagonalMatrix<double, Eigen::Dynamic> W_sqrt(w_sqrt);
  Eigen::DiagonalMatrix<double, Eigen::Dynamic> W_inv(w_inv);
  
  x = W_sqrt * x;
  y = W_sqrt * y;
  
  return x.householderQr().solve(y);
  // Environment stats("package:stats");
  // Function f = stats["C_Cqdrls"];
  // SEXP beta = f(wrap(x), wrap(y), wrap(1e-8), wrap(false));
  // const Eigen::Map<Eigen::MatrixXd> beta_map(as<Eigen::Map<Eigen::MatrixXd>>(beta));
  // return Eigen::MatrixXd(beta_map);
}


// test function
//
// [[Rcpp::export]]
Eigen::MatrixXd logistic_regression(const Eigen::MatrixXd & x, const Eigen::VectorXd & y) {
  Eigen::VectorXd beta = Eigen::VectorXd::Ones(x.cols(), 1).cast<double>();
  Eigen::VectorXd beta_old = beta;
  // save X^T
  Eigen::MatrixXd xT = x.transpose();
  
  
  double diff = 1;
  int counter = 0;
  double dev_old = 1000;
  double dev_new = 0;
  double dev = 0;
  while(diff > 1e-8){
    Rcpp::Rcout << "beta: \n" << beta << "\n";
    Eigen::VectorXd p = LogisticFunction(x, beta);
    Eigen::VectorXd variance = p.array() * (1 - p.array());
    Eigen::VectorXd modified_response = (variance.array().pow(-1) * (y - p).array());
    Eigen::VectorXd Z = x * beta + modified_response;
    beta = WeightedLS(x, Z, variance);
    // variance = variance.array().sqrt();
    Eigen::DiagonalMatrix<double, Eigen::Dynamic> W(variance);
    // Eigen::DiagonalMatrix<double, Eigen::Dynamic> W_inv(variance.pow(-1));
    // Eigen::MatrixXd matrix_to_invert = W * x;
    // Rcpp::Rcout << "Matrix inversion dims: " << (W * x).rows() << "\t" << (W * x).cols() << "\n";
    // Eigen::FullPivHouseholderQR<Eigen::MatrixXd> qr(matrix_to_invert.rows(), matrix_to_invert.cols());
    // qr.compute(matrix_to_invert);
    // Eigen::VectorXd z = W *x*beta + W_inv * (y - p);
    // beta = qr.solve(z);
    counter ++;
    
    diff = (beta - beta_old).norm();
    beta_old = beta;
    
    // compute stopping criteria
    Eigen::VectorXd one_minus_y = 1 - y.array();
    Eigen::VectorXd positiveExamples = (W * y).array() * p.array().log().unaryExpr([](double v) { return std::isfinite(v)? v : 0.0; });;
    Eigen::VectorXd negativeExamples = (W * one_minus_y).array() * (1-p.array()).log().unaryExpr([](double v) { return std::isfinite(v)? v : 0.0; });;
    
    
    // Rcpp::Rcout << (1-p.array()).minCoeff() << "\t" << (1-p.array()).maxCoeff() << "\n";
    // Rcpp::Rcout << positiveExamples.sum() << "\t" << negativeExamples.sum() << "\n";
    dev_new = 2*(positiveExamples.sum() + negativeExamples.sum());
    
    diff = std::abs(dev_new - dev_old) / (0.1 + std::abs(dev_old));
    dev_old = dev_new;
    Rcpp::Rcout << diff << "\t" << counter << "\n";
    // 
    // if(diff < 1e-3){
    //   // Rcpp::Rcout <<positiveExamples << "\n" << negativeExamples << "\n";
    // }

    
  }
  return beta;
}