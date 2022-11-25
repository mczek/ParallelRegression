// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-
// [[Rcpp::depends(RcppEigen)]]

#include <RcppEigen.h>
#include <thread>


Eigen::VectorXd LogisticFunction(const Eigen::MatrixXd x, const Eigen::VectorXd beta){
  return (1 + (-x*beta).array().exp()).pow(-1);
}

Eigen::VectorXd WeightedLS(Eigen::MatrixXd x, Eigen::VectorXd y, Eigen::VectorXd w){
  Eigen::VectorXd w_sqrt = w.array().sqrt();
  Eigen::VectorXd w_inv = w.array().pow(-1);
  
  Eigen::DiagonalMatrix<double, Eigen::Dynamic> W_sqrt(w_sqrt);
  Eigen::DiagonalMatrix<double, Eigen::Dynamic> W_inv(w_inv);
  
  x = W_sqrt * x;
  y = W_sqrt * y;
  
  // https://stackoverflow.com/questions/58350524/eigen3-select-rows-out-based-on-column-conditions
  Eigen::VectorXi is_selected = (w.array() > 0).cast<int>();
  Eigen::MatrixXd x_new(is_selected.sum(), x.cols());
  Eigen::VectorXd y_new(is_selected.sum());
  int rownew = 0;
  for (int i = 0; i < x.rows(); ++i) {
    if (is_selected[i]) {       
      x_new.row(rownew) = x.row(i);
      y_new.row(rownew) = y.row(i);
      rownew++;
    }
  }
  
  return x_new.householderQr().solve(y_new);
}


Eigen::VectorXd LogisticRegressionTask(const Eigen::MatrixXd & x, const Eigen::VectorXd & y) {
  // Rcpp::Rcout << "logistic regression: " << x.rows() << "\t" << x.cols() << "\n";
  Eigen::VectorXd beta = Eigen::VectorXd::Ones(x.cols(), 1).cast<double>();
  Eigen::VectorXd beta_old = beta;
  // save X^T
  Eigen::MatrixXd xT = x.transpose();
  
  double diff = 1;
  int counter = 0;
  double dev_old = 1000;
  double dev_new = 0;
  double dev = 0;
  while(counter < 25){
    // Rcpp::Rcout << "beta: \n" << beta << "\n";
    Eigen::VectorXd p = LogisticFunction(x, beta);
    Eigen::VectorXd variance = p.array() * (1 - p.array());
    Eigen::VectorXd modified_response = (variance.array().pow(-1) * (y - p).array());
    Eigen::VectorXd Z = x * beta + modified_response;
    beta = WeightedLS(x, Z, variance);
    Eigen::DiagonalMatrix<double, Eigen::Dynamic> W(variance);
    counter ++;
    
    diff = (beta - beta_old).norm();
    beta_old = beta;
    
    // compute stopping criteria
    Eigen::VectorXd one_minus_y = 1 - y.array();
    Eigen::VectorXd positiveExamples = (W * y).array() * p.array().log().unaryExpr([](double v) { return std::isfinite(v)? v : 0.0; });;
    Eigen::VectorXd negativeExamples = (W * one_minus_y).array() * (1-p.array()).log().unaryExpr([](double v) { return std::isfinite(v)? v : 0.0; });;
    
    dev_new = 2*(positiveExamples.sum() + negativeExamples.sum());
    
    diff = std::abs(dev_new - dev_old) / (0.1 + std::abs(dev_old));
    dev_old = dev_new;
    
  }
  return beta;
}

void PartitionedRegressionTask(const Eigen::MatrixXd & x, const Eigen::VectorXd & y, int id, int nrows, Eigen::VectorXd beta_array[]){
  Eigen::MatrixXd x_partition(nrows, x.cols());
  Eigen::VectorXd y_partition(nrows);
  
  int rownew = 0;
  int start = id*nrows;
  int end = std::min((id+1)*nrows, (int)x.rows());
  for (int i = start; i < end; i++) {
    x_partition.row(rownew) = x.row(i);
    y_partition.row(rownew) = y.row(i);
    rownew++;
  }
  
  Eigen::VectorXd beta = LogisticRegressionTask(x_partition, y_partition);
  beta_array[id] = beta;
}


// test function
//
//' @export
// [[Rcpp::export]]
Eigen::VectorXd ParLR(const Eigen::MatrixXd & x, const Eigen::VectorXd & y, int ncores=1) {
  Eigen::VectorXd beta_array[ncores];
  int nrows_per_task = (int)x.rows() / ncores + 1; 
  // Rcpp::Rcout << "nrows_per_task: " << nrows_per_task << "\n";
  
  
  std::thread workers[ncores];
  if (ncores > 1){
    for(int i=1; i<ncores; i++){
      workers[i] = std::thread([&x, &y, &i, &nrows_per_task, &beta_array] {PartitionedRegressionTask(x, y, i, nrows_per_task, beta_array);});
      PartitionedRegressionTask(x, y, 0, nrows_per_task, beta_array);
      workers[i].join();
    }
  } else {
    PartitionedRegressionTask(x, y, 0, nrows_per_task, beta_array);
  }
  
  
  
  // simple average for now 
  Eigen::VectorXd beta_agg = Eigen::VectorXd::Zero(x.cols(), 1).cast<double>();
  for(int i=0; i<ncores; i++){
    beta_agg = beta_agg + beta_array[i];
  }
  
  
  
  return beta_agg.array() / ncores;
}