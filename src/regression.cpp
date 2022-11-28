// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-
// [[Rcpp::depends(RcppEigen)]]

#include <RcppEigen.h>
#include <thread>

class Solver{
public:
  

  // declare variables shared across threads
  Eigen::MatrixXd x_;
  Eigen::VectorXd y_;
  int ncores_;
  std::vector<int> niter_;
  std::vector<Eigen::VectorXd> all_betas_;
  std::vector<Eigen::VectorXd> current_betas_;
  
  // constructor
  Solver(Eigen::MatrixXd x, Eigen::VectorXd y, int ncores) : x_(x), y_(y), ncores_(ncores) {
    
    // keeps track of how many iterations each core
    niter_ = std::vector<int>(ncores);
    
    // track all values to show convergence
    all_betas_ = std::vector<Eigen::VectorXd>(ncores*25);
    for(int i=0; i<ncores*25; i++){
      all_betas_[i] = Eigen::VectorXd::Zero(x.cols(), 1).cast<double>();
    }
    
    // track current beta values for communication
    current_betas_ = std::vector<Eigen::VectorXd>(ncores);
    for(int i=0; i<ncores; i++){
      current_betas_[i] = Eigen::VectorXd::Zero(x.cols(), 1).cast<double>();
    }
    
  }
  
  // link function for logistic regression
  Eigen::VectorXd LogisticFunction(const Eigen::MatrixXd x, const Eigen::VectorXd beta){
    return (1 + (-x*beta).array().exp()).pow(-1);
  }
  
  // solve weighted least squares, subproblem of logistic regression
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
  
  
  // solve a logistic regression problem
  Eigen::VectorXd LogisticRegressionTask(const Eigen::MatrixXd & x, const Eigen::VectorXd & y, int id) {
    Eigen::VectorXd beta = Eigen::VectorXd::Ones(x.cols(), 1).cast<double>();
    Eigen::VectorXd beta_old = beta;
    int n = (int) x.rows();
    
    // save X^T
    Eigen::MatrixXd xT = x.transpose();
    
    double diff = 1;
    int counter = 0;
    double dev_old = 1000;
    double dev_new = 0;
    double dev = 0;
    while(counter < 25 & diff > 1e-8){
      Eigen::VectorXd p = LogisticFunction(x, beta);
      Eigen::VectorXd variance = p.array() * (1 - p.array());
      Eigen::VectorXd modified_response = (variance.array().pow(-1) * (y - p).array());
      Eigen::VectorXd Z = x * beta + modified_response;
      beta = WeightedLS(x, Z, variance);
      Eigen::DiagonalMatrix<double, Eigen::Dynamic> W(variance);

      // compute stopping criteria, matches glm
      dev_new = 0;
      for(int i=0; i<n; i++) {
        dev_new +=  (y[i]*std::log(p[i])) + ((1-y[i])*std::log(1-p[i])) ;
      }
      dev_new *=2;
      // Rcpp::Rcout << dev_new << "\n";
      diff = std::abs(dev_new - dev_old) / (0.1 + std::abs(dev_old));
      dev_old = dev_new;
      beta_old = beta;
      

      // Rcpp::Rcout << counter << "\n";
      all_betas_[id*25 + counter] = beta;
      counter ++;
    }
    niter_[id] = counter;
    return beta;
  }
  
  // solve logistic regression on a partition of whole data
  void PartitionedRegressionTask(int id, int nrows){
    // Rcpp::Rcout << "\nstarting task "<< id <<"\n";
    
    Eigen::MatrixXd x_partition(nrows, x_.cols());
    Eigen::VectorXd y_partition(nrows);
    
    int rownew = 0;
    int start = id*nrows;
    int end = std::min((id+1)*nrows, (int)x_.rows());
    for (int i = start; i < end; i++) {
      x_partition.row(rownew) = x_.row(i);
      y_partition.row(rownew) = y_.row(i);
      rownew++;
    }
    
    Eigen::VectorXd beta = LogisticRegressionTask(x_partition, y_partition, id);
    // Rcpp::Rcout << "\n" << id << "\t" << beta <<"\n";
    current_betas_[id] = beta;
  }
  
  Eigen::VectorXd SolveLR(){
    int nrows_per_task = (int)x_.rows() / ncores_ + 1; 
    
    std::thread workers[ncores_-1];
    if (ncores_ > 1){
      for(int i=0; i<ncores_-1; i++){
        workers[i] = std::thread([&] {Solver::PartitionedRegressionTask(i, nrows_per_task);});
      }
      // Rcpp::Rcout << "\n all tasks started \n";
      PartitionedRegressionTask(0, nrows_per_task);
      for(int i=0; i<ncores_-1; i++){
        // Rcpp::Rcout << "\n about to join worker " << i << "\n";
        workers[i].join();
        // Rcpp::Rcout << "\n worker " << i << " joined \n";
        
      }
    } else {
      PartitionedRegressionTask(0, nrows_per_task);
    }
    
    
    
    // simple average for now 
    Eigen::VectorXd beta_agg = Eigen::VectorXd::Zero(x_.cols(), 1).cast<double>();
    for(int i=0; i<ncores_; i++){
      beta_agg = beta_agg + current_betas_[i];
    }
    
    return beta_agg.array() / ncores_;
  }
};









//' @title Parallel Logistic Regresssion
//' @param x the X matrix in logistic regression
//' @param y the response matrix in logistic regression
//' @param ncores the number of cores to use
//' @export
// [[Rcpp::export]]
Rcpp::List ParLR(const Eigen::MatrixXd & x, const Eigen::VectorXd & y, int ncores=1) {
  Solver s = Solver(x, y, ncores);
  Eigen::VectorXd beta = s.SolveLR();
  SEXP wrap_beta = Rcpp::wrap(beta);
  
  Eigen::MatrixXd all_beta(ncores*25, x.cols());
  for(int i=0; i<ncores*25; i++){
    all_beta.row(i) = s.all_betas_[i].transpose();
  }
  
  // Rcpp::Named("niter") = Rcpp::IntegerVector(*s.niter_)
  return Rcpp::List::create(Rcpp::Named("beta") = Rcpp::NumericVector(wrap_beta),
                            Rcpp::Named("niter") = Rcpp::wrap(s.niter_),
                            Rcpp::Named("all_betas") = Rcpp::wrap(all_beta));
  
}