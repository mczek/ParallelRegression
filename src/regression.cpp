// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-
// [[Rcpp::depends(RcppEigen)]]

#include <RcppEigen.h>
#include <thread>
#include <mutex>
#include <string>
#include <condition_variable> 
#include <atomic> 
#include "Barrier.cpp"

class Solver{
public:
  
  // declare variables shared across threads
  const Eigen::MatrixXd x_;
  const Eigen::VectorXd y_;
  int ncores_;
  int comm_;
  std::vector<int> niter_;
  std::vector<Eigen::VectorXd> all_betas_;
  std::vector<Eigen::VectorXd> current_betas_;
  
  std::vector<int> current_iter_;
  
  Barrier barrier_lock;
  // constructor
  Solver(Eigen::MatrixXd x, Eigen::VectorXd y, int ncores, int comm) : x_(x), y_(y), ncores_(ncores), comm_(comm) {
    // keeps track of how many iterations each core
    niter_ = std::vector<int>(ncores, 0);
    
    // track all values to show convergence
    for(int i=0; i<ncores*25; i++){
      all_betas_.push_back(Eigen::VectorXd::Zero(x.cols(), 1).cast<double>());
    }
    
    // track current beta values for communication
    for(int i=0; i<ncores; i++){
      current_betas_.push_back(Eigen::VectorXd::Zero(x.cols(), 1).cast<double>());
      current_iter_.push_back(0);
    }
    
    new (&barrier_lock) Barrier(ncores_);
  }
  
  Eigen::VectorXd AverageBetaIter(const std::vector<Eigen::VectorXd> beta_vec, int i){
    // Rcpp::Rcout << "about to allocate beta";
    Eigen::VectorXd beta = beta_vec[i];
    // Rcpp::Rcout << "beta allocated";
    for(int id=1; id<ncores_; id++){
      beta += beta_vec[id*25 + i];
    }
    return beta / ncores_;
  }
  
  Eigen::VectorXd AverageBetaCurrent(const std::vector<Eigen::VectorXd> beta_vec, const std::vector<int> iter_vec, int i){
    // Rcpp::Rcout << "about to allocate beta";
    Eigen::VectorXd beta = Eigen::VectorXd::Zero(x_.cols(), 1).cast<double>();
    int wgt = 0;
    // Rcpp::Rcout << "beta allocated";
    for(int id=0; id<ncores_; id++){
      int new_wgt = iter_vec[id];
      if(new_wgt > 0){
        wgt += new_wgt;
        beta += beta_vec[id] * new_wgt;
      }
    }
    if (wgt == 0){
      return beta_vec[i];
    }
    return beta / wgt;
  }
  
  
  // link function for logistic regression
  Eigen::VectorXd LogisticFunction(const Eigen::MatrixXd x, const Eigen::VectorXd beta){
    return (1 + (-x*beta).array().exp()).pow(-1);
  }
  
  // solve weighted least squares, subproblem of logistic regression
  Eigen::VectorXd WeightedLS(const Eigen::MatrixXd & x, const Eigen::VectorXd & y, Eigen::VectorXd w){
    Eigen::VectorXd w_sqrt = w.array().sqrt();
    Eigen::VectorXd w_inv = w.array().pow(-1);
    
    Eigen::DiagonalMatrix<double, Eigen::Dynamic> W_sqrt(w_sqrt);
    Eigen::DiagonalMatrix<double, Eigen::Dynamic> W_inv(w_inv);
    
    Eigen::MatrixXd wgt_x = W_sqrt * x;
    Eigen::VectorXd wgt_y = W_sqrt * y;
    
    // https://stackoverflow.com/questions/58350524/eigen3-select-rows-out-based-on-column-conditions
    Eigen::VectorXi is_selected = (w.array() > 0).cast<int>();
    Eigen::MatrixXd x_new(is_selected.sum(), x.cols());
    Eigen::VectorXd y_new(is_selected.sum());
    int rownew = 0;
    for (int i = 0; i < x.rows(); ++i) {
      if (is_selected[i]) {       
        x_new.row(rownew) = wgt_x.row(i);
        y_new.row(rownew) = wgt_y.row(i);
        rownew++;
      }
    }
    
    return x_new.householderQr().solve(y_new);
  }
  
  // solve a logistic regression problem
  Eigen::VectorXd LogisticRegressionTask(const Eigen::MatrixXd & x, const Eigen::VectorXd & y, int id) {
    // std::string debug = "Hi! I'm thread " + std::to_string(id) + " and I'm getting started\n";
    // Rcpp::Rcout << debug;
    Eigen::VectorXd beta = Eigen::VectorXd::Ones(x.cols(), 1).cast<double>();
    // Eigen::VectorXd beta = Eigen::VectorXd::Random(x.cols(), 1).cast<double>();
    int n = (int) x.rows();
    
    double diff = 1;
    int counter = 0;
    double dev_old = 1000;
    double dev_new = 0;
    
    
    
    while((counter < 25 && diff > 1e-8) || counter <=3){
      Eigen::VectorXd p = LogisticFunction(x, beta);
      Eigen::VectorXd variance = p.array() * (1 - p.array());
      Eigen::VectorXd modified_response = (variance.array().pow(-1) * (y - p).array());
      Eigen::VectorXd Z = x * beta + modified_response;
      beta = WeightedLS(x, Z, variance);
      
      // log state
      current_betas_[id] = beta;
      current_iter_[id] = counter;
      all_betas_[id*25 + counter] = beta;
      
      // compute stopping criteria, matches glm
      dev_new = 0;
      for(int i=0; i<n; i++) {
        if(p[i] != y[i]){
          dev_new +=  (y[i]*std::log(p[i])) + ((1-y[i])*std::log(1-p[i])) ;
        }
      }
      dev_new *=2;
      // Rcpp::Rcout << dev_new << "\n";
      diff = std::abs(dev_new - dev_old) / (0.1 + std::abs(dev_old));
      dev_old = dev_new;
      
      // prevent spurious wake ups
      if(ncores_ > 1 && comm_ == 1 && counter <= 2){
        // Rcpp::Rcout <<"waiting";
        barrier_lock.wait();
        beta = AverageBetaIter(all_betas_, counter);
      }
      
      if(ncores_ > 1 && comm_ == 2 && counter <= 2){
        beta = AverageBetaCurrent(current_betas_, current_iter_, id);
      }
      counter ++;
    }
    
    
    if(niter_[id] == 0){
      niter_[id] = counter;
    }
    return beta;
  }
  
  // solve logistic regression on a partition of whole data
  void PartitionedRegressionTask(int id, int nrows){
    // Rcpp::Rcout << "\n starting task "<< id <<"\n";
    
    int start = id*nrows;
    int end = std::min((id+1)*nrows, (int)x_.rows());
    
    
    Eigen::MatrixXd x_partition(end-start, x_.cols());
    Eigen::VectorXd y_partition(end-start);
    
    int rownew = 0;
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
        // Rcpp::Rcout << std::to_string(i) + "\t";
        workers[i] = std::thread(&Solver::PartitionedRegressionTask, this, i, nrows_per_task);
      }
      // Rcpp::Rcout << "\n all tasks started \n";
      PartitionedRegressionTask(ncores_-1, nrows_per_task);
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
//' @param comm indicates communication method. More details to follow...
//' @export
// [[Rcpp::export]]
Rcpp::List ParLR(const Eigen::MatrixXd & x, const Eigen::VectorXd & y, int ncores=1, int comm=0) {
  Solver s(x, y, ncores, comm);
  Eigen::VectorXd beta = s.SolveLR();
  
  Eigen::MatrixXd all_beta(ncores*25, x.cols());
  for(int i=0; i<ncores*25; i++){
    all_beta.row(i) = s.all_betas_[i].transpose();
  }
  
  // Rcpp::Named("niter") = Rcpp::IntegerVector(*s.niter_)
  return Rcpp::List::create(Rcpp::Named("beta") = Rcpp::wrap(beta),
                            Rcpp::Named("niter") = Rcpp::wrap(s.niter_),
                            Rcpp::Named("all_betas") = Rcpp::wrap(all_beta));
  
}