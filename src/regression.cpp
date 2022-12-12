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
    
    // remove rows with 0 weight
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
    Eigen::VectorXd beta = Eigen::VectorXd::Ones(x.cols(), 1).cast<double>();
    int n = (int) x.rows();
    
    double dev_old = 0;
    if (ncores_ > 1 && comm_ == 3){
      int subset_size = std::max((int)x.cols(), n/ncores_);
      Eigen::MatrixXd small_x(subset_size, x.cols());
      Eigen::VectorXd small_y(subset_size);
      for (int i = 0; i < subset_size; ++i) {
        small_x.row(i) = x.row(i);
        small_y.row(i) = y.row(i);
      }
      
      // iteratively reweight
      Eigen::VectorXd p = LogisticFunction(small_x, beta);
      Eigen::VectorXd variance = p.array() * (1 - p.array());
      Eigen::VectorXd modified_response = (variance.array().pow(-1) * (small_y - p).array());
      Eigen::VectorXd Z = small_x * beta + modified_response;

      // least squares
      beta = WeightedLS(small_x, Z, variance);
      all_betas_[id*25] = beta;
      
      // compute stopping criteria, matches glm
      for(int i=0; i<subset_size; i++) {
        if(p[i] != y[i]){
          dev_old +=  (small_y[i]*std::log(p[i])) + ((1-small_y[i])*std::log(1-p[i])) ;
        }
      }
      dev_old *=2;

      //
      // // wait and communicate
      barrier_lock.wait();
      beta = AverageBetaIter(all_betas_, 0);
    }
    
    double diff = 1;
    int counter = 0;

    double dev_new = 0;
    double eta = 1;
    while((counter < 25 && diff > 1e-8) || counter <=2){
      
      // iteratively reweight 
      Eigen::VectorXd p = LogisticFunction(x, beta);
      Eigen::VectorXd variance = p.array() * (1 - p.array());
      Eigen::VectorXd modified_response = (variance.array().pow(-1) * (y - p).array());
      Eigen::VectorXd Z = x * beta + modified_response;
      
      // least squares
      Eigen::VectorXd beta_new = WeightedLS(x, Z, variance);
      beta  = eta*(beta_new - beta) + beta;
      
      
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
      // if(dev_new < dev_old){
      //   eta /= 2;
      // }
      diff = std::abs(dev_new - dev_old) / (0.1 + std::abs(dev_new));
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
    // get data for current task
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
    
    // solve current task
    Eigen::VectorXd beta = LogisticRegressionTask(x_partition, y_partition, id);
    current_betas_[id] = beta;
  }
  
  Eigen::VectorXd SolveLR(){
    int nrows_per_task = (int)x_.rows() / ncores_ + 1; 
    
    std::thread workers[ncores_-1];
    if (ncores_ > 1){
      // start all threads
      for(int i=0; i<ncores_-1; i++){
        // Rcpp::Rcout << std::to_string(i) + "\t";
        workers[i] = std::thread(&Solver::PartitionedRegressionTask, this, i, nrows_per_task);
      }
      PartitionedRegressionTask(ncores_-1, nrows_per_task);
      
      // wait for all threads to complete
      for(int i=0; i<ncores_-1; i++){
        workers[i].join();
        
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
//' @param comm indicates communication method. 0 indicates no communication, 1 indicates communication with waiting, 2 indicates communication without waiting, 3 indicates communicating once on a subset and proceeding without communication
//' 
//' @returns beta the estimate logistic regression beta
//' @returns niter how many iterations each subproblem took to converge
//' @returns all_betas a list of all betas observed at each iteration. Each 25 row section is another thread with increasing iterations.
//' @export
//' @examples
//' n<- 2000
//' X <- matrix(rnorm(n*2, mean = 0, sd = 0.05), ncol=2)
//'   beta <- c(1,2)
//'   
//'   prob <- 1 / (1 + exp(-X %*% beta))
//'   y <- rbinom(n, 1, prob)
//'   
//'   ParallelRegression::ParLR(X, y, 2, 1)
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