// inspiration from https://github.com/kirksaunders/barrier/blob/master/barrier.hpp

#include <condition_variable>
#include <mutex>


class Barrier{
private:
  int nthreads_;
  int nwaiting_;
  int use_count_;
  std::mutex mtx_;
  std::condition_variable cv_;
  
public:
  Barrier() {}
  
  Barrier(int nthreads) : nthreads_(nthreads){
    nwaiting_ = 0;
    use_count_ = 0;
  }
  
  void wait(){
    std::unique_lock<std::mutex> lck(mtx_);
    int current_use = use_count_;
    if(++nwaiting_ == nthreads_){
      nwaiting_ = 0;
      use_count_++;
      cv_.notify_all();
    } else {
      cv_.wait(lck, [this, current_use]{return current_use != use_count_;});
    }
  }
};