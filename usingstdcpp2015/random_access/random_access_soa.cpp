/* usingstdcpp2015: random access performance, AOS vs SOA.
 *
 * Copyright 2015 Joaquin M Lopez Munoz.
 * Distributed under the Boost Software License, Version 1.0.
 * (See accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 */
 
#include <algorithm>
#include <array>
#include <chrono>
#include <numeric> 
    
std::chrono::high_resolution_clock::time_point measure_start,measure_pause;
        
template<typename F>
double measure(F f)
{
  using namespace std::chrono;
        
  static const int              num_trials=10;
  static const milliseconds     min_time_per_trial(200);
  std::array<double,num_trials> trials;
  volatile decltype(f())        res; /* to avoid optimizing f() away */
        
  for(int i=0;i<num_trials;++i){
    int                               runs=0;
    high_resolution_clock::time_point t2;
        
    measure_start=high_resolution_clock::now();
    do{
      res=f();
      ++runs;
      t2=high_resolution_clock::now();
    }while(t2-measure_start<min_time_per_trial);
    trials[i]=duration_cast<duration<double>>(t2-measure_start).count()/runs;
  }
  (void)(res); /* var not used warn */
        
  std::sort(trials.begin(),trials.end());
  return std::accumulate(
    trials.begin()+2,trials.end()-2,0.0)/(trials.size()-4)*1E6;
}
 
template<typename Size,typename F>
double measure(Size n,F f)
{
  return measure(f)/n;
}

void pause_timing()
{
  measure_pause=std::chrono::high_resolution_clock::now();
}
        
void resume_timing()
{
  measure_start+=std::chrono::high_resolution_clock::now()-measure_pause;
}

#include <algorithm>
#include <iostream>
#include <random>

struct particle
{
  int x,y,z;
};

using particle_aos=std::vector<particle>;

particle_aos create_particle_aos(int n)
{
  particle_aos res;
  res.reserve(n);
  for(int i=0;i<n;++i)res.push_back({i,i+1,i+2});
  return res;
}

struct particle_soa
{
  std::vector<int> x,y,z;
};

particle_soa create_particle_soa(int n)
{
  particle_soa res;
  res.x.reserve(n);
  res.y.reserve(n);
  res.z.reserve(n);
  for(int i=0;i<n;++i){
    res.x.push_back(i);
    res.y.push_back(i+1);
    res.z.push_back(i+2);
  }
  return res;
}

int main(int argc, char* argv[])
{
  if (argc < 3) {
    std::cout << "Usage: ./random_access_soa ${data_size} ${repeat_time}\n";
    return 0;
  }
  std::size_t data_size, repeat_time;
  data_size = std::stoi(argv[1]);
  repeat_time = std::stoi(argv[2]);
  double time_avg = 0;

  std::cout<<"n;soa"<<std::endl;
    
  for(std::size_t i=0;i<repeat_time;i+=1) {
    std::cout<<data_size<<";";
    {
      auto ps=create_particle_soa(data_size);
      double now_time = measure(data_size,[&](){
        std::mt19937                    gen;
        std::uniform_int_distribution<> rnd(0,data_size-1);
        long int                        res=0;
        for(std::size_t i=0;i<data_size;++i){
          auto idx=rnd(gen);
          res+=ps.x[idx]+ps.y[idx]+ps.z[idx];
        }
        return res;
      });
      time_avg += now_time;
      std::cout << now_time << "\n";
    }
  }
  time_avg /= repeat_time;
  std::cout << "Avg: " << time_avg << "\n";
}
