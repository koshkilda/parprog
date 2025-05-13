#include <iostream>
#include <thread>
#include <boost/thread.hpp>
#include <mutex>
#include <vector>
#include <chrono>
#include <cmath>

std::mutex mtx;
int count = 0;

int counter(std::vector<int> v, int begin, int end) { 
    int    k     = 0; 
    double last_div; 

    for(int i = begin; i <= end; ++i) { 
        last_div = sqrt(v[i]); 
        for (int j = 2; j <= last_div; ++j) { 
            if (v[i] % j == 0) { 
                k++; 
                break;
            }
        }
        std::lock_guard<std::mutex> lock(mtx);
        (k == 0) ? count++ : k = 0; 
    }
    return count;
}

int main() {
    int N;
    int K;
    std::cin >> N; 
    std::cin >> K; 
    N = N - 3;

    std::vector<int> v;
    for (int i = 0; i < N; i++) { 
        v.emplace_back(i + 4); 
    }

    auto start = std::chrono::high_resolution_clock::now();
    int v_partsize = N / K; 
    std::vector<boost::thread> threads;
    for (int i = 1; i <= K - 1; i++) { // 1, 2, 3
        threads.emplace_back(boost::thread(counter, v, v_partsize * (i - 1), v_partsize * i - 1)); 
    }
    threads.emplace_back(boost::thread(counter, v, v_partsize * (K - 1), v_partsize * K + N % K - 1)); 
    for (auto &t : threads) t.join();
    std::cout << "Простых чисел: " << count + 2 << std::endl;

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Время: " << std::chrono::duration<double>(end - start).count() << " секунд\n";

    return 0;
}