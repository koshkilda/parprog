/* результаты с mutex
4 потока
Сумма чисел: 45011704
Время: 1.07331 секунд

8 потоков
Сумма чисел: 45011704
Время: 1.9645 секунд

16 потоков
Сумма чисел: 45011704
Время: 3.63433 секунд

реализация через mutex немного дольше, чем реализация через atomic
*/


#include <iostream>
#include <thread>
#include <boost/thread.hpp>
#include <mutex>
#include <vector>
#include <chrono>
#include <cmath>

std::mutex mtx;
long long sum = 0;

int summer(std::vector<int> v, int begin, int end) { 
    for(int i = begin; i <= end; ++i) { 
        std::lock_guard<std::mutex> lock(mtx);
        sum += v[i];
    }
    return sum;
}

int main() {
    int N;
    std::cin >> N; 
    long long num_elements = pow(10, 7);

    std::vector<int> v;
    for (int i = 0; i < num_elements; i++) { 
        v.emplace_back(rand() % 10); 
    }

    auto start = std::chrono::high_resolution_clock::now();
    int v_partsize = num_elements / N;
    std::vector<boost::thread> threads;
    for (int i = 1; i <= N - 1; i++) { 
        threads.emplace_back(boost::thread(summer, v, v_partsize * (i - 1), v_partsize * i - 1)); 
    }
    threads.emplace_back(boost::thread(summer, v, v_partsize * (N - 1), v_partsize * N + num_elements % N - 1)); 
    for (auto &t : threads) t.join();
    std::cout << "Сумма чисел: " << sum << std::endl;

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Время: " << std::chrono::duration<double>(end - start).count() << " секунд\n";

    return 0;
}