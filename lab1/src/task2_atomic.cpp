/* результаты с atomic
4 потока
Сумма чисел: 45011704
Время: 0.915134 секунд

8 потоков
Сумма чисел: 45011704
Время: 1.90922 секунд

16 потоков
Сумма чисел: 45011704
Время: 3.62456 секунд


без синхронизации 
4 поток
Сумма чисел: 45011704
Время: 0.994451 секунд

8 потоков
Сумма чисел: 45011704
Время: 1.91913 секунд

16 потоков
Сумма чисел: 45011704
Время: 3.64056 секунд

без синхронизации стабильно немного дольше, чем с atomic
*/


#include <iostream>
#include <thread>
#include <boost/thread.hpp>
#include <atomic>
#include <vector>
#include <chrono>
#include <cmath>

std::atomic<long long> sum(0);
// long long sum = 0; // для реализации без синхронизации; тоже работает

int summer(std::vector<int> v, int begin, int end) { 
    for(int i = begin; i <= end; ++i) { 
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
