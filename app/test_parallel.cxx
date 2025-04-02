#include <iostream>
#include <thread>
#include <chrono>
#include <future>
#include <execution>

void task(int param){
    // std::thread t1(task, 10); t1.join();
    for(int i{}; i<param; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    return;
}

void calculate(int& x){
    std::this_thread::sleep_for(std::chrono::seconds(3));
    x*= x;
}

int main(){

    // std::cout << "M1" << std::endl;
    // std::thread t1(task, 10);
    // std::thread t2(task, 10);
    // std::thread t3(task, 10);
    // t1.join();
    // t2.join();
    // t3.join();

    // std::cout << "M2" << std::endl;
    // auto f1 = std::async(std::launch::async, calculate, 10);
    // auto f2 = std::async(std::launch::async, calculate, 20);
    // auto f3 = std::async(std::launch::async, calculate, 30);
    // int r1 = f1.get();
    // int r2 = f2.get();
    // int r3 = f3.get();
    // std::cout << "r1: " << r1 << " r2: " << r2 << " r3: " << r3 << std::endl;

    std::cout << "M3" << std::endl;  // 简单好用的方法！！！
    std::vector<int> data{1,2,3,4,5,6,7,8,9,10};
    std::for_each(std::execution::seq, data.begin(), data.end(), [](int& x){calculate(x);});
    for(const auto& i:data)
        std::cout << i << " ";


}