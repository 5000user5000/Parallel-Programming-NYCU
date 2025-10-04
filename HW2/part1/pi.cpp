#include <cmath>
#include <ctime>
#include <iostream>
#include <cstdlib>
#include <string>
#include <pthread.h>
#include <stdint.h>
using namespace std;

// #define PI M_PI

typedef struct{
    int thread_id;
    long long int start;
    long long int end;
    long long int in_circle;
} Arg; // 傳給 thread 的參數

// 快速的 xorshift128+ PRNG
struct xorshift128p_state {
    uint64_t s[2];
};

static inline uint64_t xorshift128p(struct xorshift128p_state *state) {
    uint64_t s1 = state->s[0];
    const uint64_t s0 = state->s[1];
    state->s[0] = s0;
    s1 ^= s1 << 23;
    state->s[1] = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);
    return state->s[1] + s0;
}

void* count_pi(void* args){
    Arg* arg = (Arg*) args;
    long long int in_circle = 0;

    // 初始化 xorshift128+ state，每個 thread 使用不同的 seed
    struct xorshift128p_state state;
    state.s[0] = arg->thread_id + 1;
    state.s[1] = (arg->thread_id + 1) * 0x123456789ABCDEF;

    // 預先計算常數
    const double scale = 2.0 / 4294967296.0; // 2.0 / 2^32

    for(long long int toss = arg->start; toss < arg->end; toss++){
        // 一個 64-bit 隨機數拆成兩個 32-bit 給 x 和 y
        uint64_t r = xorshift128p(&state);
        uint32_t rx = (uint32_t)(r & 0xFFFFFFFF);
        uint32_t ry = (uint32_t)(r >> 32);

        double x = rx * scale - 1.0;
        double y = ry * scale - 1.0;

        // Branchless
        in_circle += (x*x + y*y <= 1.0);
    }
    arg->in_circle = in_circle;
    pthread_exit((void *)0);
}



int main(int argc,char *argv[]){

    // 處理參數
    if(argc!=3){ // || isdigit(argv[1]) || isdigit(argv[2])
        cout<<"Arg Error: Please use ./pi.out <num_thread> <num_toss>"<<endl;
        return -1;
    }
    int threads = stoi(argv[1]);
    long long int number_of_tosses = stol(argv[2]);

    if(threads <= 0 || number_of_tosses <= 0){
        cout<<"Arg Error: Please use ./pi.out <num_thread> <num_toss>"<<endl;
        return -1;
    }

    pthread_t* thread_handles = new pthread_t[threads];
    Arg* arg = new Arg[threads];
    long long int chunk_size = number_of_tosses / threads;
    
    for(int i=0; i<threads; i++){
        arg[i].thread_id = i;
        arg[i].start = i * chunk_size;
        if(i == threads-1) arg[i].end = number_of_tosses;
        else arg[i].end = (i+1) * chunk_size;
        arg[i].in_circle = 0;
        pthread_create(&thread_handles[i], NULL, count_pi, (void*)&arg[i]);
    }

    for(int i=0; i<threads; i++){
        pthread_join(thread_handles[i], NULL);
    }

    long long int number_in_circle = 0;
    for(int i=0; i<threads; i++){
        number_in_circle += arg[i].in_circle;
    }
    delete[] thread_handles;
    delete[] arg;
    // cout<<"number_in_circle = "<<number_in_circle<<endl;
    // cout<<"number_of_tosses = "<<number_of_tosses<<endl;

    double pi_estimate = 4.0 * number_in_circle / number_of_tosses;
    cout<<"Estimated PI = "<<pi_estimate<<endl;

    return 0;
}