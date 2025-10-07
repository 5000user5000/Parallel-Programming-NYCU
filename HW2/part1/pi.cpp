#include <cmath>
#include <ctime>
#include <iostream>
#include <cstdlib>
#include <string>
#include <pthread.h>
#include <stdint.h>
#include <immintrin.h>
#define __AVX2_AVAILABLE__
#include "include/SIMDInstructionSet.h"
#include "include/Xoshiro256Plus.h"
using namespace std;
using namespace SEFUtility::RNG;

// #define PI M_PI

typedef struct{
    int thread_id;
    long long int start;
    long long int end;
    long long int in_circle;
} Arg; // 傳給 thread 的參數

void* count_pi(void* args){
    Arg* arg = (Arg*) args;

    // 使用 AVX2 版本的 Xoshiro256Plus，每個 thread 不同 seed
    Xoshiro256Plus<SIMDInstructionSet::AVX2> rng((arg->thread_id + 1) * 0x9e3779b97f4a7c15ULL);

    long long int total = arg->end - arg->start;
    long long int local_sum = 0;

    // AVX2 常數
    const __m256 rand_scale = _mm256_set1_ps(4.6566129e-10f);  // 1.0f / 2^31
    const __m256 one = _mm256_set1_ps(1.0f);

    // 主循環：每次處理 16 個點（unroll 2x）
    long long int main_loop = total / 16;

    for(long long int i = 0; i < main_loop; i++){
        // 第一批 8 個點
        __m256i rand_x1 = rng.next4().operator __m256i();
        __m256i rand_y1 = rng.next4().operator __m256i();

        __m256 x1 = _mm256_cvtepi32_ps(rand_x1);
        x1 = _mm256_mul_ps(x1, rand_scale);

        __m256 y1 = _mm256_cvtepi32_ps(rand_y1);
        y1 = _mm256_mul_ps(y1, rand_scale);

        x1 = _mm256_mul_ps(x1, x1);
        y1 = _mm256_mul_ps(y1, y1);
        __m256 sum1 = _mm256_add_ps(x1, y1);
        __m256 cmp1 = _mm256_cmp_ps(sum1, one, _CMP_LE_OQ);

        // 第二批 8 個點
        __m256i rand_x2 = rng.next4().operator __m256i();
        __m256i rand_y2 = rng.next4().operator __m256i();

        __m256 x2 = _mm256_cvtepi32_ps(rand_x2);
        x2 = _mm256_mul_ps(x2, rand_scale);

        __m256 y2 = _mm256_cvtepi32_ps(rand_y2);
        y2 = _mm256_mul_ps(y2, rand_scale);

        x2 = _mm256_mul_ps(x2, x2);
        y2 = _mm256_mul_ps(y2, y2);
        __m256 sum2 = _mm256_add_ps(x2, y2);
        __m256 cmp2 = _mm256_cmp_ps(sum2, one, _CMP_LE_OQ);

        // 累加結果
        local_sum += __builtin_popcount(_mm256_movemask_ps(cmp1));
        local_sum += __builtin_popcount(_mm256_movemask_ps(cmp2));
    }

    // 處理剩餘的點
    long long int remain = total % 16;
    for(long long int i = 0; i < remain; i++){
        uint64_t r = rng.next();
        float x = (int32_t)(r) * 4.6566129e-10f;
        float y = (int32_t)(r >> 32) * 4.6566129e-10f;
        local_sum += (x*x + y*y <= 1.0f);
    }

    arg->in_circle = local_sum;
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