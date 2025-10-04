#include <cmath>
#include <ctime>
#include <iostream>
#include <cstdlib>
#include <string>
#include <pthread.h>
using namespace std;

// #define PI M_PI

typedef struct{
    int thread_id;
    long long int start;
    long long int end;
    long long int in_circle;
} Arg; // 傳給 thread 的參數



void* count_pi(void* args){
    Arg* arg = (Arg*) args;
    long long int in_circle = 0;

    // 每個 thread 使用獨立的 seed
    unsigned int seed = arg->thread_id;

    for(long long int toss = arg->start; toss < arg->end; toss++){
        double x = 2.0 * rand_r(&seed) / (RAND_MAX + 1.0) - 1.0;
        double y = 2.0 * rand_r(&seed) / (RAND_MAX + 1.0) - 1.0;
        if( (x*x + y*y) <= 1) in_circle++;
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