#include <cmath>
#include <ctime>
#include <iostream>
#include <cstdlib>
using namespace std;

// #define PI M_PI

static inline double Random(){
    return 2.0 *rand() / (RAND_MAX + 1.0) - 1.0 ;
}

int main(){
    srand( time(NULL) );

    long long int number_of_tosses = 1e8;
    long long int number_in_circle = 0;
    for(long long int toss = 1; toss <= number_of_tosses; toss++){
        double x = Random() ;
        double y = Random();
        if( (x*x + y*y) <= 1) number_in_circle++;
    }
    double pi_estimate = 4.0 * number_in_circle / number_of_tosses;
    cout<<"Estimated PI = "<<pi_estimate<<endl;
    return 0;
}