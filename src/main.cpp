#include <cuda_runtime.h>
#include <iostream>
#include "test_utils.h"



int main(){
    

    bool ExpSO3_succeed = MultipleTestsExpSO3();
    if(ExpSO3_succeed){
        std::cout << "ExpSO3 test SUCCEED." << std::endl;
    } else {
        std::cout << "ExpSO3 test FAILED." << std::endl;
    }

    bool LogSO3_succeed = MultipleTestsLogSO3();
    if(LogSO3_succeed){
        std::cout << "LogSO3 test SUCCEED." << std::endl;
    } else {
        std::cout << "LogSO3 test FAILED." << std::endl;
    }

    bool Jacobians_succeed = MultipleTestsJacobiansSO3Kernel();
    if(Jacobians_succeed){
        std::cout << "Jacobians SO3 test SUCCEED." << std::endl;
    } else {
        std::cout << "Jacobians SO3 test FAILED." << std::endl;
    }


    bool ExpSE3_succeed = MultipleTestsSE3();
    if(ExpSE3_succeed){
        std::cout << "SE3 test SUCCEED." << std::endl;
    } else {
        std::cout << "SE3 test FAILED." << std::endl;
    }


    return 0;
}