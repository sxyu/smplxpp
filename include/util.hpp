#pragma once
#ifndef SMPL_UTIL_63B0803D_E0C7_4529_A796_9F6ED269E89F
#define SMPL_UTIL_63B0803D_E0C7_4529_A796_9F6ED269E89F

#define _SMPL_ASSERT(x) do { \
    auto tx = (x); \
    if (!tx) { \
    std::cout << "Assertion FAILED: \"" << #x << "\" (" << tx << \
        ")\n  at " << __FILE__ << " line " << __LINE__ <<"\n"; \
    std::exit(1); \
}} while(0)
#define _SMPL_ASSERT_EQ(x, y) do {\
    auto tx = x; auto ty = y; \
    if (tx != ty) { \
    std::cout << "Assertion FAILED: " << #x << " != " << #y << " (" << tx << " != " << ty << \
        ")\n  at " << __FILE__ << " line " << __LINE__ <<"\n"; \
    std::exit(1); \
}} while(0)

#include <chrono>
#define _SMPL_BEGIN_PROFILE auto start = std::chrono::high_resolution_clock::now()
#define _SMPL_PROFILE(x) do{printf("%s: %f ns\n", #x, std::chrono::duration<double, std::nano>(std::chrono::high_resolution_clock::now() - start).count()); start = std::chrono::high_resolution_clock::now(); }while(false)
#define _SMPL_PROFILE_STEPS(x,stp) do{printf("%s: %f ns / step\n", #x, std::chrono::duration<double, std::nano>(std::chrono::high_resolution_clock::now() - start).count()/(stp)); start = std::chrono::high_resolution_clock::now(); }while(false)

#endif  // ifndef SMPL_UTIL_63B0803D_E0C7_4529_A796_9F6ED269E89F
