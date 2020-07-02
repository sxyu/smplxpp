#include <cstdint>
#include <iostream>

#include "human.hpp"

int32_t main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Expected 2 arg";
        return 1;
    }
    using smpl::Model;
    using smpl::Pose;
    Model model(argv[1]);

    return 0;
}
