#ifndef MISC_FUNCTIONS_H
#define MISC_FUNCTIONS_H

#include <array>
#include <stdexcept>  // for std::runtime_error
#include <algorithm>  // for std::min

inline int idx_2d_to_1d(int row, int col, int cols) {
    return row * cols + col;
}

inline std::array<int, 2> idx_1d_to_2d(int index, int cols) {
    int row = index / cols;
    int col = index % cols;
    return {row, col};
}

inline void check_CFL(double alpha, double dx, double dy, double dt) {
    double min_dxdy = std::min(dx, dy);
    double C = alpha*dt/(min_dxdy*min_dxdy); // CFL condition

    if (C > 0.1) {
        throw std::runtime_error("CFL condition not fulfilled: C = " + std::to_string(C));
    }
}

#endif