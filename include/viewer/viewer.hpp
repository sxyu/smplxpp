#pragma once
#ifndef SMPL_VIEWER_A821B138_6EE4_4B99_9B92_C87508B97197
#define SMPL_VIEWER_A821B138_6EE4_4B99_9B92_C87508B97197

#include "smpl.hpp"

namespace smpl {

class Viewer {
public:
    Viewer();
    ~Viewer();

    void spin();
private:
    void* p_window;
};

}  // namespace smpl
#endif  // ifndef SMPL_VIEWER_A821B138_6EE4_4B99_9B92_C87508B97197
