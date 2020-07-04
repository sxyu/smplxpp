#include "util.hpp"

#include <fstream>

namespace smpl {
namespace util {

std::string find_data_file(const std::string& data_path) {
    static const std::string TEST_PATH = "data/models/smplx/uv.txt";
    static const int MAX_LEVELS = 3;
    static std::string data_dir_saved = "\n";
    if (data_dir_saved == "\n") {
        data_dir_saved.clear();
        const char * env = std::getenv("SMPLX_DIR");
        if (env) {
            // use environmental variable if exists and works
            data_dir_saved = env;

            // auto append slash
            if (!data_dir_saved.empty() && data_dir_saved.back() != '/' && data_dir_saved.back() != '\\')
                data_dir_saved.push_back('/');

            std::ifstream test_ifs(data_dir_saved + TEST_PATH);
            if (!test_ifs) data_dir_saved.clear();
        }

        // else check current directory and parents
        if (data_dir_saved.empty()) {
            for (int i = 0; i < MAX_LEVELS; ++i) {
                std::ifstream test_ifs(data_dir_saved + TEST_PATH);
                if (test_ifs) break;
                data_dir_saved.append("../");
            }
        }

        data_dir_saved.append("data/");
    }
    return data_dir_saved + data_path;
}
}  // namespace util
}  // namespace smpl
