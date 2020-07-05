cmake_minimum_required( VERSION 2.8 )

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
if (POLICY CMP0048)
    cmake_policy(SET CMP0048 NEW)
endif (POLICY CMP0048)
if (POLICY CMP0072)
    cmake_policy(SET CMP0072 NEW)
endif (POLICY CMP0072)


project( smpl_cpp )
option( USE_FFAST_MATH "Enable ffast-math compiler flag, may cause numerical problems" ON )
option( BUILD_VIEWER "Build OpenGL-based viewer" ON )
option( USE_SYSTEM_EIGEN "Use system Eigen rather than the included Eigen submodule if available" ON )

set( INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include" )
set( SRC_DIR "${PROJECT_SOURCE_DIR}/src" )
set( VENDOR_DIR "${PROJECT_SOURCE_DIR}/3rdparty" )
set( CNPY_DIR "${VENDOR_DIR}/cnpy" )
set( STB_DIR "${VENDOR_DIR}/stb" )
set( IMGUI_DIR "${VENDOR_DIR}/imgui" )
set( GLFW_DIR "${VENDOR_DIR}/glfw" )
set( GLEW_DIR "${VENDOR_DIR}/glew" )
# set( GLM_DIR "${VENDOR_DIR}/glm" )
set( ZLIB_DIR "${VENDOR_DIR}/zlib" )
set( EIGEN_DIR "${VENDOR_DIR}/eigen" )

set( CMAKE_CXX_STACK_SIZE "10000000" )
set( CMAKE_CXX_STANDARD 14 )
set( CMAKE_CXX_STANDARD_REQUIRED ON)
set( CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake_modules" )

set( PROJ_NAME "smpl_cpp" )
set( MESHVIEW_NAME "meshview" )

if ( CMAKE_COMPILER_IS_GNUCXX OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang") )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated -Wno-deprecated-declarations -O3 -funroll-loops -g" )
    if( ${USE_FFAST_MATH} )
        set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math" )
    endif()
elseif( MSVC )
    if( ${USE_FFAST_MATH} )
        set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /fp:fast" )
    endif()
endif ( )

# Git submodule auto update
# https://cliutils.gitlab.io/modern-cmake/chapters/projects/submodule.html
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    # Update submodules as needed
    option(GIT_SUBMODULE "Check submodules during build" ON)
    if(GIT_SUBMODULE)
        message(STATUS "Submodule update")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
        endif()
    endif()
endif()

if(NOT EXISTS "${GLFW_DIR}/CMakeLists.txt")
   message(FATAL_ERROR "A submodule as not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()

file(GLOB_RECURSE SOURCES ${SRC_DIR}/smpl/*.cpp)
file(GLOB_RECURSE HEADERS ${INCLUDE_DIR}/smpl/*.cpp)
file(GLOB_RECURSE MESHVIEW_SOURCES ${SRC_DIR}/meshview/*.cpp)
file(GLOB_RECURSE MESHVIEW_HEADERS ${INCLUDE_DIR}/meshview/*.cpp)

set( VENDOR_SOURCES ${CNPY_DIR}/cnpy.cpp)
set( MESHVIEW_VENDOR_SOURCES ${STB_DIR}/stb_image.cpp )

set( DEPENDENCIES )

# OpenMP
find_package(OpenMP QUIET)
if (OPENMP_FOUND)
    message(STATUS "Using OpenMP for parallelization")
    set( DEPENDENCIES ${DEPENDENCIES} OpenMP::OpenMP_CXX )
else()
    message(STATUS "OpenMP not found, CPU performance may suffer")
endif()

# CUDA
find_package(CUDA QUIET)

include_directories(
    ${INCLUDE_DIR}
    ${CNPY_DIR}
    ${CUDA_INCLUDE_DIRS}
)

if (BUILD_VIEWER)
    include_directories(
        ${STB_DIR}
        ${IMGUI_DIR}
        ${GLEW_DIR}/include
        ${GLFW_DIR}/include
    )

    # Viewer dependencies
    SET(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "GLFW example" FORCE)
    SET(GLFW_BUILD_TESTS OFF CACHE BOOL "GLFW tests" FORCE)
    SET(GLFW_BUILD_DOCS OFF CACHE BOOL "GLFW docs" FORCE)
    SET(GLFW_INSTALL OFF CACHE BOOL "GLFW install" FORCE)
    add_subdirectory(${GLFW_DIR})

    # glew library
    add_library(glew STATIC
        ${GLEW_DIR}/src/glew.c
        ${GLEW_DIR}/include
    )
    target_link_libraries(glew ${GLFW_LIBRARIES})
    set_target_properties(glew
        PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/3rdparty/glew")

    # OpenGL
    find_package(OpenGL REQUIRED)

    set(
        DEPENDENCIES
        glew
        OpenGL::GL
        glfw
        ${GLFW_LIBRARIES}
    )
endif(BUILD_VIEWER)

# Eigen
if ( USE_SYSTEM_EIGEN )
    find_package(Eigen3 QUIET NO_MODULE)
    if (TARGET Eigen3::Eigen)
        message(STATUS "Using system Eigen")
        set (DEPENDENCIES ${DEPENDENCIES} Eigen3::Eigen)
    else()
        message(STATUS "System Eigen not found, using vendor Eigen")
        include_directories(${EIGEN_DIR})
    endif()
else()
    message(STATUS "Using vendor Eigen")
    include_directories(${EIGEN_DIR})
endif()

# Zlib
find_package(ZLIB QUIET)
if (ZLIB_FOUND)
    message( STATUS "Using system zlib" )
    set( DEPENDENCIES ${DEPENDENCIES} ZLIB::ZLIB )
else()
    message( STATUS "Using included zlib." )
    add_subdirectory(${ZLIB_DIR})
    include_directories(${ZLIB_DIR}/include)
    set( DEPENDENCIES ${DEPENDENCIES} zlibstatic )
endif()

add_library( ${PROJ_NAME} STATIC ${SOURCES} ${VENDOR_SOURCES} )
set_target_properties( ${PROJ_NAME} PROPERTIES OUTPUT_NAME "smplx" )
target_link_libraries( ${PROJ_NAME} ${DEPENDENCIES} )
install(TARGETS ${PROJ_NAME} DESTINATION lib)

add_executable( example main_example.cpp )
target_link_libraries( example ${DEPENDENCIES} ${PROJ_NAME} )
set_target_properties( example PROPERTIES OUTPUT_NAME "smplx-example" )
install(TARGETS example DESTINATION bin)

if ( BUILD_VIEWER )
    add_library( ${MESHVIEW_NAME} STATIC ${MESHVIEW_SOURCES} ${MESHVIEW_VENDOR_SOURCES} )
    set_target_properties( ${MESHVIEW_NAME} PROPERTIES OUTPUT_NAME "meshview" )
    target_link_libraries( ${MESHVIEW_NAME} ${DEPENDENCIES} )
    install(TARGETS ${MESHVIEW_NAME} DESTINATION lib)

    add_executable( viewer main_viewer.cpp )
    target_link_libraries( viewer ${DEPENDENCIES} ${PROJ_NAME} ${MESHVIEW_NAME} )
    set_target_properties( viewer PROPERTIES OUTPUT_NAME "smplx-viewer" )

    install(TARGETS viewer DESTINATION bin)
endif( BUILD_VIEWER )

if ( MSVC )
    set_property(TARGET ${PROJ_NAME} APPEND PROPERTY LINK_FLAGS /DEBUG)
    set_property(TARGET viewer APPEND PROPERTY LINK_FLAGS /DEBUG)
endif ( MSVC )

if(WIN32)
    if (ENABLE_GUI)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /FORCE:MULTIPLE")
        # Filament is built with /MT[d], but by default CMake automatically
        # sets "/MD" and "/MDd". They can't coexist, so we need to replace them.
        string(REPLACE "/MD" "/MT" CMAKE_C_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE})
        string(REPLACE "/MDd" "/MTd" CMAKE_C_FLAGS_DEBUG ${CMAKE_C_FLAGS_DEBUG})
        string(REPLACE "/MD" "/MT" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
        string(REPLACE "/MDd" "/MTd" CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
    endif()
elseif(CYGWIN)
elseif(APPLE)
elseif(UNIX)
    target_link_libraries( ${PROJ_NAME} -pthread )
    target_link_libraries( example -pthread )
    if (BUILD_VIEWER)
        target_link_libraries( ${MESHVIEW_NAME} -pthread )
        target_link_libraries( viewer -pthread )
    endif()
    add_definitions(-DUNIX)
    add_compile_options(-Wno-deprecated-declarations)
    add_compile_options(-Wno-unused-result)
endif(WIN32)

add_definitions( -DGLEW_STATIC )

source_group( "Header Files" FILES ${HEADERS} ${MESHVIEW_HEADERS} )
source_group( "Source Files" FILES ${SOURCES} ${MESHVIEW_SOURCES} )
source_group( "Vendor Source Files" FILES ${VENDOR_SOURCES} ${MESHVIEW_VENDOR_SOURCES} )
