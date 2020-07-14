cmake_minimum_required( VERSION 3.11 )

if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting build type to Release")
    set(CMAKE_BUILD_TYPE Release)
endif()
if (POLICY CMP0048)
    cmake_policy(SET CMP0048 NEW)
endif (POLICY CMP0048)
if (POLICY CMP0072)
    cmake_policy(SET CMP0072 NEW)
endif (POLICY CMP0072)

project( smplxpp )
option( SMPLX_USE_FFAST_MATH "Enable ffast-math compiler flag, may cause numerical problems" ON )
option( SMPLX_BUILD_VIEWER "Build OpenGL-based viewer" ON )
option( SMPLX_USE_SYSTEM_EIGEN "Use system Eigen rather than the included Eigen submodule if available" OFF )
option( SMPLX_USE_CUDA "Use cuda if available" ON )

set( INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include" )
set( SRC_DIR "${PROJECT_SOURCE_DIR}/src" )
set( VENDOR_DIR "${PROJECT_SOURCE_DIR}/3rdparty" )
set( MESHVIEW_DIR "${VENDOR_DIR}/meshview" )
set( CNPY_DIR "${VENDOR_DIR}/cnpy" )
set( ZLIB_DIR "${VENDOR_DIR}/zlib" )
set( EIGEN_DIR "${VENDOR_DIR}/eigen" )

set( CMAKE_CXX_STACK_SIZE "10000000" )
set( CMAKE_CXX_STANDARD 17 )
set( CMAKE_CXX_STANDARD_REQUIRED ON)
set( CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake_modules" )

if (SMPLX_USE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
endif()
if (SMPLX_USE_CUDA AND CMAKE_CUDA_COMPILER)
    set(SMPLX_CUDA_ENABLED ON)
    set(_SMPLX_CUDA_ENABLED_ "")
    enable_language(CUDA)
    message(STATUS "CUDA enabled")
    set( CMAKE_CUDA_STANDARD 14 )
    set( CMAKE_CUDA_STANDARD_REQUIRED ON)
    set( CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  -g -Xcudafe \"--display_error_number --diag_suppress=3057 --diag_suppress=3058 --diag_suppress=3059 --diag_suppress=3060\" -lineinfo")
else()
    message(STATUS "CUDA disabled")
    set(SMPLX_CUDA_ENABLED OFF)
    set(_SMPLX_CUDA_ENABLED_ "//")
endif()

set( PROJ_NAME "smplxpp" )

if ( CMAKE_COMPILER_IS_GNUCXX OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang") )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated -Wno-deprecated-declarations -O3 -funroll-loops -g" )
    if( ${SMPLX_USE_FFAST_MATH} )
        set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math" )
    endif()
elseif( MSVC )
    if( ${SMPLX_USE_FFAST_MATH} )
        set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /fp:fast" )
    endif()
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT /GLT /Ox")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /MTd")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /MT /GLT")
    if (SMPLX_USE_CUDA)
        set( CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -Xcompiler=\"/MT\"" )
    endif()
    # Silence deprecation warning coming from Eigen; silence secure warning
    add_definitions(-D_SILENCE_CXX17_RESULT_OF_DEPRECATION_WARNING -D_CRT_SECURE_NO_WARNINGS)
endif ( )

# Git submodule auto update
# https://cliutils.gitlab.io/modern-cmake/chapters/projects/submodule.html
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    # Update submodules as needed
    option(GIT_SUBMODULE "Check submodules during build" ON)
    if(GIT_SUBMODULE)
        message(STATUS "Running update --init --recursive")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
        endif()
        message(STATUS "Submodule update done")
    endif()
endif()

if(NOT EXISTS "${MESHVIEW_DIR}/CMakeLists.txt")
   message(FATAL_ERROR "A submodule as not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()

file(GLOB_RECURSE SOURCES ${SRC_DIR}/*.cpp)
file(GLOB_RECURSE HEADERS ${INCLUDE_DIR}/smplx/*.cpp)

file(GLOB IMGUI_SOURCES ${IMGUI_DIR}/*.cpp)

set( VENDOR_SOURCES ${CNPY_DIR}/cnpy.cpp)

if ( SMPLX_CUDA_ENABLED )
    file(GLOB_RECURSE SOURCES_CUDA ${SRC_DIR}/cuda/*.cu)
    set (SOURCES ${SOURCES} ${SOURCES_CUDA})
endif ( SMPLX_CUDA_ENABLED )

# ensure M_PI, etc available
add_definitions( -DGLEW_STATIC -D_USE_MATH_DEFINES )

set( DEPENDENCIES )

#OpenMP
# find_package(OpenMP QUIET)
# if (OPENMP_FOUND)
#     message(STATUS "Using OpenMP for parallelization")
#     set( DEPENDENCIES ${DEPENDENCIES} OpenMP::OpenMP_CXX )
#     set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp" )
# else()
#     message(STATUS "OpenMP not found, CPU performance may suffer")
# endif()

include_directories(
    ${VENDOR_DIR}/misc
    ${CNPY_DIR}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

if (SMPLX_BUILD_VIEWER)
    include_directories(
        ${STB_DIR}
        ${IMGUI_DIR}
    )
endif(SMPLX_BUILD_VIEWER)

# Eigen
if ( SMPLX_USE_SYSTEM_EIGEN )
    find_package(Eigen3 3.3.90 QUIET NO_MODULE)
    if (TARGET Eigen3::Eigen)
        message(STATUS "Using system Eigen")
        set (DEPENDENCIES ${DEPENDENCIES} Eigen3::Eigen)
    else()
        message(STATUS "System Eigen > 3.3.90 not found, using vendor Eigen")
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
    include_directories(${ZLIB_DIR})
    set( DEPENDENCIES ${DEPENDENCIES} zlibstatic )
endif()

SET(MESHVIEW_BUILD_EXAMPLE OFF CACHE BOOL "meshview example" FORCE)
SET(Eigen3_DIR '${VENDOR_DIR}/eigen/cmake' CACHE BOOL "eigen dir" FORCE)
add_subdirectory(${MESHVIEW_DIR})

#Link cuda libraries
# if (SMPLX_USE_CUDA)
#     set( DEPENDENCIES ${DEPENDENCIES} cublas cusparse )
# endif()

add_library( ${PROJ_NAME} STATIC ${SOURCES} ${VENDOR_SOURCES} )
set_target_properties( ${PROJ_NAME} PROPERTIES OUTPUT_NAME "smplx" )
target_include_directories(${PROJ_NAME} PUBLIC
    $<BUILD_INTERFACE:${INCLUDE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries( ${PROJ_NAME} ${DEPENDENCIES} )
install(TARGETS ${PROJ_NAME} DESTINATION lib)
if (SMPLX_USE_CUDA)
    set_target_properties( ${PROJ_NAME} PROPERTIES CUDA_SEPERABLE_COMPILATION ON)
endif()

add_executable( example main_example.cpp )
target_link_libraries( example ${PROJ_NAME} )
set_target_properties( example PROPERTIES OUTPUT_NAME "smplx-example" )
install(TARGETS example DESTINATION bin)

if ( SMPLX_BUILD_VIEWER )
    add_executable( viewer main_viewer.cpp )
    target_link_libraries( viewer meshview ${PROJ_NAME} )
    set_target_properties( viewer PROPERTIES OUTPUT_NAME "smplx-viewer" )
    install(TARGETS viewer DESTINATION bin)

    add_executable( amass main_amass.cpp )
    target_link_libraries( amass meshview ${PROJ_NAME} )
    set_target_properties( amass PROPERTIES OUTPUT_NAME "smplx-amass" )
    install(TARGETS amass DESTINATION bin)
endif( SMPLX_BUILD_VIEWER )

if ( MSVC )
    #set_property(TARGET ${PROJ_NAME} APPEND PROPERTY LINK_FLAGS /DEBUG /LTCG )
    set_property(TARGET viewer APPEND PROPERTY LINK_FLAGS "/DEBUG /LTCG" )
    set_property(TARGET amass APPEND PROPERTY LINK_FLAGS "/DEBUG /LTCG" )
    set_property(TARGET example APPEND PROPERTY LINK_FLAGS "/DEBUG /LTCG" )
endif ( MSVC )

if(WIN32)
    # string(REPLACE "/MD" "/MT" CMAKE_C_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE})
    # string(REPLACE "/MDd" "/MTd" CMAKE_C_FLAGS_DEBUG ${CMAKE_C_FLAGS_DEBUG})
    # string(REPLACE "/MD" "/MT" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
    # string(REPLACE "/MDd" "/MTd" CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
    add_definitions(-DNOMINMAX)
elseif(CYGWIN)
elseif(APPLE)
elseif(UNIX)
    target_link_libraries( ${PROJ_NAME} -pthread )
    target_link_libraries( example -pthread )
    if (SMPLX_BUILD_VIEWER)
        target_link_libraries( viewer -pthread )
    endif()
    add_definitions(-DUNIX)
    add_compile_options(-Wno-deprecated-declarations)
    add_compile_options(-Wno-unused-result)
endif(WIN32)

# create version.hpp
configure_file( version.hpp.in "${PROJECT_BINARY_DIR}/include/smplx/version.hpp" )

source_group( "Header Files" FILES ${HEADERS} )
source_group( "Source Files" FILES ${SOURCES} )
source_group( "Vendor Source Files" FILES ${VENDOR_SOURCES} )
