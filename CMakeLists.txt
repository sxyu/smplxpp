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

set( INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include" )
set( SRC_DIR "${PROJECT_SOURCE_DIR}/src" )
set( VENDOR_DIR "${PROJECT_SOURCE_DIR}/3rdparty" )
set( CNPY_DIR "${VENDOR_DIR}/cnpy" )
set( GLFW_DIR "${VENDOR_DIR}/glfw" )
set( GLEW_DIR "${VENDOR_DIR}/glew" )
set( GLM_DIR "${VENDOR_DIR}/glm" )
set( ZLIB_DIR "${VENDOR_DIR}/zlib" )
set( EIGEN_DIR "${VENDOR_DIR}/eigen" )

set( CMAKE_CXX_STACK_SIZE "10000000" )
set( CMAKE_CXX_STANDARD 14 )
set( CMAKE_CXX_STANDARD_REQUIRED ON)
set( CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake_modules" )

set( PROJ_NAME "smpl_cpp" )

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

SET(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "GLFW example" FORCE)
SET(GLFW_BUILD_TESTS OFF CACHE BOOL "GLFW tests" FORCE)
SET(GLFW_BUILD_DOCS OFF CACHE BOOL "GLFW docs" FORCE)
SET(GLFW_INSTALL OFF CACHE BOOL "GLFW install" FORCE)
add_subdirectory(${GLFW_DIR})

include_directories(
    ${INCLUDE_DIR}
    ${CNPY_DIR}
    ${EIGEN_DIR}
    ${GLM_DIR}
    ${GLEW_DIR}/include
    ${GLFW_DIR}/include
    ${CUDA_INCLUDE_DIRS}
)

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
    OpenGL::GL
    glfw
    ${GLFW_LIBRARIES}
    glew
)

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

find_package(ZLIB QUIET)
if (ZLIB_FOUND)
    message( STATUS "Using system zlib." )
    set( DEPENDENCIES ${DEPENDENCIES} ZLIB::ZLIB )
else()
    message( STATUS "Using included zlib." )
    add_subdirectory(${ZLIB_DIR})
    include_directories(${ZLIB_DIR}/include)
    set( DEPENDENCIES ${DEPENDENCIES} zlibstatic )
endif()

set(
    VENDOR_SOURCES
    ${GLEW_DIR}/src/glew.c
    ${CNPY_DIR}/cnpy.cpp
)

file(GLOB SOURCES ${SRC_DIR}/*.cpp)
file(GLOB HEADERS ${INCLUDE_DIR}/*.cpp)

add_library( ${PROJ_NAME} ${SOURCES} ${VENDOR_SOURCES} )
set_target_properties( ${PROJ_NAME} PROPERTIES OUTPUT_NAME "smplxpp" )
target_link_libraries( ${PROJ_NAME} ${DEPENDENCIES} )

add_executable( demo demo.cpp )
target_link_libraries( demo ${DEPENDENCIES} ${PROJ_NAME} )

if ( MSVC )
    set_property(TARGET ${PROJ_NAME} APPEND PROPERTY LINK_FLAGS /DEBUG)
    set_property(TARGET demo APPEND PROPERTY LINK_FLAGS /DEBUG)
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
    target_link_libraries( demo -pthread )
    add_definitions(-DUNIX)
    add_compile_options(-Wno-deprecated-declarations)
    add_compile_options(-Wno-unused-result)
endif(WIN32)

add_definitions( -DGLEW_STATIC)

source_group( "Header Files" FILES ${HEADERS} )
source_group( "Source Files" FILES ${SOURCES} )
source_group( "Vendor Source Files" FILES ${VENDOR_SOURCES} )
