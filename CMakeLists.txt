cmake_minimum_required( VERSION 2.8 )

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

project( smpl_cpp )
option( USE_FFAST_MATH "Enable ffast-math compiler flag, may cause numerical problems" ON )

set( INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include" )
set( INCLUDE_DIR_THIRD_PARTY "${PROJECT_SOURCE_DIR}/3rdparty/include" )

set( CMAKE_CXX_STACK_SIZE "10000000" )
set( CMAKE_CXX_STANDARD 11 )
set( CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake_modules" )
if (WIN32)
    set( COLOR_TERM OFF )
else()
    set( COLOR_TERM ON )
endif()

if( NOT CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Release )
endif()

set( PROJ_NAME "smpl_cpp" )

include( CheckCXXCompilerFlag )
CHECK_CXX_COMPILER_FLAG( "-std=c++11" COMPILER_SUPPORTS_CXX11 )
CHECK_CXX_COMPILER_FLAG( "-std=c++0x" COMPILER_SUPPORTS_CXX0X )

if( COMPILER_SUPPORTS_CXX11 )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11" )
elseif( COMPILER_SUPPORTS_CXX0X )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x" )
else()
    message( STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler." )
endif()

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

find_package(Eigen3 REQUIRED)

# require zlib
find_package(ZLIB)

# # require Boost
# set( Boost_USE_STATIC_LIBS ON )
# set( Boost_USE_STATIC ON )
# find_package( Boost REQUIRED COMPONENTS filesystem program_options thread system )
#
# # require Ceres
# find_package( Ceres REQUIRED )
# IF(Ceres_FOUND)
#     MESSAGE(STATUS "Found Ceres: ${CERES_INCLUDE_DIRS} ${CERES_LIBRARIES}")
# ENDIF(Ceres_FOUND)

include_directories(
    ${INCLUDE_DIR}
    ${INCLUDE_DIR_THIRD_PARTY}
    # ${CERES_INCLUDE_DIRS}
)

set(
  DEPENDENCIES
  Eigen3::Eigen
  ZLIB::ZLIB
  # ${CERES_LIBRARIES}
)

set(
  SOURCES
  3rdparty/cnpy.cpp
)

add_executable( ${PROJ_NAME} ${SOURCES} demo.cpp )
#target_include_directories( ${PROJ_NAME} PRIVATE ${INCLUDE_DIR} )
set_target_properties( ${PROJ_NAME} PROPERTIES OUTPUT_NAME ${PROJ_NAME} )
target_link_libraries( ${PROJ_NAME} ${DEPENDENCIES} )
if ( PCL_FOUND )
    set_target_properties( ${PROJ_NAME} PROPERTIES COMPILE_FLAGS ${TARGET_COMPILE_FLAGS} )
endif()

if ( MSVC )
    set_property(TARGET ${PROJ_NAME} APPEND PROPERTY LINK_FLAGS /DEBUG)
else ()
    target_link_libraries( ${PROJ_NAME} -pthread )
endif ( MSVC )
