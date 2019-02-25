file(GLOB_RECURSE thunderVisionSources
    "*.h"
    "*.cpp"
)

include_directories(include/ThunderVision)
add_library(ThunderVision STATIC ${thunderVisionSources})

#find_package(OpenMP REQUIRED)
#if(NOT TARGET OpenMP::OpenMP_CXX)
#    add_library(OpenMP_TARGET INTERFACE)
#    add_library(OpenMP::OpenMP_CXX ALIAS OpenMP_TARGET)
#    target_compile_options(OpenMP_TARGET INTERFACE ${OpenMP_CXX_FLAGS})
#    find_package(Threads REQUIRED)
#    target_link_libraries(OpenMP_TARGET INTERFACE Threads::Threads)
#    target_link_libraries(OpenMP_TARGET INTERFACE ${OpenMP_CXX_FLAGS})
#endif()
#target_link_libraries(ThunderVision PUBLIC OpenMP::OpenMP_CXX)

target_include_directories (ThunderVision PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include/ThunderVision)
