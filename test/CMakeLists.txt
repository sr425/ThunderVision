file(GLOB_RECURSE files
    "src/*.h"
    "src/*.cpp"
)

add_executable (XTest ${files})


target_link_libraries (XTest LINK_PUBLIC ThunderVision)
