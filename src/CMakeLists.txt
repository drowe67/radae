include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(ch ch.c ch_fdmdv.c)
target_link_libraries(ch m)

add_executable(mksine mksine.c)
target_link_libraries(mksine m)

add_executable(lpcnet_demo lpcnet_demo.c)
target_link_libraries(lpcnet_demo opus m)

if(APPLE AND BUILD_OSX_UNIVERSAL)
add_dependencies(lpcnet_demo build_opus_x86)
add_dependencies(lpcnet_demo build_opus_arm)
else(APPLE AND BUILD_OSX_UNIVERSAL)
add_dependencies(lpcnet_demo build_opus)
endif(APPLE AND BUILD_OSX_UNIVERSAL)
