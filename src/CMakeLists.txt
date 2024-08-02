include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(lpcnet_demo lpcnet_demo.c)
target_link_libraries(lpcnet_demo opus m)
