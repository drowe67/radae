include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(lpcnet_demo lpcnet_demo.c)
target_link_libraries(lpcnet_demo opus m)

add_library(rade rade_api.c)
target_link_libraries(rade Python3::Python)
target_include_directories(rade PRIVATE
    "$<TARGET_PROPERTY:Python3::NumPy,INTERFACE_INCLUDE_DIRECTORIES>")

set_target_properties(rade PROPERTIES
    SOVERSION "0.1"
    PUBLIC_HEADER "rade_api.h"
)

add_executable(radae_tx radae_tx.c)
target_link_libraries(radae_tx rade Python3::Python)
target_include_directories(radae_tx PRIVATE
    "$<TARGET_PROPERTY:Python3::NumPy,INTERFACE_INCLUDE_DIRECTORIES>")

add_executable(radae_rx radae_rx.c)
target_link_libraries(radae_rx rade Python3::Python)
target_include_directories(radae_rx PRIVATE
    "$<TARGET_PROPERTY:Python3::NumPy,INTERFACE_INCLUDE_DIRECTORIES>")

