add_subdirectory(vanilla_locvol)
add_library(nets INTERFACE)
target_link_libraries(nets INTERFACE vanilla_locvol modules)
