# ================= CUDA + MSVC + GLAD + GLFW + NRD (Windows) =================
# Usage:
#   make              # release build
#   make CONFIG=debug # debug build

SHELL := cmd

# ---- Paths -------------------------------------------------------------
PROJECT_DIR := .
SRC_DIR     := src
BUILD_DIR   := build
OBJ_DIR     := $(BUILD_DIR)/obj
BIN_DIR     := bin

GLAD_ROOT   := libs/glad
GLAD_INC    := $(GLAD_ROOT)/include

# GLAD setup
ifneq ("$(wildcard $(GLAD_ROOT)/src/gl.c)","")
  GLAD_SRC := $(GLAD_ROOT)/src/gl.c
else
  GLAD_SRC := $(GLAD_ROOT)/src/glad.c
endif

GLFW_ROOT   := libs/glfw-3.4.bin.WIN64
GLFW_INC    := $(GLFW_ROOT)/include
GLFW_LIBDIR := $(GLFW_ROOT)/lib-vc2022

# NRD
NRD_ROOT    := libs/NRD
NRD_INC     := $(NRD_ROOT)/Include

# CUDA
CUDA_PATH ?= C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0
NVCC      ?= "$(CUDA_PATH)/bin/nvcc.exe"

# Visual Studio vcvars
VCVARS ?= C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat

# ---- Target ------------------------------------------------------------
TARGET := engine.exe

# ---- Config -----------------------------------------------------------
CONFIG ?= release

DEFS      := /DWIN32_LEAN_AND_MEAN /DNOMINMAX /D_CRT_SECURE_NO_WARNINGS /DGLFW_INCLUDE_NONE
# Added "." to includes so you can include "src/math/vec3.cuh" or relative paths easily
# The includes will now allow you to do #include "raytracer/header.h" or #include "pathtracer/header.h"
INCLUDES  := -I"$(GLFW_INC)" -I"$(GLAD_INC)" -I"$(NRD_INC)" -I"$(SRC_DIR)" -I.

ARCH ?= -gencode arch=compute_86,code=sm_86

ifeq ($(CONFIG),debug)
  CLFLAGS    := /nologo /MDd /Zi /Od /EHsc $(DEFS)
  # Add -dc for separate compilation to handle .cu files calling functions from other .cu files
  NVCCFLAGS  := -std=c++17 -dc -G -Xcompiler="/MDd /Zi /EHsc" $(INCLUDES)
  LDFLAGS    := /DEBUG
else
  CLFLAGS    := /nologo /MD /O2 /EHsc $(DEFS)
  # Add -dc for separate compilation
  NVCCFLAGS  := -std=c++17 -dc -O2 -Xcompiler="/MD /EHsc" $(INCLUDES)
  LDFLAGS    :=
endif

# ---- File Discovery ---------------------------------------------------
# More explicit file discovery that works better on Windows

# Find files in root
CU_SOURCES_ROOT := $(wildcard $(SRC_DIR)/*.cu)
CPP_SOURCES_ROOT := $(wildcard $(SRC_DIR)/*.cpp)

# Find files in original subdirectories
CU_SOURCES_APP := $(wildcard $(SRC_DIR)/app/*.cu)
CU_SOURCES_MATH := $(wildcard $(SRC_DIR)/math/*.cu)
CU_SOURCES_RENDERING := $(wildcard $(SRC_DIR)/rendering/*.cu)
CU_SOURCES_SCENE := $(wildcard $(SRC_DIR)/scene/*.cu)

CPP_SOURCES_APP := $(wildcard $(SRC_DIR)/app/*.cpp)
CPP_SOURCES_MATH := $(wildcard $(SRC_DIR)/math/*.cpp)
CPP_SOURCES_RENDERING := $(wildcard $(SRC_DIR)/rendering/*.cpp)
CPP_SOURCES_SCENE := $(wildcard $(SRC_DIR)/scene/*.cpp)

# --- NEW: Common Discovery ---
CU_SOURCES_COMMON  := $(wildcard $(SRC_DIR)/common/*.cu)
CPP_SOURCES_COMMON := $(wildcard $(SRC_DIR)/common/*.cpp)

# --- NEW: Raytracer Discovery ---
CU_SOURCES_RT  := $(wildcard $(SRC_DIR)/raytracer/*.cu)
CPP_SOURCES_RT := $(wildcard $(SRC_DIR)/raytracer/*.cpp)

# --- NEW: Pathtracer Discovery (Recursive-ish) ---
# 1. Files directly inside src/pathtracer
CU_SOURCES_PT_ROOT  := $(wildcard $(SRC_DIR)/pathtracer/*.cu)
CPP_SOURCES_PT_ROOT := $(wildcard $(SRC_DIR)/pathtracer/*.cpp)

# 2. Files inside the "4 folders" (src/pathtracer/*/*.cu)
# This wildcard grabs any .cu file inside any immediate subdirectory of pathtracer
CU_SOURCES_PT_SUBS  := $(wildcard $(SRC_DIR)/pathtracer/*/*.cu)
CPP_SOURCES_PT_SUBS := $(wildcard $(SRC_DIR)/pathtracer/*/*.cpp)

# Combine all sources
CU_SOURCES := $(CU_SOURCES_ROOT) $(CU_SOURCES_APP) $(CU_SOURCES_MATH) \
              $(CU_SOURCES_RENDERING) $(CU_SOURCES_SCENE) \
              $(CU_SOURCES_COMMON) \
              $(CU_SOURCES_RT) $(CU_SOURCES_PT_ROOT) $(CU_SOURCES_PT_SUBS)

CPP_SOURCES := $(CPP_SOURCES_ROOT) $(CPP_SOURCES_APP) $(CPP_SOURCES_MATH) \
               $(CPP_SOURCES_RENDERING) $(CPP_SOURCES_SCENE) \
               $(CPP_SOURCES_COMMON) \
               $(CPP_SOURCES_RT) $(CPP_SOURCES_PT_ROOT) $(CPP_SOURCES_PT_SUBS)

# Strip duplicates just in case
CU_SOURCES := $(sort $(CU_SOURCES))
CPP_SOURCES := $(sort $(CPP_SOURCES))

# 3. Object Generation
# Map src/%.cu -> build/obj/%.obj
CU_OBJECTS     := $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.obj,$(CU_SOURCES))
# Map src/%.cpp -> build/obj/%.obj
CPP_OBJECTS    := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.obj,$(CPP_SOURCES))

ALL_OBJECTS    := $(CU_OBJECTS) $(CPP_OBJECTS)
GLAD_OBJ       := $(OBJ_DIR)/glad.obj

# For CUDA device link
DLINK_OBJ      := $(OBJ_DIR)/device_link.obj

# ---- Helpers -----------------------------------------------------------
wpath = $(subst /,\,$1)

BUILD_DIR_WIN   := $(call wpath,$(BUILD_DIR))
OBJ_DIR_WIN     := $(call wpath,$(OBJ_DIR))
BIN_DIR_WIN     := $(call wpath,$(BIN_DIR))
GLFW_LIBDIR_WIN := $(call wpath,$(GLFW_LIBDIR))

ifeq ($(CONFIG),debug)
	NRD_LIBDIR    := $(NRD_ROOT)/_Bin/Debug
	NRD_LIB       := NRD.lib
else
	NRD_LIBDIR    := $(NRD_ROOT)/_Bin/Release
	NRD_LIB       := NRD.lib
endif
NRD_LIBDIR_WIN  := $(call wpath,$(NRD_LIBDIR))

# ---- Targets -----------------------------------------------------------
.PHONY: all clean dirs info

all: $(BIN_DIR)/$(TARGET)

# Debug target: Run 'make info' to see what files are being found
info:
	@echo ===== FILE DISCOVERY =====
	@echo [CUDA Sources Found]
	@echo $(CU_SOURCES)
	@echo.
	@echo [CPP Sources Found]
	@echo $(CPP_SOURCES)
	@echo.
	@echo [CUDA Objects to Build]
	@echo $(CU_OBJECTS)
	@echo.
	@echo [CPP Objects to Build]
	@echo $(CPP_OBJECTS)
	@echo.

# Create directory structure in build/obj
# UPDATED: Uses xcopy /t /e to automatically mirror the entire SRC structure 
# into BUILD/OBJ. This handles your new pathtracer subfolders automatically.
dirs:
	@if not exist "$(BUILD_DIR_WIN)" mkdir "$(BUILD_DIR_WIN)"
	@if not exist "$(OBJ_DIR_WIN)"   mkdir "$(OBJ_DIR_WIN)"
	@if not exist "$(BIN_DIR_WIN)"   mkdir "$(BIN_DIR_WIN)"
	@echo Mirroring source directory structure...
	@xcopy /t /e /y "$(SRC_DIR)" "$(OBJ_DIR_WIN)" >nul 2>&1

# ---- Compile GLAD -----------------------------------------------------
$(GLAD_OBJ): $(GLAD_SRC) | dirs
	@echo Compiling GLAD...
	@call "$(VCVARS)" x64 >nul && cl $(CLFLAGS) /I"$(call wpath,$(GLAD_INC))" /I"$(call wpath,$(GLFW_INC))" /c "$(call wpath,$<)" /Fo"$(call wpath,$@)"

# ---- Compile C++ (.cpp) -----------------------------------------------
$(OBJ_DIR)/%.obj: $(SRC_DIR)/%.cpp | dirs
	@echo Compiling C++ $<
	@call "$(VCVARS)" x64 >nul && cl $(CLFLAGS) /I"$(call wpath,$(GLAD_INC))" /I"$(call wpath,$(GLFW_INC))" /I"$(call wpath,$(NRD_INC))" /I"$(call wpath,$(SRC_DIR))" /I. /c "$(call wpath,$<)" /Fo"$(call wpath,$@)"

# ---- Compile CUDA (.cu) -----------------------------------------------
$(OBJ_DIR)/%.obj: $(SRC_DIR)/%.cu | dirs
	@echo Compiling CUDA $<
	@call "$(VCVARS)" x64 >nul && $(NVCC) $(NVCCFLAGS) $(ARCH) -c "$(call wpath,$<)" -o "$(call wpath,$@)"

# ---- CUDA Device Link (if using separate compilation) -----------------
$(DLINK_OBJ): $(CU_OBJECTS) | dirs
	@echo Performing CUDA device link...
	@call "$(VCVARS)" x64 >nul && $(NVCC) $(ARCH) -dlink $(foreach o,$(CU_OBJECTS),"$(call wpath,$(o))") -o "$(call wpath,$@)"

# ---- Link -------------------------------------------------------------
$(BIN_DIR)/$(TARGET): $(ALL_OBJECTS) $(GLAD_OBJ) $(DLINK_OBJ) | dirs
	@echo Linking $@ ...
	@call "$(VCVARS)" x64 >nul && link /nologo /OUT:"$(call wpath,$@)" \
		$(foreach o,$(ALL_OBJECTS) $(GLAD_OBJ) $(DLINK_OBJ),"$(call wpath,$(o))") \
		/LIBPATH:"$(GLFW_LIBDIR_WIN)" /LIBPATH:"$(call wpath,$(CUDA_PATH))/lib/x64" /LIBPATH:"$(NRD_LIBDIR_WIN)" \
		glfw3dll.lib opengl32.lib user32.lib gdi32.lib shell32.lib cudart.lib $(NRD_LIB) $(LDFLAGS)
	@if exist "$(GLFW_LIBDIR_WIN)\glfw3.dll" copy /Y "$(GLFW_LIBDIR_WIN)\glfw3.dll" "$(BIN_DIR_WIN)" >nul
	@if exist "$(NRD_LIBDIR_WIN)\NRD.dll" copy /Y "$(NRD_LIBDIR_WIN)\NRD.dll" "$(BIN_DIR_WIN)" >nul
	@if exist "$(NRD_LIBDIR_WIN)\nrd.dll" copy /Y "$(NRD_LIBDIR_WIN)\nrd.dll" "$(BIN_DIR_WIN)" >nul
	@echo Build complete!

# ---- Clean -------------------------------------------------------------
clean:
	@if exist "$(BIN_DIR_WIN)\NRD.dll" del /Q "$(BIN_DIR_WIN)\NRD.dll"
	@if exist "$(BIN_DIR_WIN)\nrd.dll" del /Q "$(BIN_DIR_WIN)\nrd.dll"
	@if exist "$(BIN_DIR_WIN)\glfw3.dll" del /Q "$(BIN_DIR_WIN)\glfw3.dll"
	@if exist "$(OBJ_DIR_WIN)" rmdir /S /Q "$(OBJ_DIR_WIN)"
	@if exist "$(BIN_DIR_WIN)\$(TARGET)" del /Q "$(BIN_DIR_WIN)\$(TARGET)"
	@echo Clean complete!