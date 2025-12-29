
// glfw_view_interop.hpp - GLFW + GLAD + CUDA-OpenGL PBO interop (no CPU copy)
// NOTE: Include this from a .cu or a TU compiled with nvcc (uses <cuda_gl_interop.h>).
#pragma once

#pragma once
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#endif

#define GLFW_INCLUDE_NONE
#include <glad/gl.h>
#include <GLFW/glfw3.h>

// extra guard in case some other header snuck them in
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

#include <stdexcept>
#include <string>
#include <vector>
#include <cstdint>

#define GLFW_INCLUDE_NONE
#include <glad/gl.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

namespace rtgl
{

    struct InteropViewer
    {
        GLFWwindow *window = nullptr;
        GLuint tex = 0, vao = 0, vbo = 0, pbo = 0, program = 0;
        cudaGraphicsResource *cuda_pbo = nullptr;

        int viewW = 0, viewH = 0;
        // store windowed state for restoring from fullscreen
        int winX = 100, winY = 100, winW = 1280, winH = 720;
        bool isFullscreen = false;
    };

    static void resize_backing(InteropViewer &V, int w, int h)
    {
        if (w == 0 || h == 0)
            return;
        V.viewW = w;
        V.viewH = h;

        glBindTexture(GL_TEXTURE_2D, V.tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);

        if (V.cuda_pbo)
        {
            cudaGraphicsUnregisterResource(V.cuda_pbo);
            V.cuda_pbo = nullptr;
        }
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, V.pbo);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, (GLsizeiptr)((size_t)w * h * 3), nullptr, GL_STREAM_DRAW);
        cudaGraphicsGLRegisterBuffer(&V.cuda_pbo, V.pbo, cudaGraphicsMapFlagsWriteDiscard);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        glViewport(0, 0, w, h);
    }

    static void enter_fullscreen(InteropViewer &V)
    {
        // remember windowed placement
        glfwGetWindowPos(V.window, &V.winX, &V.winY);
        glfwGetWindowSize(V.window, &V.winW, &V.winH);

        GLFWmonitor *monitor = glfwGetPrimaryMonitor();
        const GLFWvidmode *mode = glfwGetVideoMode(monitor);
        glfwSetWindowMonitor(V.window, monitor, 0, 0, mode->width, mode->height, mode->refreshRate);

        resize_backing(V, mode->width, mode->height);
        V.isFullscreen = true;
    }

    static void exit_fullscreen(InteropViewer &V)
    {
        glfwSetWindowMonitor(V.window, nullptr, V.winX, V.winY, V.winW, V.winH, 0);
        resize_backing(V, V.winW, V.winH);
        V.isFullscreen = false;
    }

    static void toggle_fullscreen(InteropViewer &V)
    {
        if (V.isFullscreen)
            exit_fullscreen(V);
        else
            enter_fullscreen(V);
    }

    static inline void _check(bool ok, const char *msg)
    {
        if (!ok)
            throw std::runtime_error(msg);
    }

    static inline unsigned int compileShader(unsigned int type, const char *src)
    {
        unsigned int s = glCreateShader(type);
        glShaderSource(s, 1, &src, nullptr);
        glCompileShader(s);
        int ok = 0;
        glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
        if (!ok)
        {
            char log[4096];
            glGetShaderInfoLog(s, 4096, nullptr, log);
            std::string m = std::string("Shader compile failed: ") + log;
            glDeleteShader(s);
            throw std::runtime_error(m);
        }
        return s;
    }

    static inline unsigned int linkProgram(const char *vs, const char *fs)
    {
        unsigned int v = compileShader(GL_VERTEX_SHADER, vs);
        unsigned int f = compileShader(GL_FRAGMENT_SHADER, fs);
        unsigned int p = glCreateProgram();
        glAttachShader(p, v);
        glAttachShader(p, f);
        glLinkProgram(p);
        glDeleteShader(v);
        glDeleteShader(f);
        int ok = 0;
        glGetProgramiv(p, GL_LINK_STATUS, &ok);
        if (!ok)
        {
            char log[4096];
            glGetProgramInfoLog(p, 4096, nullptr, log);
            glDeleteProgram(p);
            throw std::runtime_error(std::string("Program link failed: ") + log);
        }
        return p;
    }

    static const char *kVS = R"GLSL(
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aUV;
out vec2 vUV;
void main() {
    vUV = vec2(aUV.x, 1.0 - aUV.y); // flip Y
    gl_Position = vec4(aPos, 0.0, 1.0);
}
)GLSL";

    static const char *kFS = R"GLSL(
#version 330 core
in vec2 vUV;
out vec4 FragColor;
uniform sampler2D uTex;
void main() {
    FragColor = texture(uTex, vUV);
}
)GLSL";

    inline void init_interop_viewer(InteropViewer &V, int width, int height, const char *title, int cudaDevice = 0)
    {
        V.viewW = width;
        V.viewH = height;

        // Pick CUDA device (must be GL-compatible). Do this before we register GL resources.
        cudaSetDevice(cudaDevice);

        if (!glfwInit())
            throw std::runtime_error("glfwInit failed");
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
        V.window = glfwCreateWindow(width, height, title, nullptr, nullptr);
        _check(V.window != nullptr, "glfwCreateWindow failed");
        glfwMakeContextCurrent(V.window);
        glfwSwapInterval(0); // vsync off

        if (!gladLoadGL(glfwGetProcAddress))
        {
            throw std::runtime_error("Failed to load GL with GLAD (gladLoadGL)");
        }

        glfwSetWindowUserPointer(V.window, &V);
        glfwSetKeyCallback(V.window, [](GLFWwindow *win, int key, int, int action, int)
                           {
            if (action == GLFW_PRESS && key == GLFW_KEY_F11) 
            {
                auto* vp = static_cast<InteropViewer*>(glfwGetWindowUserPointer(win));
                toggle_fullscreen(*vp);
            } });

        glfwSetFramebufferSizeCallback(V.window, [](GLFWwindow *win, int w, int h)
                                       {
    auto* vp = static_cast<InteropViewer*>(glfwGetWindowUserPointer(win));
    resize_backing(*vp, w, h); });

        // Texture
        glGenTextures(1, &V.tex);
        glBindTexture(GL_TEXTURE_2D, V.tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);

        // Fullscreen quad
        float verts[] = {
            // pos   // uv
            -1.f,
            -1.f,
            0.f,
            1.f,
            1.f,
            -1.f,
            1.f,
            1.f,
            1.f,
            1.f,
            1.f,
            0.f,
            -1.f,
            -1.f,
            0.f,
            1.f,
            1.f,
            1.f,
            1.f,
            0.f,
            -1.f,
            1.f,
            0.f,
            0.f,
        };
        glGenVertexArrays(1, &V.vao);
        glGenBuffers(1, &V.vbo);
        glBindVertexArray(V.vao);
        glBindBuffer(GL_ARRAY_BUFFER, V.vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)(2 * sizeof(float)));

        V.program = linkProgram(kVS, kFS);
        glUseProgram(V.program);
        glUniform1i(glGetUniformLocation(V.program, "uTex"), 0);

        // Create PBO for pixel unpack and register with CUDA
        glGenBuffers(1, &V.pbo);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, V.pbo);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, (GLsizeiptr)((size_t)width * height * 3), nullptr, GL_STREAM_DRAW);

        cudaError_t cerr = cudaGraphicsGLRegisterBuffer(&V.cuda_pbo, V.pbo, cudaGraphicsMapFlagsWriteDiscard);
        if (cerr != cudaSuccess)
        {
            throw std::runtime_error(std::string("cudaGraphicsGLRegisterBuffer failed: ") + cudaGetErrorString(cerr));
        }

        // Unbind PBO
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    }

    inline uint8_t *map_pbo_device_ptr(InteropViewer &V, size_t *nbytes = nullptr)
    {
        cudaError_t cerr = cudaGraphicsMapResources(1, &V.cuda_pbo);
        if (cerr != cudaSuccess)
        {
            throw std::runtime_error(std::string("cudaGraphicsMapResources failed: ") + cudaGetErrorString(cerr));
        }
        void *dev_ptr = nullptr;
        size_t sz = 0;
        cerr = cudaGraphicsResourceGetMappedPointer(&dev_ptr, &sz, V.cuda_pbo);
        if (cerr != cudaSuccess)
        {
            throw std::runtime_error(std::string("cudaGraphicsResourceGetMappedPointer failed: ") + cudaGetErrorString(cerr));
        }
        if (nbytes)
            *nbytes = sz;
        return reinterpret_cast<uint8_t *>(dev_ptr);
    }

    inline void unmap_pbo(InteropViewer &V)
    {
        cudaError_t cerr = cudaGraphicsUnmapResources(1, &V.cuda_pbo);
        if (cerr != cudaSuccess)
        {
            throw std::runtime_error(std::string("cudaGraphicsUnmapResources failed: ") + cudaGetErrorString(cerr));
        }
    }

    inline void blit_pbo_to_texture(const InteropViewer &V)
    {
        glBindTexture(GL_TEXTURE_2D, V.tex);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, V.pbo);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        // Data ptr is taken from bound PBO; pass nullptr offset.
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, V.viewW, V.viewH, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    }

    inline void draw_interop(const InteropViewer &V)
    {
        int winW, winH;
        glfwGetFramebufferSize(V.window, &winW, &winH);
        glViewport(0, 0, winW, winH);
        glClear(GL_COLOR_BUFFER_BIT);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, V.tex);
        glUseProgram(V.program);
        glBindVertexArray(V.vao);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glfwSwapBuffers(V.window);
        glfwPollEvents();
    }

    inline void destroy_interop_viewer(InteropViewer &V)
    {
        if (V.cuda_pbo)
        {
            cudaGraphicsUnregisterResource(V.cuda_pbo);
            V.cuda_pbo = nullptr;
        }
        if (V.pbo)
        {
            glDeleteBuffers(1, &V.pbo);
            V.pbo = 0;
        }
        if (V.program)
        {
            glDeleteProgram(V.program);
            V.program = 0;
        }
        if (V.vbo)
        {
            glDeleteBuffers(1, &V.vbo);
            V.vbo = 0;
        }
        if (V.vao)
        {
            glDeleteVertexArrays(1, &V.vao);
            V.vao = 0;
        }
        if (V.tex)
        {
            glDeleteTextures(1, &V.tex);
            V.tex = 0;
        }
        if (V.window)
        {
            glfwDestroyWindow(V.window);
            V.window = nullptr;
        }
        glfwTerminate();
    }

} // namespace rtgl
