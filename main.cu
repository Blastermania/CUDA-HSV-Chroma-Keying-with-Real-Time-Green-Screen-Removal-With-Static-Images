#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <cuda_runtime.h>
#include <iostream>

using namespace cv;
using namespace std;

__device__ void rgb_to_hsv(unsigned char r, unsigned char g, unsigned char b, float &h, float &s, float &v) {
    float fr = r / 255.0f;
    float fg = g / 255.0f;
    float fb = b / 255.0f;

    float cmax = fmaxf(fr, fmaxf(fg, fb));
    float cmin = fminf(fr, fminf(fg, fb));
    float delta = cmax - cmin;

    h = 0.0f;

    if (delta != 0.0f) {
        if (cmax == fr) {
            h = 60.0f * fmodf(((fg - fb) / delta), 6.0f);
        } else if (cmax == fg) {
            h = 60.0f * (((fb - fr) / delta) + 2.0f);
        } else {
            h = 60.0f * (((fr - fg) / delta) + 4.0f);
        }
    }

    if (h < 0.0f) h += 360.0f;

    s = (cmax == 0.0f) ? 0.0f : delta / cmax;
    v = cmax;
}

__global__ void hsv_chroma_kernel(uchar3* input, uchar3* bg, uchar3* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;

    uchar3 pixel = input[idx];
    float h, s, v;
    rgb_to_hsv(pixel.x, pixel.y, pixel.z, h, s, v);

    bool is_green = (h >= 60.0f && h <= 180.0f && s > 0.4f && v > 0.3f);
    output[idx] = is_green ? bg[idx] : pixel;
}

int main() {
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Failed to open webcam!" << endl;
        return -1;
    }

    Mat frame, background, resized_bg;
    cap >> frame;
    int width = frame.cols;
    int height = frame.rows;

    // Load replacement background image
    background = imread("background.jpg");
    if (background.empty()) {
        cerr << "Failed to load background.jpg!" << endl;
        return -1;
    }
    resize(background, resized_bg, Size(width, height));

    // Allocate host/device memory
    uchar3 *d_input, *d_bg, *d_output;
    size_t total = width * height * sizeof(uchar3);

    cudaMalloc(&d_input, total);
    cudaMalloc(&d_bg, total);
    cudaMalloc(&d_output, total);

    Mat output_frame(height, width, CV_8UC3);

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        resize(background, resized_bg, frame.size());

        cudaMemcpy(d_input, frame.ptr<uchar3>(), total, cudaMemcpyHostToDevice);
        cudaMemcpy(d_bg, resized_bg.ptr<uchar3>(), total, cudaMemcpyHostToDevice);

        dim3 block(16, 16);
        dim3 grid((width + 15) / 16, (height + 15) / 16);
        hsv_chroma_kernel<<<grid, block>>>(d_input, d_bg, d_output, width, height);

        cudaMemcpy(output_frame.ptr<uchar3>(), d_output, total, cudaMemcpyDeviceToHost);

        imshow("HSV Chroma Key", output_frame);
        if (waitKey(1) == 27) break;  // ESC key to exit
    }

    cudaFree(d_input);
    cudaFree(d_bg);
    cudaFree(d_output);
    cap.release();
    destroyAllWindows();
    return 0;
}
