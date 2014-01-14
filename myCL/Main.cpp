#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#include <iostream>
#include <fstream>
#include <string>

#include "test.hpp"

int main(int argc, char* argv[])
{
	try {
		//Platform layer
		std::vector <cl::Platform> platforms;
		cl::Platform::get(&platforms);
		if(platforms.size() == 0) {
			std::cout << "No platforms found!\n";
			exit(1);
		}
		cl::Platform default_platform = platforms[0];
		std::cout << "Using platform: " << default_platform.getInfo < CL_PLATFORM_NAME > () << std::endl;

		std::vector <cl::Device> devices;
		default_platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
		if(devices.size() == 0){
			std::cout << "No devices found\n";
			exit(1);
		}
		cl::Device device = devices[0];

		std::cout << "Using device: " << device.getInfo < CL_DEVICE_NAME > ()  << std::endl;
		std::cout << "Device mem: " << device.getInfo < CL_DEVICE_MAX_MEM_ALLOC_SIZE > () << std::endl;
		std::cout << "Device freq: " << device.getInfo <CL_DEVICE_MAX_CLOCK_FREQUENCY> () << "MHz\n";
		std::cout << "Compute units: " << device.getInfo < CL_DEVICE_MAX_COMPUTE_UNITS > () << std::endl;
		std::cout << "Work item dimensions: " << device.getInfo < CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS > () << std::endl;
		size_t MAX_WORK_GROUP_SIZE = device.getInfo < CL_DEVICE_MAX_WORK_GROUP_SIZE > ();
		std::cout << "Work group size: " << MAX_WORK_GROUP_SIZE << std::endl;
		std::cout << "Working intems: " << device.getInfo < CL_DEVICE_MAX_WORK_ITEM_SIZES > ()[0]  << std::endl;
		std::cout << "Kernel group size: " << CL_KERNEL_WORK_GROUP_SIZE << std::endl;
		std::cout << "Profile: " << device.getInfo < CL_DEVICE_PROFILE > ()  << std::endl;
		std::cout << "Version: " << device.getInfo < CL_DEVICE_OPENCL_C_VERSION > ()  << std::endl;

		cl::Context context(devices);
		cl::CommandQueue queue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);

		//Data prepare
		const unsigned int SIZE = 1024*6;
		const unsigned int N_ELEMENTS = SIZE * SIZE;
		size_t V_size = sizeof(float) * N_ELEMENTS;

		float* V1 = (float*)malloc(V_size);
		float* V2 = (float*)malloc(V_size);
		float* P = (float*)malloc(V_size);
		float* vP = (float*)malloc(V_size);

		register int i;
		#pragma omp parallel for schedule(static) private(i)
		for(i = 0; i < N_ELEMENTS; i++){
			V1[i] = i;
			V2[i] = i;
			vP[i] = i + i;
			P[i] = 0;
		}	

		//Runtime layer
		cl::Buffer bufferV1= cl::Buffer(context, CL_MEM_READ_ONLY, V_size);
		cl::Buffer bufferV2= cl::Buffer(context, CL_MEM_READ_ONLY, V_size);
		cl::Buffer bufferP= cl::Buffer(context, CL_MEM_WRITE_ONLY, V_size);

		queue.enqueueWriteBuffer(bufferV1, CL_TRUE, 0, V_size, V1);
		queue.enqueueWriteBuffer(bufferV2, CL_TRUE, 0, V_size, V2);

		//Compiler
		std::ifstream src("vecaddinv.cl");
		std::string code(std::istreambuf_iterator<char>(src), (std::istreambuf_iterator<char>()));
		cl::Program::Sources source(1, std::make_pair(code.c_str(), code.length() + 1));
		cl::Program program = cl::Program(context, source);
		program.build(devices);
		cl::Kernel kernel(program, "vecaddinv");

		kernel.setArg(0, bufferV1);
		kernel.setArg(1, bufferV2);
		kernel.setArg(2, bufferP);

		cl::Event event;
		cl_ulong queued;
		cl_ulong submit;
		cl_ulong start;
		cl_ulong end;

		cl::NDRange global(SIZE, SIZE);
		cl::NDRange local(16, 16);
		queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local, NULL, &event);
		queue.enqueueReadBuffer(bufferP, CL_TRUE, 0, V_size, P);
		queue.flush();
		event.wait();

		event.getProfilingInfo <cl_ulong> (CL_PROFILING_COMMAND_QUEUED, &queued);
		event.getProfilingInfo <cl_ulong> (CL_PROFILING_COMMAND_SUBMIT, &submit);
		event.getProfilingInfo <cl_ulong> (CL_PROFILING_COMMAND_START, &start);
		event.getProfilingInfo <cl_ulong> (CL_PROFILING_COMMAND_END, &end);

		bool result = assert_inv(P, vP, SIZE); 

		if(result)
			std::cout << "Success!" << std::endl;
		else
			std::cout << "Failed!" << std::endl;

		//Profiling info
		std::cout << "Memory: " << (3 * V_size)  << std::endl;
		std::cout << "Queued: " << submit - queued << "ns\n";
		std::cout << "Submit: " << start - submit << "ns\n";
		std::cout << "Computation: " << end - start << "ns\n";

	} catch(cl::Error error) {
		std::cout << error.what() << "(" << error.err() << ")" << std::endl;
	}

	system("PAUSE");
	exit(0);
}
