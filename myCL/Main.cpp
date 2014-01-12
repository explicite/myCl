#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#include <iostream>
#include <fstream>
#include <string>

#include "test.hpp"


int main(int argc, char* argv[])
{
	const unsigned int N_ELEMENTS = 2048;
	float* M = new float[N_ELEMENTS * N_ELEMENTS];
	float* V = new float[N_ELEMENTS];
	float* W = new float[N_ELEMENTS];

	register int i;
	for(i = 0; i < N_ELEMENTS * N_ELEMENTS; i++)
		M[i] = i;

	for(i = 0; i < N_ELEMENTS ; i++)		
		V[i] = i;

	
	try {
		//Platform layer
		std::vector < cl::Platform > platforms;
		cl::Platform::get(&platforms);
		if(platforms.size() == 0) {
			std::cout << "No platforms found!\n";
			exit(1);
		}
		cl::Platform default_platform = platforms[0];
		std::cout << "Using platform: " << default_platform.getInfo < CL_PLATFORM_NAME > () << std::endl;

		std::vector < cl::Device > devices;
		default_platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
		if(devices.size() == 0){
			std::cout << "No devices found\n";
			exit(1);
		}

		cl::Device device = devices[0];
		std::cout << "Using device: " << device.getInfo < CL_DEVICE_NAME > ()  << std::endl;
		std::cout << "Working intems: " << device.getInfo < CL_DEVICE_MAX_WORK_ITEM_SIZES > ()[0]  << std::endl;
		std::cout << "Profile: " << device.getInfo < CL_DEVICE_PROFILE > ()  << std::endl;
		std::cout << "Version: " << device.getInfo < CL_DEVICE_OPENCL_C_VERSION > ()  << std::endl;

		cl::Context context(devices);

		cl::CommandQueue queue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);

		//Runtime layer
		cl::Buffer bufferM= cl::Buffer(context, CL_MEM_READ_ONLY, N_ELEMENTS * N_ELEMENTS * sizeof(float));
		cl::Buffer bufferV= cl::Buffer(context, CL_MEM_READ_ONLY, N_ELEMENTS * sizeof(float));
		cl::Buffer bufferW= cl::Buffer(context, CL_MEM_WRITE_ONLY, N_ELEMENTS * sizeof(float));

		queue.enqueueWriteBuffer(bufferM, CL_TRUE, 0, N_ELEMENTS * sizeof(float), M);
		queue.enqueueWriteBuffer(bufferV, CL_TRUE, 0, N_ELEMENTS * sizeof(float), V);

		//Compiler
		std::ifstream src("mv.cl");
		std::string code(std::istreambuf_iterator < char > (src), (std::istreambuf_iterator < char > ()));
		cl::Program::Sources source(1, std::make_pair(code.c_str(), code.length() + 1));
		cl::Program program = cl::Program(context, source);
		program.build(devices);
		cl::Kernel mv(program, "mv");

		mv.setArg(0, bufferM);
		mv.setArg(1, N_ELEMENTS);
		mv.setArg(2, N_ELEMENTS);
		mv.setArg(3, bufferV);
		mv.setArg(4, bufferW);

		cl::Event event;
		cl_ulong queued;
		cl_ulong submit;
		cl_ulong start;
		cl_ulong end;

		queue.enqueueTask(mv, NULL, &event);
		queue.enqueueReadBuffer(bufferW, CL_TRUE, 0, N_ELEMENTS * sizeof(float), W);
		queue.flush();
		event.wait();

		event.getProfilingInfo < cl_ulong > (CL_PROFILING_COMMAND_QUEUED, &queued);
		event.getProfilingInfo < cl_ulong > (CL_PROFILING_COMMAND_SUBMIT, &submit);
		event.getProfilingInfo < cl_ulong > (CL_PROFILING_COMMAND_START, &start);
		event.getProfilingInfo < cl_ulong > (CL_PROFILING_COMMAND_END, &end);

		bool result = assert(W, omv(M, V, N_ELEMENTS), N_ELEMENTS);

		if(result)
			std::cout << "Success!" << std::endl;
		else
			std::cout << "Failed!" << std::endl;

		//Profiling info
		std::cout << "Queued: " << submit - queued << "ns\n";
		std::cout << "Submit: " << start - submit << "ns\n";
		std::cout << "Computation: " << end - start << "ns\n";

	}catch(cl::Error error)
	{
		std::cout << error.what() << "(" << error.err() << ")" << std::endl;
	}

	system("PAUSE");
	exit(0);
}

