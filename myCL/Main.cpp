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
		
		//Size for vector and matrix
		const unsigned int SIZE = 1024*8;
		size_t V_size = sizeof(float) * SIZE;
		size_t M_size = sizeof(float) * SIZE * SIZE;

		//Prepare kernels
		//Matrix - Vector Multiplication
		std::ifstream matvecmul_src("matvecmul.cl");
		std::string matvecmul_code(std::istreambuf_iterator<char>(matvecmul_src), (std::istreambuf_iterator<char>()));
		cl::Program::Sources matvecmul_source(1, std::make_pair(matvecmul_code.c_str(), matvecmul_code.length() + 1));
		cl::Program matvecmul_program = cl::Program(context, matvecmul_source);
		matvecmul_program.build(devices);
		cl::Kernel matvecmul(matvecmul_program, "matvecmul");

		//Add Vectors
		std::ifstream vecadd_src("vecadd.cl");
		std::string vecadd_code(std::istreambuf_iterator<char>(vecadd_src), (std::istreambuf_iterator<char>()));
		cl::Program::Sources vecadd_source(1, std::make_pair(vecadd_code.c_str(), vecadd_code.length() + 1));
		cl::Program vecadd_program = cl::Program(context, vecadd_source);
		vecadd_program.build(devices);
		cl::Kernel vecadd(vecadd_program, "vecadd");

		//Invers Matrix
		std::ifstream matinv_src("matinv.cl");
		std::string matinv_code(std::istreambuf_iterator<char>(matinv_src), (std::istreambuf_iterator<char>()));
		cl::Program::Sources matinv_source(1, std::make_pair(matinv_code.c_str(), matinv_code.length() + 1));
		cl::Program matinv_program = cl::Program(context, matinv_source);
		matinv_program.build(devices);
		cl::Kernel matinv(matinv_program, "matinv");

		//Profiler
		cl::Event profiler;

		//Preapare data
		float* V = (float*)malloc(V_size);
		float* V1 = (float*)malloc(V_size);
		float* M = (float*)malloc(M_size);
		float* invM = (float*)malloc(M_size);
		float* V2 = (float*)malloc(V_size);
		float* P = (float*)malloc(V_size);

		register int i;
		#pragma omp parallel for schedule(static) private(i)
		for(i = 0; i < SIZE; i++){
			V[i] = i;
			V1[i] = 0;
			V2[i] = 0;
			P[i] = 0;
		}	

		#pragma omp parallel for schedule(static) private(i)
		for(i = 0; i < SIZE * SIZE; i++){
			M[i] = i;
			invM[i] = 0;
		}	

		//Matrix vector multiplication
		cl::Buffer readM = cl::Buffer(context, CL_MEM_READ_ONLY, M_size);
		cl::Buffer readV = cl::Buffer(context, CL_MEM_READ_ONLY, V_size);
		cl::Buffer writeV = cl::Buffer(context, CL_MEM_WRITE_ONLY, V_size);
		
		queue.enqueueWriteBuffer(readM, CL_TRUE, 0, M_size, M);
		queue.enqueueWriteBuffer(readV, CL_TRUE, 0, V_size, V);
		

		matvecmul.setArg(0, readM);
		matvecmul.setArg(1, readV);
		matvecmul.setArg(2, writeV);
	
		queue.enqueueNDRangeKernel(matvecmul, cl::NullRange, cl::NDRange(SIZE, SIZE), cl::NDRange(16, 16), NULL, &profiler);
		queue.enqueueReadBuffer(writeV, CL_TRUE, 0, V_size, V1);
		queue.flush();
		profiler.wait();

		//Profiler
		cl_ulong matvecmul_queued;
		cl_ulong matvecmul_submit;
		cl_ulong matvecmul_start;
		cl_ulong matvecmul_end;

		profiler.getProfilingInfo <cl_ulong> (CL_PROFILING_COMMAND_QUEUED, &matvecmul_queued);
		profiler.getProfilingInfo <cl_ulong> (CL_PROFILING_COMMAND_SUBMIT, &matvecmul_submit);
		profiler.getProfilingInfo <cl_ulong> (CL_PROFILING_COMMAND_START, &matvecmul_start);
		profiler.getProfilingInfo <cl_ulong> (CL_PROFILING_COMMAND_END, &matvecmul_end);

		//Veryfication
		bool result = assert(omv(M, V, SIZE), V1, SIZE); 

		if(result)
			std::cout << "Success!" << std::endl;
		else
			std::cout << "Failed!" << std::endl;
		
		//Profiling info
		std::cout << "Size: " << SIZE << std::endl;
		std::cout << "Memory: " << (2 * V_size) + M_size  << std::endl;
		std::cout << "Queued: " << matvecmul_submit - matvecmul_queued << "ns\n";
		std::cout << "Submit: " << matvecmul_start - matvecmul_submit << "ns\n";
		std::cout << "Computation: " << matvecmul_end - matvecmul_start << "ns\n";

		//Matrix invers
		cl::Buffer writeM = cl::Buffer(context, CL_MEM_WRITE_ONLY, M_size);

		queue.enqueueWriteBuffer(readM, CL_TRUE, 0, M_size, M);

		matinv.setArg(0, readM);
		matinv.setArg(1, writeM);

		queue.enqueueNDRangeKernel(matinv, cl::NullRange, cl::NDRange(SIZE, SIZE), cl::NDRange(16, 16), NULL, &profiler);
		queue.enqueueReadBuffer(writeM, CL_TRUE, 0, M_size, invM);
		queue.flush();
		profiler.wait();

		cl_ulong matinv_queued;
		cl_ulong matinv_submit;
		cl_ulong matinv_start;
		cl_ulong matinv_end;

		profiler.getProfilingInfo <cl_ulong> (CL_PROFILING_COMMAND_QUEUED, &matinv_queued);
		profiler.getProfilingInfo <cl_ulong> (CL_PROFILING_COMMAND_SUBMIT, &matinv_submit);
		profiler.getProfilingInfo <cl_ulong> (CL_PROFILING_COMMAND_START, &matinv_start);
		profiler.getProfilingInfo <cl_ulong> (CL_PROFILING_COMMAND_END, &matinv_end);

		result = assert_inv(M, invM, SIZE); 

		if(result)
			std::cout << "Success!" << std::endl;
		else
			std::cout << "Failed!" << std::endl;
		
		//Profiling info
		std::cout << "Size: " << SIZE << std::endl;
		std::cout << "Memory: " << 2 * M_size  << std::endl;
		std::cout << "Queued: " << matinv_submit - matinv_queued << "ns\n";
		std::cout << "Submit: " << matinv_start - matinv_submit << "ns\n";
		std::cout << "Computation: " << matinv_end - matinv_start << "ns\n";

		//Matrix vector multiplication
		queue.enqueueWriteBuffer(readM, CL_TRUE, 0, M_size, invM);
		queue.enqueueWriteBuffer(readV, CL_TRUE, 0, V_size, V);

		matvecmul.setArg(0, readM);
		matvecmul.setArg(1, readV);
		matvecmul.setArg(2, writeV);

		queue.enqueueNDRangeKernel(matvecmul, cl::NullRange, cl::NDRange(SIZE, SIZE), cl::NDRange(16, 16), NULL, &profiler);
		queue.enqueueReadBuffer(writeV, CL_TRUE, 0, V_size, V2);
		queue.flush();
		profiler.wait();

		profiler.getProfilingInfo <cl_ulong> (CL_PROFILING_COMMAND_QUEUED, &matvecmul_queued);
		profiler.getProfilingInfo <cl_ulong> (CL_PROFILING_COMMAND_SUBMIT, &matvecmul_submit);
		profiler.getProfilingInfo <cl_ulong> (CL_PROFILING_COMMAND_START, &matvecmul_start);
		profiler.getProfilingInfo <cl_ulong> (CL_PROFILING_COMMAND_END, &matvecmul_end);

		//Veryfication
		result = assert(omv(invM, V, SIZE), V2, SIZE); 

		if(result)
			std::cout << "Success!" << std::endl;
		else
			std::cout << "Failed!" << std::endl;
		
		//Profiling info
		std::cout << "Size: " << SIZE << std::endl;
		std::cout << "Memory: " << (2 * V_size) + M_size  << std::endl;
		std::cout << "Queued: " << matvecmul_submit - matvecmul_queued << "ns\n";
		std::cout << "Submit: " << matvecmul_start - matvecmul_submit << "ns\n";
		std::cout << "Computation: " << matvecmul_end - matvecmul_start << "ns\n";
		

		//Sum vectors
		cl::Buffer readV1 = cl::Buffer(context, CL_MEM_READ_ONLY, V_size);

		queue.enqueueWriteBuffer(readV, CL_TRUE, 0, V_size, V1);
		queue.enqueueWriteBuffer(readV1, CL_TRUE, 0, V_size, V2);

		vecadd.setArg(0, readV);
		vecadd.setArg(1, readV1);
		vecadd.setArg(2, writeV);

		queue.enqueueNDRangeKernel(vecadd, cl::NullRange, cl::NDRange(SIZE), cl::NDRange(64), NULL, &profiler);
		queue.enqueueReadBuffer(writeV, CL_TRUE, 0, V_size, P);
		queue.flush();
		profiler.wait();

		cl_ulong vecadd_queued;
		cl_ulong vecadd_submit;
		cl_ulong vecadd_start;
		cl_ulong vecadd_end;

		profiler.getProfilingInfo <cl_ulong> (CL_PROFILING_COMMAND_QUEUED, &vecadd_queued);
		profiler.getProfilingInfo <cl_ulong> (CL_PROFILING_COMMAND_SUBMIT, &vecadd_submit);
		profiler.getProfilingInfo <cl_ulong> (CL_PROFILING_COMMAND_START, &vecadd_start);
		profiler.getProfilingInfo <cl_ulong> (CL_PROFILING_COMMAND_END, &vecadd_end);

		//Veryfication
		V = add(V1, V2, SIZE);
		result = assert(P, V, SIZE); 

		if(result)
			std::cout << "Success!" << std::endl;
		else
			std::cout << "Failed!" << std::endl;
		
		//Profiling info
		std::cout << "Size: " << SIZE << std::endl;
		std::cout << "Memory: " << 2 * V_size << std::endl;
		std::cout << "Queued: " << vecadd_submit - vecadd_queued << "ns\n";
		std::cout << "Submit: " << vecadd_start - vecadd_submit << "ns\n";
		std::cout << "Computation: " << vecadd_end - vecadd_start << "ns\n";

		//Clean up
		free(V);
		free(V1);
		free(V2);
		free(P);
		free(M);
		free(invM);

	} catch(cl::Error error) {
		std::cout << error.what() << "(" << error.err() << ")" << std::endl;
	}

	system("PAUSE");
	exit(0);
}

