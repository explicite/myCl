#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#include <iostream>
#include <fstream>
#include <string>

#include "test.hpp"

#define AMD_RADEON_HD_5670

#ifdef AMD_RADEON_HD_5670
#define NR_COMP_UNITS 5
#define NR_CORES_PER_CU 20
#endif

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

		size_t COMPUTE_UNITS = device.getInfo < CL_DEVICE_MAX_COMPUTE_UNITS > ();
		std::cout << "Compute units: " << COMPUTE_UNITS << std::endl;

		size_t MAX_WORK_ITEMS_DIMENSION = device.getInfo < CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS > ();
		std::cout << "Work item dimensions: " << device.getInfo < CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS > () << std::endl;

		size_t MAX_WORK_GROUP_SIZE = device.getInfo < CL_DEVICE_MAX_WORK_GROUP_SIZE > ();
		std::cout << "Work group size: " << MAX_WORK_GROUP_SIZE << std::endl;

		std::cout << "Preferreed group size: " << CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE << std::endl;

		std::vector <size_t> MAX_WORK_ITEMS = device.getInfo < CL_DEVICE_MAX_WORK_ITEM_SIZES > ();
		std::cout << "Working items: ";
		for(int i = 0; i < MAX_WORK_ITEMS_DIMENSION; i++)
			std::cout << MAX_WORK_ITEMS[i] << " ";

		std::cout << std::endl;

		std::cout << "Kernel group size: " << CL_KERNEL_WORK_GROUP_SIZE << std::endl;
		std::cout << "Profile: " << device.getInfo < CL_DEVICE_PROFILE > ()  << std::endl;
		std::cout << "Version: " << device.getInfo < CL_DEVICE_OPENCL_C_VERSION > ()  << std::endl;

		cl::Context context(devices);
		cl::CommandQueue queue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);
		
		//Size for vector and matrix
		const unsigned int SIZE = MAX_WORK_GROUP_SIZE * COMPUTE_UNITS;
		size_t V_size = sizeof(float) * SIZE;
		size_t M_size = sizeof(float) * SIZE * SIZE;

		//Prepare standard kernels
		//Matrix - Vector Multiplication
		std::ifstream matvecmul_src("kernel/matvecmul.cl");
		std::string matvecmul_code(std::istreambuf_iterator<char>(matvecmul_src), (std::istreambuf_iterator<char>()));
		cl::Program::Sources matvecmul_source(1, std::make_pair(matvecmul_code.c_str(), matvecmul_code.length() + 1));
		cl::Program matvecmul_program = cl::Program(context, matvecmul_source);
		matvecmul_program.build(devices);
		cl::Kernel matvecmul(matvecmul_program, "matvecmul");

		//Add Vectors
		std::ifstream vecadd_src("kernel/vecadd.cl");
		std::string vecadd_code(std::istreambuf_iterator<char>(vecadd_src), (std::istreambuf_iterator<char>()));
		cl::Program::Sources vecadd_source(1, std::make_pair(vecadd_code.c_str(), vecadd_code.length() + 1));
		cl::Program vecadd_program = cl::Program(context, vecadd_source);
		vecadd_program.build(devices);
		cl::Kernel vecadd(vecadd_program, "vecadd");

		//Invers Matrix
		std::ifstream mattrans_src("kernel/mattrans.cl");
		std::string mattrans_code(std::istreambuf_iterator<char>(mattrans_src), (std::istreambuf_iterator<char>()));
		cl::Program::Sources mattrans_source(1, std::make_pair(mattrans_code.c_str(), mattrans_code.length() + 1));
		cl::Program mattrans_program = cl::Program(context, mattrans_source);
		mattrans_program.build(devices);
		cl::Kernel matinv(mattrans_program, "mattrans");
		
		//Prepare optimized kernels
		//Matrix - Vector Multiplication
		std::ifstream opt_matvecmul_src("kernel/opt_matvecmul.cl");
		std::string opt_matvecmul_code(std::istreambuf_iterator<char>(opt_matvecmul_src), (std::istreambuf_iterator<char>()));
		cl::Program::Sources opt_matvecmul_source(1, std::make_pair(opt_matvecmul_code.c_str(), opt_matvecmul_code.length() + 1));
		cl::Program opt_matvecmul_program = cl::Program(context, opt_matvecmul_source);
		opt_matvecmul_program.build(devices);
		cl::Kernel opt_matvecmul(opt_matvecmul_program, "opt_matvecmul");

		//Add Vectors
		std::ifstream opt_vecadd_src("kernel/opt_vecadd.cl");
		std::string opt_vecadd_code(std::istreambuf_iterator<char>(opt_vecadd_src), (std::istreambuf_iterator<char>()));
		cl::Program::Sources opt_vecadd_source(1, std::make_pair(opt_vecadd_code.c_str(), opt_vecadd_code.length() + 1));
		cl::Program opt_vecadd_program = cl::Program(context, opt_vecadd_source);
		opt_vecadd_program.build(devices);
		cl::Kernel opt_vecadd(opt_vecadd_program, "opt_vecadd");

		//Invers Matrix
		std::ifstream opt_mattrans_src("kernel/opt_mattrans.cl");
		std::string opt_mattrans_code(std::istreambuf_iterator<char>(opt_mattrans_src), (std::istreambuf_iterator<char>()));
		cl::Program::Sources opt_mattrans_source(1, std::make_pair(opt_mattrans_code.c_str(), opt_mattrans_code.length() + 1));
		cl::Program opt_mattrans_program = cl::Program(context, opt_mattrans_source);
		opt_mattrans_program.build(devices);
		cl::Kernel opt_mattrans(opt_mattrans_program, "opt_mattrans");

		//Matrix multiplication
		std::ifstream opt_mat_mul_src("kernel/opt_mat_mul.cl");
		std::string opt_mat_mul_code(std::istreambuf_iterator<char>(opt_mat_mul_src), (std::istreambuf_iterator<char>()));
		cl::Program::Sources opt_mat_mul_source(1, std::make_pair(opt_mat_mul_code.c_str(), opt_mat_mul_code.length() + 1));
		cl::Program opt_mat_mul_program = cl::Program(context, opt_mat_mul_source);
		opt_mat_mul_program.build(devices);
		cl::Kernel opt_mat_mul(opt_mat_mul_program, "opt_mat_mul");

		//Profiler
		cl::Event profiler;

		//Preapare data
		float* V = (float*)malloc(V_size);
		float* V1 = (float*)malloc(V_size);
		float* M = (float*)malloc(M_size);
		float* transM = (float*)malloc(M_size);
		float* mulM = (float*)malloc(M_size);
		float* V2 = (float*)malloc(V_size);
		float* P = (float*)malloc(V_size);

		register int i;
		#pragma omp parallel for schedule(static) private(i)
		for(i = 0; i < SIZE; i++){
			V[i] = i*2;
			V1[i] = 0;
			V2[i] = 0;
			P[i] = 0;
		}	

		#pragma omp parallel for schedule(static) private(i)
		for(i = 0; i < SIZE * SIZE; i++){
			M[i] = i;
			transM[i] = 0;
			mulM[i] = 0;
		}	
		
		
		//Matrix vector multiplication
		cl::Buffer readM = cl::Buffer(context, CL_MEM_READ_ONLY, M_size);
		cl::Buffer readV = cl::Buffer(context, CL_MEM_READ_ONLY, V_size);
		cl::Buffer writeV1 = cl::Buffer(context, CL_MEM_WRITE_ONLY, V_size);
		
		queue.enqueueWriteBuffer(readM, CL_TRUE, 0, M_size, M);
		queue.enqueueWriteBuffer(readV, CL_TRUE, 0, V_size, V);
		
		opt_matvecmul.setArg(0, readM);
		opt_matvecmul.setArg(1, SIZE);
		opt_matvecmul.setArg(2, SIZE);
		opt_matvecmul.setArg(3, readV);
		opt_matvecmul.setArg(4, writeV1);
		opt_matvecmul.setArg(5, cl::Local(MAX_WORK_GROUP_SIZE*sizeof(float)));

		queue.enqueueNDRangeKernel(opt_matvecmul, cl::NullRange, cl::NDRange((SIZE/MAX_WORK_GROUP_SIZE)*MAX_WORK_GROUP_SIZE), cl::NDRange(MAX_WORK_GROUP_SIZE), NULL, &profiler);
		queue.enqueueReadBuffer(writeV1, CL_TRUE, 0, V_size, V1);
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
		bool result = assert(mat_vec(M, V, SIZE), V1, SIZE); 

		if(result)
			std::cout << "\nMatrix-Vector: Success!" << std::endl;
		else
			std::cout << "\nMatrix-Vector: Failed!" << std::endl;
		
		//Profiling info
		std::cout << "Size: " << SIZE << std::endl;
		std::cout << "Memory: " << (2 * V_size) + M_size  << std::endl;
		std::cout << "Queued: " << matvecmul_submit - matvecmul_queued << "ns\n";
		std::cout << "Submit: " << matvecmul_start - matvecmul_submit << "ns\n";
		std::cout << "Computation: " << matvecmul_end - matvecmul_start << "ns\n";
		std::cout << "Performance: " << (2.0*(double)SIZE*(double)SIZE/((double)matvecmul_end-(double)matvecmul_start))  << " GFlops\n";
		std::cout << "Transfer speed: " << (sizeof(float)*2.0*(double)SIZE*(double)SIZE/((double)matvecmul_end-(double)matvecmul_start))  << " GB/s\n";


		//Matrix invers
		cl::Buffer writeM = cl::Buffer(context, CL_MEM_WRITE_ONLY, M_size);

		queue.enqueueWriteBuffer(readM, CL_TRUE, 0, M_size, M);

		opt_mattrans.setArg(0, readM);
		opt_mattrans.setArg(1, writeM);

		queue.enqueueNDRangeKernel(opt_mattrans, cl::NullRange, cl::NDRange(SIZE, SIZE), cl::NDRange(16, 16), NULL, &profiler);
		queue.enqueueReadBuffer(writeM, CL_TRUE, 0, M_size, transM);
		queue.flush();
		profiler.wait();

		cl_ulong mattrans_queued;
		cl_ulong mattrans_submit;
		cl_ulong mattrans_start;
		cl_ulong mattrans_end;

		profiler.getProfilingInfo <cl_ulong> (CL_PROFILING_COMMAND_QUEUED, &mattrans_queued);
		profiler.getProfilingInfo <cl_ulong> (CL_PROFILING_COMMAND_SUBMIT, &mattrans_submit);
		profiler.getProfilingInfo <cl_ulong> (CL_PROFILING_COMMAND_START, &mattrans_start);
		profiler.getProfilingInfo <cl_ulong> (CL_PROFILING_COMMAND_END, &mattrans_end);

		result = assert_inv(M, transM, SIZE); 

		if(result)
			std::cout << "\nMatrix trans: Success!" << std::endl;
		else
			std::cout << "\nMatrix trans: Failed!" << std::endl;
		
		//Profiling info
		std::cout << "Size: " << SIZE << std::endl;
		std::cout << "Memory: " << 2 * M_size  << std::endl;
		std::cout << "Queued: " << mattrans_submit - mattrans_queued << "ns\n";
		std::cout << "Submit: " << mattrans_start - mattrans_submit << "ns\n";
		std::cout << "Computation: " << mattrans_end - mattrans_start << "ns\n";
		std::cout << "Transfer speed: " << (2.0*(double)SIZE*sizeof(float)/((double)mattrans_end-(double)mattrans_start))  << " GB/s\n";


		//Matrix vector multiplication
		cl::Buffer writeV2 = cl::Buffer(context, CL_MEM_WRITE_ONLY, V_size);

		queue.enqueueWriteBuffer(readM, CL_TRUE, 0, M_size, transM);
		queue.enqueueWriteBuffer(readV, CL_TRUE, 0, V_size, V);

		opt_matvecmul.setArg(0, readM);
		opt_matvecmul.setArg(1, SIZE);
		opt_matvecmul.setArg(2, SIZE);
		opt_matvecmul.setArg(3, readV);
		opt_matvecmul.setArg(4, writeV2);
		opt_matvecmul.setArg(5, cl::Local(MAX_WORK_GROUP_SIZE*sizeof(float)));

		queue.enqueueNDRangeKernel(opt_matvecmul, cl::NullRange, cl::NDRange((SIZE/COMPUTE_UNITS) * MAX_WORK_GROUP_SIZE), cl::NDRange(MAX_WORK_GROUP_SIZE), NULL, &profiler);
		queue.enqueueReadBuffer(writeV2, CL_TRUE, 0, V_size, V2);
		queue.flush();
		profiler.wait();

		profiler.getProfilingInfo <cl_ulong> (CL_PROFILING_COMMAND_QUEUED, &matvecmul_queued);
		profiler.getProfilingInfo <cl_ulong> (CL_PROFILING_COMMAND_SUBMIT, &matvecmul_submit);
		profiler.getProfilingInfo <cl_ulong> (CL_PROFILING_COMMAND_START, &matvecmul_start);
		profiler.getProfilingInfo <cl_ulong> (CL_PROFILING_COMMAND_END, &matvecmul_end);

		//Veryfication
		result = assert(mat_vec(transM, V, SIZE), V2, SIZE); 

		if(result)
			std::cout << "\nMatrix-Vector: Success!" << std::endl;
		else
			std::cout << "\nMatrix-Vector: Failed!" << std::endl;
		
		//Profiling info
		std::cout << "Size: " << SIZE << std::endl;
		std::cout << "Memory: " << (2 * V_size) + M_size  << std::endl;
		std::cout << "Queued: " << matvecmul_submit - matvecmul_queued << "ns\n";
		std::cout << "Submit: " << matvecmul_start - matvecmul_submit << "ns\n";
		std::cout << "Computation: " << matvecmul_end - matvecmul_start << "ns\n";
		std::cout << "Performance: " << (2.0*(double)SIZE*(double)SIZE/((double)matvecmul_end-(double)matvecmul_start))  << " GFlops\n";
		std::cout << "Transfer speed: " << (sizeof(float)*2.0*(double)SIZE*(double)SIZE/((double)matvecmul_end-(double)matvecmul_start))  << " GB/s\n";


		//Sum vectors
		cl::Buffer readV1 = cl::Buffer(context, CL_MEM_READ_ONLY, V_size);
		cl::Buffer writeP = cl::Buffer(context, CL_MEM_READ_ONLY, V_size);

		queue.enqueueWriteBuffer(readV, CL_TRUE, 0, V_size, V1);
		queue.enqueueWriteBuffer(readV1, CL_TRUE, 0, V_size, V2);

		opt_vecadd.setArg(0, readV);
		opt_vecadd.setArg(1, readV1);
		opt_vecadd.setArg(2, writeP);
		opt_vecadd.setArg(3, SIZE);

		queue.enqueueNDRangeKernel(opt_vecadd, cl::NullRange, cl::NDRange(COMPUTE_UNITS * MAX_WORK_GROUP_SIZE), cl::NDRange(MAX_WORK_GROUP_SIZE), NULL, &profiler);
		queue.enqueueReadBuffer(writeP, CL_TRUE, 0, V_size, P);
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
		result = assert(P, add(V1, V2, SIZE), SIZE); 

		if(result)
			std::cout << "\nAdd Vectors: Success!" << std::endl;
		else
			std::cout << "\nAdd Vectors: Failed!" << std::endl;
		
		//Profiling info
		std::cout << "Size: " << SIZE << std::endl;
		std::cout << "Memory: " << 2 * V_size << std::endl;
		std::cout << "Queued: " << vecadd_submit - vecadd_queued << "ns\n";
		std::cout << "Submit: " << vecadd_start - vecadd_submit << "ns\n";
		std::cout << "Computation: " << vecadd_end - vecadd_start << "ns\n";
		std::cout << "Performance: " << ((double)SIZE/((double)vecadd_end-(double)vecadd_start))  << " GFlops\n";
		std::cout << "Transfer speed: " << (sizeof(float)*(double)SIZE/((double)vecadd_end-(double)vecadd_start))  << " GB/s\n";

	
		//Matrix multiplication
		cl::Buffer read_invM = cl::Buffer(context, CL_MEM_READ_ONLY, M_size);
		cl::Buffer write_mulM = cl::Buffer(context, CL_MEM_WRITE_ONLY, M_size);

		queue.enqueueWriteBuffer(readM, CL_TRUE, 0, M_size, M);
		queue.enqueueWriteBuffer(read_invM, CL_TRUE, 0, M_size, transM);

		opt_mat_mul.setArg(0, readM);
		opt_mat_mul.setArg(1, read_invM);
		opt_mat_mul.setArg(2, write_mulM);
		opt_mat_mul.setArg(3, SIZE);

		queue.enqueueNDRangeKernel(opt_mat_mul, cl::NullRange, cl::NDRange(SIZE, SIZE), cl::NDRange(16, 16), NULL, &profiler);
		queue.enqueueReadBuffer(write_mulM, CL_TRUE, 0, M_size, mulM);
		queue.flush();
		profiler.wait();

		cl_ulong opt_mat_mul_queued;
		cl_ulong opt_mat_mul_submit;
		cl_ulong opt_mat_mul_start;
		cl_ulong opt_mat_mul_end;

		profiler.getProfilingInfo <cl_ulong> (CL_PROFILING_COMMAND_QUEUED, &opt_mat_mul_queued);
		profiler.getProfilingInfo <cl_ulong> (CL_PROFILING_COMMAND_SUBMIT, &opt_mat_mul_submit);
		profiler.getProfilingInfo <cl_ulong> (CL_PROFILING_COMMAND_START, &opt_mat_mul_start);
		profiler.getProfilingInfo <cl_ulong> (CL_PROFILING_COMMAND_END, &opt_mat_mul_end);

		//Veryfication
		result = assert(mat_mul(transM, M, SIZE), mulM, SIZE*SIZE); 

		if(result)
			std::cout << "\nMatrix multiplication: Success!" << std::endl;
		else
			std::cout << "\nMatrix multiplication: Failed!" << std::endl;
		
		double operations = SIZE*(SIZE*(SIZE*2));
		double time = (double)opt_mat_mul_end - (double)opt_mat_mul_start;
	
		//Profiling info
		std::cout << "Size: " << SIZE << std::endl;
		std::cout << "Memory: " << (2 * V_size) + M_size  << std::endl;
		std::cout << "Queued: " << opt_mat_mul_submit - opt_mat_mul_queued << "ns\n";
		std::cout << "Submit: " << opt_mat_mul_start - opt_mat_mul_submit << "ns\n";
		std::cout << "Computation: " << opt_mat_mul_end - opt_mat_mul_start << "ns\n";
		std::cout << "Performance: " << operations/time<< " GFlops\n";
		std::cout << "Transfer speed: " << sizeof(float)*operations/time << " GB/s\n";


		//Clean up
		free(V);
		free(V1);
		free(V2);
		free(P);
		free(M);
		free(transM); 
		free(mulM);

	} catch(cl::Error error) {
		std::cout << error.what() << "(" << error.err() << ")" << std::endl;
	}

	system("PAUSE");
	exit(0);
}

