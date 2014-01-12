#define __CL_ENABLE_EXCEPTIONS
#include < CL/cl.hpp >
#include < iostream >
#include < fstream >
#include < string >

int main()
{
	const int N_ELEMENTS = 1024;
	float* a = new float[N_ELEMENTS];
	float* b = new float[N_ELEMENTS];
	float* c = new float[N_ELEMENTS];

	register int i;
	for(i = 0; i < N_ELEMENTS; i++)
	{
		a[i] = i;
		b[i] = i;
	}

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
		default_platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
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
		cl::Buffer buffera= cl::Buffer(context, CL_MEM_READ_ONLY, N_ELEMENTS * sizeof(float));
		cl::Buffer bufferb= cl::Buffer(context, CL_MEM_READ_ONLY, N_ELEMENTS * sizeof(float));
		cl::Buffer bufferc= cl::Buffer(context, CL_MEM_WRITE_ONLY, N_ELEMENTS * sizeof(float));

		queue.enqueueWriteBuffer(buffera, CL_TRUE, 0, N_ELEMENTS * sizeof(float), a);
		queue.enqueueWriteBuffer(bufferb, CL_TRUE, 0, N_ELEMENTS * sizeof(float), b);

		//Compiler
		std::ifstream src("vva.cl");
		std::string code(std::istreambuf_iterator < char > (src), (std::istreambuf_iterator < char > ()));
		cl::Program::Sources source(1, std::make_pair(code.c_str(), code.length() + 1));
		cl::Program program = cl::Program(context, source);
		program.build(devices);
		cl::Kernel vva(program, "vva");

		vva.setArg(0, buffera);
		vva.setArg(1, bufferb);
		vva.setArg(2, bufferc);

		cl::Event event;
		cl_ulong queued;
		cl_ulong submit;
		cl_ulong start;
		cl_ulong end;

		cl::NDRange global(N_ELEMENTS);
		cl::NDRange local(256);
		queue.enqueueTask(vva, NULL, &event);
		queue.enqueueNDRangeKernel(vva, cl::NullRange, global, local);
		queue.enqueueReadBuffer(bufferc, CL_TRUE, 0, N_ELEMENTS * sizeof(float), c);
		queue.flush();
		event.wait();

		event.getProfilingInfo < cl_ulong > (CL_PROFILING_COMMAND_QUEUED, &queued);
		event.getProfilingInfo < cl_ulong > (CL_PROFILING_COMMAND_SUBMIT, &submit);
		event.getProfilingInfo < cl_ulong > (CL_PROFILING_COMMAND_START, &start);
		event.getProfilingInfo < cl_ulong > (CL_PROFILING_COMMAND_END, &end);

		bool result = true;
		for(i = 0; i < N_ELEMENTS; i++) 
		{
			if(c[i] != a[i] + b[i]){
				result = false;
				break;
			}
		}
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
