#include <CL/cl.hpp>

#include <iostream>
#include <fstream>

#ifndef DEBUG
#define DEBUG false
#endif

#ifndef COMPUTE_HPP  // include guard
#define COMPUTE_HPP


class Compute
{
    public:
        // Platform
        std::vector< cl::Platform > platforms;
        cl::Platform platform;
        // Device
        std::vector< cl::Device > devices;
        cl::Device device;
        // Context
        cl::Context context;
        // Buffer
        std::vector< cl::Buffer* > buffers;
        // Program
        cl::Program program;
        const static std::string kernel_src;
        // Kernel
        cl::Kernel kernel;
        std::string kernel_name;
        // CommandQueue
        cl::CommandQueue command_queue;


        Compute(std::string name)
            : kernel_name(name)
        {
            // Platform
            this->init_platform();

            // Device
            this->init_device();

            // Context
            this->init_context();

            // Program
            this->init_program();

            // Kernel
            this->init_kernel();

            // Command Queue
            this->init_command_queue();
        }

        ~Compute()
        {
            for(auto &i: this->buffers)
                delete i;
        }

        void run(const int dm1, const int dm2=0, const int dm3=0)
        {
            cl::NDRange global_range;

           if (dm3 != 0)
                global_range = cl::NDRange(dm1, dm2, dm3);
            else if (dm2 !=0)
                global_range = cl::NDRange(dm1, dm2);
            else
                global_range = cl::NDRange(dm1);

            cl::Event event;
            this->command_queue.enqueueNDRangeKernel(
                this->kernel,
                cl::NullRange,  // offset
                global_range,  // global
                cl::NullRange,  // local
                NULL,  // events
                &event  // event
                );
            event.wait();
        }

        template<typename T>
        void set_buffer(T *buffs,
            const int shape,
            cl_mem_flags flags=CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR)
        {
            cl_int err;

            cl::Buffer *buf = new cl::Buffer(
                this->context,
                buffs,
                buffs + shape,
                true,  // readonly
                true,  // usehostptr
                &err
                );

            this->buffers.push_back(buf);

            check_err(err, "cl::Buffer constructor");

            // set kernel argument
            if (DEBUG)
            {
                std::cout << "[DEBUG] cl::Kernel::setArg "
                          << this->buffers.size() - 1 << std::endl;
            }
            err = this->kernel.setArg(this->buffers.size() - 1, *buf);
            check_err(err, "cl::Kernel::Kernel");
        }

        template<typename T>
        void set_ret_buffer(T *buffs, const int shape)
        {
            this->set_buffer(buffs, shape, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR);
        }

    private:

        void check_err(cl_int err, const char * name)
        {
            if (err == CL_SUCCESS)
                return;

            std::cerr << "[ERROR] " << name << " (" << err << ")" << std::endl;
            exit(1);
        }

        void init_platform()
        {
            cl::Platform::get(&this->platforms);
            this->check_err(platforms.size()? CL_SUCCESS : -1, "cl::Platform::get");

            this->platform = platforms[0];

            // debug
            std::string name;
            platform.getInfo(CL_PLATFORM_NAME, &name);
            std::cout << "Platform:\t" << name << std::endl;
        }

        void init_device()
        {
            platform.getDevices(CL_DEVICE_TYPE_CPU, &this->devices);
            this->check_err(devices.size()? CL_SUCCESS : -1, "cl:Platform::getDevices");

            this->device = this->devices[0];

            // debug
            std::string dev_info;
            for(auto &dev: devices)
            {
                dev.getInfo(CL_DEVICE_NAME, &dev_info);
                std::cout << "Device:\t" << dev_info << std::endl;
            }
        }

        void init_context()
        {
            cl_int err;

            this->context = cl::Context::Context(devices,
                NULL,  // property
                NULL,  // call back
                NULL,  // user_data
                &err
                );
            check_err(err, "cl::Context::Context");
        }

        void init_program()
        {
            cl_int err;
            this->program = cl::Program::Program(
                    context,
                    this->kernel_src,
                    true,  // build
                    &err);

            // debug
            if (DEBUG)
            {
                std::string tmps;
                program.getBuildInfo(this->device, CL_PROGRAM_BUILD_LOG, &tmps);
                std::cout << "[cl::Program] Build log ==========================" << std::endl;
                std::cout << tmps << std::endl;
                std::cout << "[cl::Program] End of build log ===================" << std::endl;
                check_err(err, "cl::Program::Program");
            }
        }

        void init_kernel()
        {
            cl_int err;
            this->kernel = cl::Kernel::Kernel(program, this->kernel_name.c_str(), &err);

            check_err(err, "cl::Kernel::Kernel");
        }

        void init_command_queue()
        {
            cl_int err;
            this->command_queue = cl::CommandQueue::CommandQueue(
                this->context,
                0,  // property
                &err);

            check_err(err, "cl::CommandQueue::CommandQueue");
        }
};

#endif /* end of include guard: COMPUTE_HPP */

#include "kernel.hpp"
