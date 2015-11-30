

const std::string Compute::kernel_src = R"(
kernel void add(global float *a, global float *b, global float *ret)
{
	const int row = get_global_id(0);

	ret[row] = a[row] + b[row];
})";
