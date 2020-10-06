#include <thrust/mr/pool.h>
#include <thrust/system/cuda/memory_resource.h>

// do not extend function signatures for CUDA.
#define GPU_HOST_ATTR

#include <gpusat_types.h>

namespace gpusat {
    PinnedSuballocator::PinnedSuballocator() {
        allocator = thrust::mr::disjoint_unsynchronized_pool_resource<thrust::system::cuda::universal_host_pinned_memory_resource, thrust::mr::new_delete_resource>();
    }

    void* PinnedSuballocator::allocate(size_t bytes) {
        return allocator.allocate(bytes, sizeof(uint64_t)).get();
    }

    void PinnedSuballocator::deallocate(void* p, size_t bytes) {
        allocator.deallocate(p, bytes, sizeof(uint64_t));
    }

    void PinnedSuballocator::deinit() {
        allocator.release();
    }
}
