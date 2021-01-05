#include <thrust/mr/pool.h>
#include <thrust/system/cuda/memory_resource.h>

// do not extend function signatures for CUDA.
#define GPU_HOST_ATTR

#include <gpusat_types.h>

namespace gpusat {

    void* PinnedSuballocator::allocate(size_t bytes) {
        if (is_pinned) {
            return allocator.allocate(bytes, sizeof(uint64_t)).get();
        } else {
            return unpinned.allocate(bytes, sizeof(uint64_t));
        }
    }

    void PinnedSuballocator::deallocate(void* p, size_t bytes) {
        if (is_pinned) {
            allocator.deallocate(p, bytes, sizeof(uint64_t));
        } else {
            unpinned.deallocate(p, bytes, sizeof(uint64_t));
        }
    }

    void PinnedSuballocator::deinit() {
        allocator.release();
    }

    PinnedSuballocator::PinnedSuballocator(bool pinned) : is_pinned(pinned) {
        unpinned = thrust::mr::new_delete_resource();
        allocator = thrust::mr::disjoint_unsynchronized_pool_resource<thrust::system::cuda::universal_host_pinned_memory_resource, thrust::mr::new_delete_resource>();
    }
}
