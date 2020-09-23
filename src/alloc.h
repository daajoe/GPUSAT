#ifndef ALLOC_H
#define ALLOC_H

#include <thrust/mr/new.h>
#include <thrust/mr/disjoint_pool.h>
#include <thrust/system/cuda/memory_resource.h>

namespace gpusat {
    class PinnedSuballocator {
        public:
            PinnedSuballocator();
            void* allocate(size_t bytes);
            void deallocate(void* p, size_t bytes);
            void deinit();
        protected:
            thrust::mr::disjoint_unsynchronized_pool_resource<thrust::system::cuda::universal_host_pinned_memory_resource, thrust::mr::new_delete_resource> allocator;
    };
}

#endif // ALLOC_H
