use yarvk::physical_device::memory_properties::MemoryType;
use yarvk::DeviceSize;
use yarvk::MemoryRequirements;

pub mod array_device_memory;
pub mod block_based_allocator;
pub mod memory_object;
pub mod staging_buffers;

pub fn try_memory_type<T>(
    memory_requirements: &MemoryRequirements,
    interest_types: &[Option<MemoryType>],
    size: DeviceSize,
    f: impl Fn(&MemoryType) -> Option<T>,
) -> Option<T> {
    for memory_type in interest_types.iter().flatten() {
        if memory_requirements.memory_type_bits & (1 << memory_type.index) != 0
            && size < memory_type.heap_size
        {
            if let Some(result) = f(memory_type) {
                return Some(result);
            }
        }
    }
    None
}
