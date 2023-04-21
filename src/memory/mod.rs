use crate::memory::auto_mapped_device_memory::AutoMappedDeviceMemory;
use std::sync::Arc;
use yarvk::binding_resource::BindingResource;
use yarvk::device::Device;
use yarvk::device_memory::mapped_ranges::MappedRanges;
use yarvk::device_memory::UnboundResource;
use yarvk::physical_device::memory_properties::{MemoryType, PhysicalDeviceMemoryProperties};
use yarvk::ImageUsageFlags;
use yarvk::{
    Buffer, ContinuousBufferBuilder, ContinuousImageBuilder, DeviceSize, IBuffer, Image,
    MemoryPropertyFlags,
};
use yarvk::{BufferCopy, BufferUsageFlags};
use yarvk::{MemoryRequirements, UnboundContinuousBuffer, UnboundContinuousImage};

use crate::memory::private::PrivateMemoryBackedResource;
use crate::queue::parallel_recording_queue::ParallelRecordingQueue;

pub mod array_device_memory;
mod auto_mapped_device_memory;
pub mod block_based_memory;
pub mod dedicated_memory;
pub mod memory_updater;
mod private;
pub mod variable_length_buffer;

fn get_aligned_offset(offset: DeviceSize, alignment: DeviceSize) -> DeviceSize {
    (offset + alignment - 1) & (alignment).wrapping_neg()
}

fn copy_buffer_from_start(
    device: &Arc<Device>,
    src_memory: &AutoMappedDeviceMemory,
    dst_memory: &AutoMappedDeviceMemory,
    size: DeviceSize,
) {
    src_memory.map_memory(0, size, Box::new(|_| {})).unwrap();
    dst_memory.map_memory(0, size, Box::new(|_| {})).unwrap();
    let src_memory = src_memory.get_device_memory();
    let dst_memory = dst_memory.get_device_memory();
    if !src_memory
        .memory_type
        .property_flags
        .contains(MemoryPropertyFlags::HOST_COHERENT)
    {
        let mut mapped_ranges = MappedRanges::new(&device);
        mapped_ranges.add_range(&src_memory, 0, size);
        mapped_ranges.invalidate().unwrap();
    }
    let src = src_memory.get_memory(0, size).unwrap();
    let dst = dst_memory.get_memory(0, size).unwrap();

    dst.copy_from_slice(src);
    if !dst_memory
        .memory_type
        .property_flags
        .contains(MemoryPropertyFlags::HOST_COHERENT)
    {
        let mut mapped_ranges = MappedRanges::new(&device);
        mapped_ranges.add_range(&dst_memory, 0, size);
        mapped_ranges.flush().unwrap();
    }
}
fn cmd_copy_buffer_from_start(
    src: Arc<IBuffer>,
    dst: Arc<IBuffer>,
    size: DeviceSize,
    queue: &mut ParallelRecordingQueue,
) {
    queue
        .record(move |command_buffer| {
            command_buffer.cmd_copy_buffer(
                src.clone(),
                dst.clone(),
                &[BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size,
                }],
            );
            Ok(())
        })
        .unwrap();
}

pub fn try_memory_type<T>(
    memory_requirements: &MemoryRequirements,
    physical_device_memory_properties: &PhysicalDeviceMemoryProperties,
    required_flags: Option<MemoryPropertyFlags>,
    estimated_usage_size: DeviceSize,
    f: impl Fn(&MemoryType) -> Option<T>,
) -> Option<T> {
    for memory_type in physical_device_memory_properties.memory_type_in_order() {
        if memory_requirements.memory_type_bits & (1 << memory_type.index) != 0
            && estimated_usage_size < memory_type.heap_size
        {
            if let Some(required_flags) = required_flags {
                if memory_type.property_flags & required_flags != required_flags {
                    continue;
                }
            }
            if let Some(result) = f(memory_type) {
                return Some(result);
            }
        }
    }
    None
}

pub type MemBakRes<T> = dyn MemoryBackedResource<RawTy = T>;

pub type IMemBakBuf = MemBakRes<Buffer>;
pub type IMemBakImg = MemBakRes<Image>;

pub trait MemoryBackedResource: BindingResource + PrivateMemoryBackedResource {}

pub trait MemoryObjectBuilder {
    type Ty: UnboundResource + 'static;
    type Usage;
    fn build(&self) -> Result<Self::Ty, yarvk::Result>;
}

impl MemoryObjectBuilder for ContinuousBufferBuilder {
    type Ty = UnboundContinuousBuffer;
    type Usage = BufferUsageFlags;

    fn build(&self) -> Result<Self::Ty, yarvk::Result> {
        self.build()
    }
}

impl MemoryObjectBuilder for ContinuousImageBuilder {
    type Ty = UnboundContinuousImage;
    type Usage = ImageUsageFlags;

    fn build(&self) -> Result<Self::Ty, yarvk::Result> {
        self.build()
    }
}
