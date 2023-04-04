use yarvk::binding_resource::BindingResource;
use yarvk::device_memory::UnboundResource;
use yarvk::physical_device::memory_properties::{MemoryType, PhysicalDeviceMemoryProperties};
use yarvk::{
    Buffer, ContinuousBufferBuilder, ContinuousImageBuilder, DeviceSize, Image, MemoryPropertyFlags,
};
use yarvk::{MemoryRequirements, UnboundContinuousBuffer, UnboundContinuousImage};

use crate::memory::private::PrivateMemoryBackedResource;

pub mod array_device_memory;
mod auto_mapped_device_memory;
pub mod block_based_memory;
pub mod dedicated_memory;
pub mod memory_updater;
mod private;

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
    fn build(&self) -> Result<Self::Ty, yarvk::Result>;
}

impl MemoryObjectBuilder for ContinuousBufferBuilder {
    type Ty = UnboundContinuousBuffer;

    fn build(&self) -> Result<Self::Ty, yarvk::Result> {
        self.build()
    }
}

impl MemoryObjectBuilder for ContinuousImageBuilder {
    type Ty = UnboundContinuousImage;

    fn build(&self) -> Result<Self::Ty, yarvk::Result> {
        self.build()
    }
}
