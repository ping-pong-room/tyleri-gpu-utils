use crate::memory::auto_mapped_device_memory::AutoMappedDeviceMemory;
use crate::memory::private::PrivateMemoryBackedResource;
use crate::memory::{MemoryBackedResource, MemoryResource};
use std::sync::Arc;
use yarvk::binding_resource::BindingResource;
use yarvk::device::Device;
use yarvk::device_memory::{DeviceMemory, UnboundResource};
use yarvk::physical_device::memory_properties::MemoryType;
use yarvk::DeviceSize;
use yarvk::MemoryPropertyFlags;

struct DedicatedResourceMemory<T: UnboundResource> {
    object: T::BoundType,
    device_memory: Arc<AutoMappedDeviceMemory>,
}

impl<T: UnboundResource> BindingResource for DedicatedResourceMemory<T> {
    type RawTy = T::RawTy;

    fn raw(&self) -> &Self::RawTy {
        self.object.raw()
    }

    fn raw_mut(&mut self) -> &mut Self::RawTy {
        self.object.raw_mut()
    }

    fn offset(&self) -> DeviceSize {
        self.object.offset()
    }

    fn size(&self) -> DeviceSize {
        self.object.size()
    }

    fn device(&self) -> &Arc<Device> {
        self.object.device()
    }
}

impl<T: UnboundResource> PrivateMemoryBackedResource for DedicatedResourceMemory<T> {
    fn memory_property_flags(&self) -> MemoryPropertyFlags {
        self.device_memory.memory_type.property_flags
    }

    fn memory_memory(
        &mut self,
        offset: DeviceSize,
        size: DeviceSize,
        f: &dyn Fn(&mut [u8]),
    ) -> Result<(), yarvk::Result> {
        assert!(offset + size <= self.size());
        let offset = self.offset() + offset;
        self.device_memory.map_memory(offset, size, f)
    }

    fn get_device_memory(&self) -> Arc<AutoMappedDeviceMemory> {
        self.device_memory.clone()
    }
}

impl<T: UnboundResource> MemoryBackedResource for DedicatedResourceMemory<T> {}

pub fn new_dedicated_resource<T: UnboundResource + 'static>(
    resource: T,
    memory_type: &MemoryType,
) -> Result<Arc<MemoryResource<T::RawTy>>, yarvk::Result> {
    let memory_requirements = resource.get_memory_requirements();
    let device_memory = DeviceMemory::builder(memory_type, resource.device())
        .allocation_size(memory_requirements.size)
        .dedicated_info(resource.dedicated_info())
        .build()?;
    let resource = resource.bind_memory(&device_memory, 0)?;
    Ok(Arc::new(DedicatedResourceMemory::<T> {
        object: resource,
        device_memory: Arc::new(AutoMappedDeviceMemory::new(device_memory)),
    }) as Arc<MemoryResource<T::RawTy>>)
}
