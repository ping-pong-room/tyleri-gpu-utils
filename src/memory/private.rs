use crate::memory::auto_mapped_device_memory::AutoMappedDeviceMemory;
use std::sync::Arc;
use yarvk::DeviceSize;
use yarvk::MemoryPropertyFlags;

pub trait PrivateMemoryBackedResource {
    fn memory_property_flags(&self) -> MemoryPropertyFlags;
    fn map_memory(
        &self,
        offset: DeviceSize,
        size: DeviceSize,
        f: &dyn Fn(&mut [u8]),
    ) -> Result<(), yarvk::Result>;
    fn get_device_memory(&self) -> Arc<AutoMappedDeviceMemory>;
}
