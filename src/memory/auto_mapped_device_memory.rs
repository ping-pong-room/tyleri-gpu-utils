use std::hash::{Hash, Hasher};
use std::sync::{RwLock, RwLockReadGuard};
use yarvk::device_memory::DeviceMemory;
use yarvk::physical_device::memory_properties::MemoryType;
use yarvk::WHOLE_SIZE;
use yarvk::{DeviceSize, Handle};

pub struct AutoMappedDeviceMemory {
    pub memory_type: MemoryType,
    pub size: DeviceSize,
    handle: u64,
    device_memory: RwLock<DeviceMemory>,
}

impl PartialEq for AutoMappedDeviceMemory {
    fn eq(&self, other: &Self) -> bool {
        self.handle == other.handle
    }
}

impl Eq for AutoMappedDeviceMemory {}

impl Hash for AutoMappedDeviceMemory {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.handle.hash(state)
    }
}

impl AutoMappedDeviceMemory {
    pub fn new(device_memory: DeviceMemory) -> Self {
        let memory_type = device_memory.memory_type.clone();
        Self {
            memory_type,
            size: device_memory.size,
            handle: device_memory.handle(),
            device_memory: RwLock::new(device_memory),
        }
    }
    pub fn map_memory(
        &self,
        offset: DeviceSize,
        size: DeviceSize,
        f: &dyn Fn(&mut [u8]),
    ) -> Result<(), yarvk::Result> {
        {
            let device_memory = self.device_memory.read().unwrap();
            if let Ok(memory) = device_memory.get_memory(offset, size) {
                f(memory);
                return Ok(());
            }
        }
        let mut device_memory = self.device_memory.write().unwrap();
        device_memory.unmap_memory();
        if device_memory.map_memory(0, WHOLE_SIZE).is_err() {
            device_memory.map_memory(offset, size)?;
        }
        drop(device_memory);

        {
            let device_memory = self.device_memory.read().unwrap();
            if let Ok(memory) = device_memory.get_memory(offset, size) {
                f(memory);
                return Ok(());
            }
        }
        // mapped memory changed after we re-mapped memory, this time call f in a write lock
        let mut device_memory = self.device_memory.write().unwrap();
        device_memory.unmap_memory();
        if device_memory.map_memory(0, WHOLE_SIZE).is_err() {
            device_memory.map_memory(offset, size)?;
        }
        let memory = device_memory.get_memory(offset, size).unwrap();
        f(memory);
        Ok(())
    }
    pub fn get_device_memory(&self) -> RwLockReadGuard<DeviceMemory> {
        self.device_memory.read().unwrap()
    }
}
