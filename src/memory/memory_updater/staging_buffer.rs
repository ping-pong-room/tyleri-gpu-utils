use crate::memory::try_memory_type;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use yarvk::binding_resource::BindingResource;
use yarvk::device::Device;
use yarvk::device_memory::{DeviceMemory, IMemoryRequirements, UnboundResource};
use yarvk::{BoundContinuousBuffer, BufferUsageFlags, ContinuousBuffer, WHOLE_SIZE};
use yarvk::{Buffer, DeviceSize};

pub struct StagingBuffer {
    offset: AtomicU64,
    pub device_memory: DeviceMemory,
    buffer: BoundContinuousBuffer,
}

impl BindingResource for StagingBuffer {
    type RawTy = Buffer;

    fn raw(&self) -> &Self::RawTy {
        &self.buffer
    }

    fn raw_mut(&mut self) -> &mut Self::RawTy {
        &mut self.buffer
    }

    fn offset(&self) -> DeviceSize {
        0
    }

    fn size(&self) -> DeviceSize {
        self.device_memory.size
    }

    fn device(&self) -> &Arc<Device> {
        self.buffer.device()
    }
}

impl StagingBuffer {
    pub(crate) fn new(device: &Arc<Device>, size: DeviceSize) -> Option<Self> {
        let mut buffer_builder = ContinuousBuffer::builder(device);
        buffer_builder.size(size);
        buffer_builder.usage(BufferUsageFlags::TRANSFER_SRC);
        let buffer = buffer_builder.build().unwrap();
        let memory_properties = device.physical_device.memory_properties();
        let device_memory = try_memory_type(
            buffer.get_memory_requirements(),
            memory_properties,
            None,
            size,
            |memory_type| {
                DeviceMemory::builder(memory_type, device)
                    .dedicated_info(buffer.dedicated_info())
                    .allocation_size(size)
                    .build()
                    .ok()
            },
        );
        let mut device_memory = device_memory?;
        device_memory.map_memory(0, WHOLE_SIZE).ok()?;
        let buffer = buffer.bind_memory(&device_memory, 0).ok()?;
        Some(StagingBuffer {
            offset: AtomicU64::new(0),
            device_memory,
            buffer,
        })
    }
    pub(crate) fn write_and_get_offset(
        &self,
        size: DeviceSize,
        f: impl Fn(&mut [u8]),
    ) -> Result<DeviceSize, yarvk::Result> {
        let offset = self.offset.fetch_add(size, Ordering::Relaxed);
        let slice = self.device_memory.get_memory(offset, size)?;
        f(slice);
        Ok(offset)
    }
}
