use std::sync::Arc;
use yarvk::device::Device;
use yarvk::device_memory::{DeviceMemory, IMemoryRequirements, UnboundResource};
use yarvk::physical_device::memory_properties::MemoryType;
use yarvk::{
    Buffer, BufferCreateFlags, BufferUsageFlags, ContinuousBuffer, ContinuousBufferBuilder,
    DeviceSize, WHOLE_SIZE,
};

pub struct StagingVectorBuilder {
    flags: Vec<BufferCreateFlags>,
    usage: BufferUsageFlags,
    capacity: DeviceSize,
}

impl StagingVectorBuilder {
    pub fn add_flag(mut self, flag: BufferCreateFlags) -> Self {
        self.flags.push(flag);
        self
    }

    pub fn usage(mut self, usage: BufferUsageFlags) -> Self {
        self.usage = usage;
        self
    }

    pub fn capacity(mut self, capacity: DeviceSize) -> Self {
        self.capacity = capacity;
        self
    }

    pub fn build(
        self,
        device: &Arc<Device>,
        memory_type: &MemoryType,
    ) -> Result<StagingVector, yarvk::Result> {
        let mut builder = ContinuousBuffer::builder(device.clone());
        for flag in self.flags {
            builder.add_flag(flag);
        }
        builder.size(4);
        builder.usage(self.usage);
        let unbound = builder.build()?;
        let memory_req = unbound.get_memory_requirements();
        let alignment = memory_req.alignment;
        let device = unbound.device().clone();
        let mut device_memory = DeviceMemory::builder(memory_type, device)
            .allocation_size(self.capacity)
            .build()?;
        device_memory.map_memory(0, WHOLE_SIZE)?;
        Ok(StagingVector {
            device_memories: vec![device_memory],
            buffers: vec![],
            current_device_memory: 0,
            current_offset: 0,
            proto_type_builder: builder,
            alignment,
        })
    }
}

pub struct StagingVector {
    device_memories: Vec<DeviceMemory>,
    buffers: Vec<Arc<Buffer>>,
    current_device_memory: usize,
    current_offset: DeviceSize,
    proto_type_builder: ContinuousBufferBuilder,
    alignment: DeviceSize,
}

impl StagingVector {
    pub fn allocate(
        &mut self,
        size: DeviceSize,
        f: impl FnOnce(&mut [u8]),
    ) -> Result<Arc<Buffer>, yarvk::Result> {
        let alignment = self.alignment;
        let mut offset = self.current_offset + alignment - self.current_offset % alignment;
        let device_memory = &self.device_memories[self.current_device_memory];
        if offset + size > device_memory.size {
            let mut device_memory =
                DeviceMemory::builder(&device_memory.memory_type, device_memory.device.clone())
                    .allocation_size(std::cmp::max(device_memory.size, size))
                    .build()?;
            device_memory.map_memory(0, WHOLE_SIZE)?;
            self.current_device_memory = self.device_memories.len();
            self.device_memories.push(device_memory);
            self.current_offset = 0;
            offset = 0;
        }

        f(self.device_memories[self.current_device_memory]
            .get_memory(offset, offset + size)
            .unwrap());
        self.proto_type_builder.size(size);
        let unbound = self.proto_type_builder.build()?;
        let bound = Arc::new(
            unbound.bind_memory(&self.device_memories[self.current_device_memory], offset)?,
        );
        self.buffers.push(bound.clone());
        Ok(bound)
    }
    pub fn clear(&mut self) {
        self.buffers.clear();
        self.current_device_memory = 0;
        self.current_offset = 0;
    }
}
