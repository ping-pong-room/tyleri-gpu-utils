use std::sync::Arc;
use yarvk::device::Device;
use yarvk::device_memory::mapped_memory::MappedMemory;
use yarvk::device_memory::{DeviceMemory, MemoryRequirement, UnBoundMemory};
use yarvk::physical_device::memory_properties::{MemoryType, PhysicalDeviceMemoryProperties};
use yarvk::MemoryRequirements;
use yarvk::{
    Buffer, BufferCreateFlags, BufferUsageFlags, ContinuousBuffer, ContinuousBufferBuilder,
    DeviceSize, MemoryPropertyFlags, WHOLE_SIZE,
};

fn find_memory_type_index(
    memory_req: &MemoryRequirements,
    memory_prop: &PhysicalDeviceMemoryProperties,
    flags: MemoryPropertyFlags,
) -> Option<MemoryType> {
    memory_prop
        .memory_types
        .iter()
        .enumerate()
        .find(|(index, memory_type)| {
            (1 << index) & memory_req.memory_type_bits != 0
                && memory_type.property_flags & flags == flags
        })
        .map(|(_index, memory_type)| memory_type.clone())
}

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

    pub fn build(self, device: &Arc<Device>) -> Result<StagingVector, yarvk::Result> {
        let mut builder = ContinuousBuffer::builder(device.clone());
        for flag in self.flags {
            builder = builder.add_flag(flag);
        }
        let builder = builder.size(4).usage(self.usage);
        let unbound = builder.clone().build()?;
        let memory_req = unbound.get_memory_requirements();
        let alignment = memory_req.alignment;
        let device = unbound.device().clone();
        let device_memory_properties = unbound.device().physical_device.memory_properties();
        let memory_index = find_memory_type_index(
            &memory_req,
            &device_memory_properties,
            MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT,
        )
        .ok_or(yarvk::Result::ERROR_INITIALIZATION_FAILED)?;
        let device_memory = DeviceMemory::builder(&memory_index, device)
            .allocation_size(self.capacity)
            .build()?;
        let mapped_memory = device_memory.map_memory(0, WHOLE_SIZE)?;
        Ok(StagingVector {
            device_memories: vec![mapped_memory],
            buffers: vec![],
            current_device_memory: 0,
            current_offset: 0,
            proto_type_builder: builder,
            alignment,
        })
    }
}

pub struct StagingVector {
    device_memories: Vec<MappedMemory>,
    buffers: Vec<Arc<dyn Buffer>>,
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
    ) -> Result<Arc<dyn Buffer>, yarvk::Result> {
        let alignment = self.alignment;
        let mut offset = self.current_offset + alignment - self.current_offset % alignment;
        let device_memory = &self.device_memories[self.current_device_memory].device_memory;
        if offset + size > device_memory.size {
            let device_memory =
                DeviceMemory::builder(&device_memory.memory_type, device_memory.device.clone())
                    .allocation_size(std::cmp::max(device_memory.size, size))
                    .build()?;
            let mapped_memory = device_memory.map_memory(0, WHOLE_SIZE)?;
            self.current_device_memory = self.device_memories.len();
            self.device_memories.push(mapped_memory);
            self.current_offset = 0;
            offset = 0;
        }

        f(
            &mut self.device_memories[self.current_device_memory]
                [offset as _..(offset + size) as _],
        );
        let unbound = self.proto_type_builder.clone().size(size).build()?;
        let bound = Arc::new(unbound.bind_memory(
            &self.device_memories[self.current_device_memory].device_memory,
            offset,
        )?);
        self.buffers.push(bound.clone());
        Ok(bound)
    }
    pub fn clear(&mut self) {
        self.buffers.clear();
        self.current_device_memory = 0;
        self.current_offset = 0;
    }
}
