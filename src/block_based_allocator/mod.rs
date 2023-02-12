use std::collections::BTreeMap;

use crate::block_based_allocator::chunk::Chunk;
use derive_more::{Deref, DerefMut};
use std::sync::{Arc, Mutex};
use yarvk::device::Device;
use yarvk::device_memory::State::Unbound;
use yarvk::device_memory::{DeviceMemory, MemoryRequirement, UnBoundMemory};
use yarvk::physical_device::memory_properties::MemoryType;
use yarvk::{Buffer, ContinuousBuffer, ContinuousImage, Image, RawBuffer, RawImage};

mod chunk;
mod unused_blocks;

#[derive(PartialEq, Eq, Hash)]
pub struct BlockIndex {
    pub offset: u64,
    pub chunk_index: u64,
}

#[derive(Deref, DerefMut)]
pub struct BlockBasedResource<T: UnBoundMemory> {
    #[deref]
    #[deref_mut]
    yarvk_resource: T::BoundType,
    block_index: BlockIndex,
    allocator: Arc<Mutex<BlockBasedAllocator>>,
}

impl<T: UnBoundMemory> Drop for BlockBasedResource<T> {
    fn drop(&mut self) {
        let mut allocator = self.allocator.lock().unwrap();

        allocator.free(&self.block_index)
    }
}

impl Buffer for BlockBasedResource<ContinuousBuffer<{ Unbound }>> {
    fn raw(&self) -> &RawBuffer {
        self.yarvk_resource.raw()
    }

    fn raw_mut(&mut self) -> &mut RawBuffer {
        self.yarvk_resource.raw_mut()
    }
}

impl Image for BlockBasedResource<ContinuousImage<{ Unbound }>> {
    fn raw(&self) -> &RawImage {
        self.yarvk_resource.raw()
    }

    fn raw_mut(&mut self) -> &mut RawImage {
        self.yarvk_resource.raw_mut()
    }
}

#[derive(Deref, DerefMut)]
pub struct VkChunk {
    #[deref]
    #[deref_mut]
    chunk: Chunk,
    device_memory: DeviceMemory,
}

pub struct BlockBasedAllocator {
    device: Arc<Device>,
    memory_type: MemoryType,
    chunks: BTreeMap<u64 /*size*/, VkChunk>,
    total_size: u64,
}

impl BlockBasedAllocator {
    pub fn new(device: Arc<Device>, memory_type: MemoryType) -> BlockBasedAllocator {
        BlockBasedAllocator {
            device,
            memory_type,
            chunks: BTreeMap::default(),
            total_size: 0,
        }
    }
    /// Ask vulkan device to allocate a device memory which is large enough to hold `len` bytes.
    pub fn capacity(&mut self, len: u64) -> Result<&Self, yarvk::Result> {
        // we allocate the space twice then asked, to make sure the memory is big enough to hold
        // resource with any alignment
        let len = len * 2;
        let device_memory = DeviceMemory::builder(&self.memory_type, self.device.clone())
            .allocation_size(len)
            .build()?;
        self.chunks.insert(
            len,
            VkChunk {
                chunk: Chunk::new(len),
                device_memory,
            },
        );
        self.total_size += len;
        Ok(self)
    }
    pub fn capacity_with_allocate(
        &mut self,
        len: u64,
        allocate_len: u64,
    ) -> Result<BlockIndex, yarvk::Result> {
        // we allocate the space twice then asked, to make sure the memory is big enough to hold
        // resource with any alignment
        let len = len * 2;
        let device_memory = DeviceMemory::builder(&self.memory_type, self.device.clone())
            .allocation_size(len)
            .build()?;
        self.chunks.insert(
            len,
            VkChunk {
                chunk: Chunk::new_and_allocated(len, allocate_len),
                device_memory,
            },
        );
        let block_index = BlockIndex {
            offset: 0,
            chunk_index: len,
        };
        self.total_size += len;
        Ok(block_index)
    }
    pub fn allocate<T: UnBoundMemory + MemoryRequirement>(
        allocator: &Arc<Mutex<Self>>,
        t: T,
    ) -> Result<BlockBasedResource<T>, yarvk::Result> {
        let mut this = allocator.lock().unwrap();
        let memory_requirements = t.get_memory_requirements();
        let mut block_index = None;
        for (len, chunk) in this.chunks.iter_mut().rev() {
            match chunk.allocate(memory_requirements.size, memory_requirements.alignment) {
                None => {
                    continue;
                }
                Some(offset) => {
                    block_index = Some(BlockIndex {
                        offset,
                        chunk_index: *len,
                    });
                    break;
                }
            }
        }
        let block_index = match block_index {
            None => {
                let total_size = this.total_size;
                this.capacity_with_allocate(total_size, memory_requirements.size)?
            }
            Some(block_index) => block_index,
        };

        let chunk = this.chunks.get(&block_index.chunk_index).unwrap();
        let res = t
            .bind_memory(&chunk.device_memory, block_index.offset)
            .expect("internal error: bind_memory failed");
        Ok(BlockBasedResource {
            yarvk_resource: res,
            block_index,
            allocator: allocator.clone(),
        })
    }

    fn free(&mut self, block_index: &BlockIndex) {
        if let Some(chunk) = self.chunks.get_mut(&block_index.chunk_index) {
            chunk.free(block_index.offset);
            if chunk.is_unused() {
                self.chunks.remove(&block_index.chunk_index);
            }
        }
    }
}
