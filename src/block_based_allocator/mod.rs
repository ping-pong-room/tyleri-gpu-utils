use crate::block_based_allocator::chunk::Chunk;
use crate::memory_object::{MemoryBackedResource, MemoryResource};
use derive_more::{Deref, DerefMut};
use std::collections::BTreeMap;
use std::sync::{Arc, RwLock};
use yarvk::binding_resource::BindingResource;
use yarvk::device::Device;

use yarvk::device_memory::{DeviceMemory, IMemoryRequirements, UnboundResource};
use yarvk::physical_device::memory_properties::MemoryType;
use yarvk::WHOLE_SIZE;

mod chunk;
mod unused_blocks;

struct BlockBasedResource<T: UnboundResource> {
    resource: T::BoundType,
    block_index: BlockIndex,
    allocator: Arc<BlockBasedAllocator>,
}

impl<T: UnboundResource> BlockBasedResource<T> {
    fn try_get_memory(&mut self, f: &dyn Fn(&mut [u8])) -> Result<(), yarvk::Result> {
        let chunks = self.allocator.chunks.read().unwrap();
        let memory = chunks
            .get(&self.block_index.chunk_index)
            .unwrap()
            .device_memory
            .get_memory(
                self.block_index.offset,
                self.resource.get_memory_requirements().size,
            )?;
        f(memory);
        Ok(())
    }
}

impl<T: UnboundResource> BindingResource for BlockBasedResource<T> {
    type RawTy = T::RawTy;

    fn raw(&self) -> &Self::RawTy {
        self.resource.raw()
    }

    fn raw_mut(&mut self) -> &mut Self::RawTy {
        self.resource.raw_mut()
    }
}

impl<T: UnboundResource> MemoryBackedResource for BlockBasedResource<T> {
    fn memory_memory(&mut self, f: &dyn Fn(&mut [u8])) -> Result<(), yarvk::Result> {
        match self.try_get_memory(f) {
            Ok(_) => Ok(()),
            Err(_) => {
                {
                    let mut chunks = self.allocator.chunks.write().unwrap();
                    let device_memory = &mut chunks
                        .get_mut(&self.block_index.chunk_index)
                        .unwrap()
                        .device_memory;
                    if device_memory.map_memory(0, WHOLE_SIZE).is_err() {
                        device_memory.map_memory(
                            self.block_index.offset,
                            self.resource.get_memory_requirements().size,
                        )?
                    }
                }
                self.try_get_memory(f)
            }
        }
    }
}

impl<T: UnboundResource> Drop for BlockBasedResource<T> {
    fn drop(&mut self) {
        self.allocator.free(&self.block_index)
    }
}

#[derive(PartialEq, Eq, Hash)]
struct BlockIndex {
    pub offset: u64,
    pub chunk_index: u64,
}

#[derive(Deref, DerefMut)]
struct VkChunk {
    #[deref]
    #[deref_mut]
    chunk: Chunk,
    device_memory: DeviceMemory,
}

pub struct BlockBasedAllocator {
    device: Arc<Device>,
    memory_type: MemoryType,
    chunks: RwLock<BTreeMap<u64 /*size*/, VkChunk>>,
}

impl BlockBasedAllocator {
    pub fn capacity(&self, len: u64) -> Result<(), yarvk::Result> {
        // we allocate the space twice then asked, to make sure the memory is big enough to hold
        // resource with any alignment
        let len = len * 2;
        let device_memory = DeviceMemory::builder(&self.memory_type, self.device.clone())
            .allocation_size(len)
            .build()?;
        self.chunks.write().unwrap().insert(
            len,
            VkChunk {
                chunk: Chunk::new(len),
                device_memory,
            },
        );
        Ok(())
    }
    fn capacity_with_allocate(
        device: &Arc<Device>,
        memory_type: &MemoryType,
        chunks: &mut BTreeMap<u64 /*size*/, VkChunk>,
        len: u64,
        allocate_len: u64,
    ) -> Result<BlockIndex, yarvk::Result> {
        // we allocate the space twice then asked, to make sure the memory is big enough to hold
        // resource with any alignment
        let len = len * 2;
        let device_memory = DeviceMemory::builder(memory_type, device.clone())
            .allocation_size(len)
            .build()?;
        chunks.insert(
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
        Ok(block_index)
    }
    pub fn allocate<T: UnboundResource + IMemoryRequirements + 'static>(
        self: &Arc<Self>,
        t: T,
    ) -> Result<Arc<MemoryResource<T::RawTy>>, yarvk::Result> {
        let memory_requirements = t.get_memory_requirements();
        let mut block_index = None;
        let block_index = {
            let mut chunks = self.chunks.write().unwrap();
            for (len, chunk) in chunks.iter_mut().rev() {
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
            match block_index {
                None => {
                    let max_chunk_size = chunks.iter().rev().next().unwrap().1.device_memory.size;
                    Self::capacity_with_allocate(
                        &self.device,
                        &self.memory_type,
                        &mut chunks,
                        max_chunk_size * 2,
                        memory_requirements.size,
                    )?
                }
                Some(block_index) => block_index,
            }
        };
        let chunks = self.chunks.read().unwrap();
        let chunk = chunks.get(&block_index.chunk_index).unwrap();
        let resource = t
            .bind_memory(&chunk.device_memory, block_index.offset)
            .expect("internal error: bind_memory failed");
        Ok(Arc::new(BlockBasedResource::<T> {
            resource,
            block_index,
            allocator: self.clone(),
        }))
    }
    fn free(&self, block_index: &BlockIndex) {
        let mut chunks = self.chunks.write().unwrap();
        if let Some(chunk) = chunks.get_mut(&block_index.chunk_index) {
            chunk.free(block_index.offset);
            if chunk.is_unused() {
                chunks.remove(&block_index.chunk_index);
            }
        }
    }
}
