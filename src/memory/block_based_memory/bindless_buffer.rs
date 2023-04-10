use std::sync::Arc;

use arc_swap::ArcSwap;
use parking_lot::Mutex;
use rayon::iter::ParallelIterator;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator};
use yarvk::binding_resource::BindingResource;
use yarvk::device::Device;
use yarvk::device_memory::mapped_ranges::MappedRanges;
use yarvk::device_memory::{DeviceMemory, IMemoryRequirements, UnboundResource};
use yarvk::physical_device::memory_properties::MemoryType;
use yarvk::pipeline::pipeline_stage_flags::PipelineStageFlag;
use yarvk::{AccessFlags, DeviceSize};
use yarvk::{BoundContinuousBuffer, ContinuousBufferBuilder, IBuffer, MemoryPropertyFlags};
use yarvk::{Buffer, BufferCopy};

use crate::memory::auto_mapped_device_memory::AutoMappedDeviceMemory;
use crate::memory::block_based_memory::chunk::Chunk;
use crate::memory::memory_updater::MemoryUpdater;
use crate::memory::private::PrivateMemoryBackedResource;
use crate::memory::{IMemBakBuf, MemoryBackedResource};
use crate::queue::parallel_recording_queue::ParallelRecordingQueue;

struct DedicatedBuffer {
    buffer: BoundContinuousBuffer,
    device_memory: Arc<AutoMappedDeviceMemory>,
}

impl DedicatedBuffer {
    fn new(
        memory_type: &MemoryType,
        buffer_builder: &mut ContinuousBufferBuilder,
        size: DeviceSize,
    ) -> Result<Self, yarvk::Result> {
        buffer_builder.size(size);
        let buffer = buffer_builder.build()?;
        let device_memory = DeviceMemory::builder(&memory_type, buffer.device())
            .allocation_size(size)
            .build()?;
        let buffer = buffer.bind_memory(&device_memory, 0)?;
        let device_memory = Arc::new(AutoMappedDeviceMemory::new(device_memory));
        Ok(Self {
            buffer,
            device_memory,
        })
    }
}

impl BindingResource for DedicatedBuffer {
    type RawTy = Buffer;

    fn raw(&self) -> &Self::RawTy {
        self.buffer.raw()
    }

    fn raw_mut(&mut self) -> &mut Self::RawTy {
        self.buffer.raw_mut()
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

impl PrivateMemoryBackedResource for DedicatedBuffer {
    fn memory_property_flags(&self) -> MemoryPropertyFlags {
        self.device_memory.memory_type.property_flags
    }

    fn map_memory(
        &self,
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

impl MemoryBackedResource for DedicatedBuffer {}

#[derive(Clone)]
pub struct BindlessBuffer {
    pub offset: DeviceSize,
    pub size: DeviceSize,
    pub(crate) bindless_buffer: Arc<BindlessBufferAllocator>,
}

impl Drop for BindlessBuffer {
    fn drop(&mut self) {
        self.bindless_buffer.chunk.lock().free(self.offset);
    }
}

pub struct BindlessBufferAllocator {
    buffer_builder: ContinuousBufferBuilder,
    dedicated_buffer: ArcSwap<DedicatedBuffer>,
    chunk: Mutex<Chunk>,
}

impl BindlessBufferAllocator {
    pub fn new(
        size: DeviceSize,
        memory_type: &MemoryType,
        mut buffer_builder: ContinuousBufferBuilder,
    ) -> Result<Arc<Self>, yarvk::Result> {
        let dedicated_buffer = ArcSwap::from(Arc::new(DedicatedBuffer::new(
            memory_type,
            &mut buffer_builder,
            size,
        )?));
        let chunk = Mutex::new(Chunk::new(size));
        Ok(Arc::new(Self {
            buffer_builder,
            dedicated_buffer,
            chunk,
        }))
    }
    pub fn get_buffer(&self) -> Arc<IMemBakBuf> {
        self.dedicated_buffer.load().clone()
    }
    pub fn allocate<'a>(
        self: &Arc<Self>,
        data: &[(DeviceSize, Arc<dyn Fn(&mut [u8]) + Send + Sync>)],
        queue: &mut ParallelRecordingQueue,
    ) -> Vec<Arc<BindlessBuffer>> {
        let mut chunk = self.chunk.lock();
        let mut result_buffers = vec![
            Arc::new(BindlessBuffer {
                offset: 0,
                size: 0,
                bindless_buffer: self.clone(),
            });
            data.len()
        ];
        let mut unallocated = Vec::new();
        let mut out_of_size = 0;
        let dedicated_buffer = self.dedicated_buffer.load();
        let buffer = &dedicated_buffer.buffer;
        let memory_requirement = buffer.get_memory_requirements().clone();
        let alignment = memory_requirement.alignment;
        data.iter().enumerate().for_each(|(index, (size, _))| {
            match chunk.allocate(*size as _, alignment as _) {
                Some(offset) => {
                    let buffer = Arc::get_mut(&mut result_buffers[index]).unwrap();
                    buffer.offset = offset;
                    buffer.size = *size;
                }
                None => {
                    unallocated.push(index);
                    out_of_size += size;
                }
            }
        });
        if !unallocated.is_empty() {
            let device_memory = &dedicated_buffer.device_memory;
            let new_alloc_size = std::cmp::max(buffer.size(), out_of_size * 2);
            let new_dedicated_buffer = DedicatedBuffer::new(
                &device_memory.memory_type,
                &mut self.buffer_builder.clone(),
                new_alloc_size,
            )
            .unwrap();
            let new_dedicated_buffer = Arc::new(new_dedicated_buffer);
            chunk.expand(new_alloc_size);
            for index in unallocated {
                let size = data[index].0;
                let offset = chunk.allocate(size as _, alignment as _).unwrap();
                let buffer = Arc::get_mut(&mut result_buffers[index]).unwrap();
                buffer.offset = offset;
                buffer.size = size;
            }
            self.dedicated_buffer.swap(new_dedicated_buffer.clone());
            // can be parallel with above
            if dedicated_buffer
                .device_memory
                .memory_type
                .property_flags
                .contains(MemoryPropertyFlags::HOST_VISIBLE)
            {
                Self::copy_buffer(
                    &buffer.device,
                    &dedicated_buffer.device_memory,
                    &new_dedicated_buffer.device_memory,
                    dedicated_buffer.buffer.size(),
                )
            } else {
                Self::cmd_copy_buffer_from_offset_0(
                    dedicated_buffer.clone(),
                    new_dedicated_buffer.clone(),
                    dedicated_buffer.size(),
                    queue,
                );
                queue.simple_submit().unwrap();
            }
        }
        let updater = MemoryUpdater::default();
        result_buffers
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, bindless_buffer)| {
                let (size, f) = &data[index];
                updater.add_bindless_buffer(
                    bindless_buffer,
                    0,
                    *size,
                    AccessFlags::TRANSFER_WRITE,
                    PipelineStageFlag::Transfer.into(),
                    f.clone(),
                )
            });
        updater.update(queue);
        result_buffers
    }
    fn copy_buffer(
        device: &Arc<Device>,
        src_memory: &AutoMappedDeviceMemory,
        dst_memory: &AutoMappedDeviceMemory,
        size: DeviceSize,
    ) {
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
    fn cmd_copy_buffer_from_offset_0(
        src: Arc<IBuffer>,
        dst: Arc<IBuffer>,
        size: DeviceSize,
        queue: &mut ParallelRecordingQueue,
    ) {
        queue
            .record(move |command_buffer| {
                // let begin_barrier = BufferMemoryBarrier::builder(dst.clone())
                //     .dst_access_mask(AccessFlags::TRANSFER_WRITE)
                //     .offset(0)
                //     .size(size)
                //     .build();
                //
                //
                // let end_barrier = BufferMemoryBarrier::builder(dst.clone())
                //     .src_access_mask(AccessFlags::TRANSFER_WRITE)
                //     .dst_access_mask(AccessFlags::SHADER_READ) // whatever, doesn't matter since we don't use it before copy finished.
                //     .offset(0)
                //     .size(size)
                //     .build();
                //
                // command_buffer.cmd_pipeline_barrier(
                //     PipelineStageFlags::new(PipelineStageFlag::BottomOfPipe),
                //     PipelineStageFlags::new(PipelineStageFlag::Transfer),
                //     DependencyFlags::empty(),
                //     [],
                //     [begin_barrier],
                //     [],
                // );
                command_buffer.cmd_copy_buffer(
                    src.clone(),
                    dst.clone(),
                    &[BufferCopy {
                        src_offset: 0,
                        dst_offset: 0,
                        size,
                    }],
                );
                // command_buffer.cmd_pipeline_barrier(
                //     PipelineStageFlags::new(PipelineStageFlag::Transfer),
                //     pipeline_stage_flags,
                //     DependencyFlags::empty(),
                //     [],
                //     [end_barrier],
                //     [],
                // );
                Ok(())
            })
            .unwrap();
    }
}
