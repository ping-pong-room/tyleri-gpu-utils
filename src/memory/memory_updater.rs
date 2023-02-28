mod staging_buffer;
mod update_entry;

use crate::memory::auto_mapped_device_memory::AutoMappedDeviceMemory;
use crate::memory::memory_updater::staging_buffer::StagingBuffer;
use crate::memory::memory_updater::update_entry::ImageTargetInfo;
use crate::memory::memory_updater::update_entry::{BufferTargetInfo, CmdCopyBufferTo};
use crate::memory::{IBufferResource, IImageResource};
use crate::queue::parallel_recording_queue::ParallelRecordingQueue;
use dashmap::DashMap;
use rayon::iter::IntoParallelIterator;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use yarvk::device_memory::mapped_ranges::MappedRanges;

use yarvk::pipeline::pipeline_stage_flags::PipelineStageFlags;
use yarvk::BufferCopy;
use yarvk::DeviceSize;
use yarvk::Offset3D;
use yarvk::{AccessFlags, ImageLayout, ImageSubresourceLayers};
use yarvk::{BufferImageCopy, MemoryPropertyFlags};
use yarvk::{Extent3D, Handle};

type WriteFn = dyn Fn(&mut [u8]) + Send + Sync;
struct PendingImage {
    image: Arc<IImageResource>,
    pipeline_stage_flags: PipelineStageFlags,
    regions: Vec<SubresourceInfo>,
}

struct SubresourceInfo {
    staging_size: u64,
    image_subresource: ImageSubresourceLayers,
    image_offset: Offset3D,
    image_extent: Extent3D,
    f: Arc<WriteFn>,
    access_mask: AccessFlags,
    image_layout: ImageLayout,
}

pub struct PendingBuffer {
    buffer: Arc<IBufferResource>,
    pipeline_stage_flags: PipelineStageFlags,
    sub_info: Vec<SubBufferInfo>,
}

pub struct SubBufferInfo {
    offset: u64,
    size: u64,
    access_mask: AccessFlags,
    f: Arc<WriteFn>,
}

#[derive(Default)]
pub struct MemoryUpdater {
    staging_size: AtomicU64,
    pending_images: DashMap<u64 /*image handle*/, PendingImage>,
    pending_buffers: DashMap<u64 /*image handle*/, PendingBuffer>,
    no_coherent_regions: DashMap<Arc<AutoMappedDeviceMemory>, Vec<(DeviceSize, DeviceSize)>>,
}

impl MemoryUpdater {
    pub fn add_image(
        &self,
        image: Arc<IImageResource>,
        format_size_in_bytes: u64,
        image_subresource: ImageSubresourceLayers,
        image_offset: Offset3D,
        image_extent: Extent3D,
        access_mask: AccessFlags,
        image_layout: ImageLayout,
        pipeline_stage_flags: PipelineStageFlags,
        f: Arc<WriteFn>,
    ) {
        // TODO VK_IMAGE_TILING_LINEAR image
        let staging_size = std::cmp::max(image_extent.width as i32 - image_offset.x, 0) as u64
            * std::cmp::max(image_extent.height as i32 - image_offset.y, 0) as u64
            * std::cmp::max(image_extent.depth as i32 - image_offset.z, 0) as u64
            * format_size_in_bytes;
        self.staging_size.fetch_add(staging_size, Ordering::Relaxed);
        self.pending_images
            .entry(image.raw().handle())
            .or_insert(PendingImage {
                image,
                pipeline_stage_flags,
                regions: vec![],
            })
            .regions
            .push(SubresourceInfo {
                staging_size,
                image_subresource,
                image_offset,
                image_extent,
                f,
                access_mask,
                image_layout,
            });
    }

    pub fn add_buffer(
        &self,
        buffer: &mut Arc<IBufferResource>,
        offset: DeviceSize,
        size: DeviceSize,
        access_mask: AccessFlags,
        pipeline_stage_flags: PipelineStageFlags,
        f: Arc<WriteFn>,
    ) {
        if !buffer
            .memory_property_flags()
            .contains(MemoryPropertyFlags::HOST_VISIBLE)
        {
            self.staging_size.fetch_add(size, Ordering::Relaxed);
            self.pending_buffers
                .entry(buffer.raw().handle())
                .or_insert(PendingBuffer {
                    buffer: buffer.clone(),
                    pipeline_stage_flags,
                    sub_info: vec![],
                })
                .sub_info
                .push(SubBufferInfo {
                    offset,
                    size,
                    access_mask,
                    f,
                });
            return;
        }
        if buffer
            .memory_property_flags()
            .contains(MemoryPropertyFlags::HOST_VISIBLE)
        {
            let mut_buffer = Arc::get_mut(buffer).expect("buffer is using");
            mut_buffer.memory_memory(offset, size, f.as_ref()).unwrap();
        }
        if !buffer
            .memory_property_flags()
            .contains(MemoryPropertyFlags::HOST_COHERENT)
        {
            let device_memory = buffer.get_device_memory();
            self.no_coherent_regions
                .entry(device_memory)
                .or_default()
                .push((buffer.offset() + offset, buffer.size()));
        }
    }
    fn update_pending_images(
        staging_buffer: &Arc<StagingBuffer>,
        pending_images: DashMap<u64 /*image handle*/, PendingImage>,
        queue: &ParallelRecordingQueue,
    ) {
        if !pending_images.is_empty() {
            pending_images
                .into_par_iter()
                .for_each(|(_, pending_image_wrapper)| {
                    queue
                        .record(|command_buffer| {
                            let mut regions = pending_image_wrapper
                                .regions
                                .iter()
                                .map(|pending_image| {
                                    let offset = staging_buffer
                                        .write_and_get_offset(pending_image.staging_size, |slice| {
                                            (pending_image.f)(slice)
                                        })
                                        .unwrap();
                                    (
                                        ImageTargetInfo {
                                            access_mask: pending_image.access_mask,
                                            image_layout: pending_image.image_layout,
                                        },
                                        BufferImageCopy {
                                            buffer_offset: offset,
                                            buffer_row_length: 0,
                                            buffer_image_height: 0,
                                            image_subresource: pending_image.image_subresource,
                                            image_offset: pending_image.image_offset,
                                            image_extent: pending_image.image_extent,
                                        },
                                    )
                                })
                                .collect();
                            pending_image_wrapper.image.copy_buffer_to(
                                staging_buffer.clone(),
                                command_buffer,
                                pending_image_wrapper.pipeline_stage_flags,
                                &mut regions,
                            );
                            Ok(())
                        })
                        .unwrap();
                });
        }
    }
    pub fn update_pending_buffers(
        staging_buffer: &Arc<StagingBuffer>,
        pending_buffers: DashMap<u64 /*image handle*/, PendingBuffer>,
        queue: &ParallelRecordingQueue,
    ) {
        if !pending_buffers.is_empty() {
            pending_buffers
                .into_par_iter()
                .for_each(|(_, pending_buffer)| {
                    queue
                        .record(|command_buffer| {
                            let mut regions = pending_buffer
                                .sub_info
                                .iter()
                                .map(|pending_buffer| {
                                    let offset = staging_buffer
                                        .write_and_get_offset(pending_buffer.size, |slice| {
                                            (pending_buffer.f)(slice)
                                        })
                                        .unwrap();
                                    (
                                        BufferTargetInfo {
                                            access_mask: pending_buffer.access_mask,
                                        },
                                        BufferCopy {
                                            src_offset: offset,
                                            dst_offset: pending_buffer.offset,
                                            size: pending_buffer.size,
                                        },
                                    )
                                })
                                .collect();
                            pending_buffer.buffer.copy_buffer_to(
                                staging_buffer.clone(),
                                command_buffer,
                                pending_buffer.pipeline_stage_flags,
                                &mut regions,
                            );
                            Ok(())
                        })
                        .unwrap();
                });
        }
    }
    pub fn update(self, queue: &mut ParallelRecordingQueue) {
        let device = &queue.device;
        let staging_size = self.staging_size.load(Ordering::Relaxed);
        if staging_size == 0 && self.no_coherent_regions.is_empty() {
            return;
        }
        let mut mapped_ranges = MappedRanges::new(device);
        let mut staging_buffer = None;
        if staging_size != 0 {
            staging_buffer = Some(Arc::new(
                StagingBuffer::new(device, staging_size).expect("init staging buffer failed"),
            ))
        }
        if let Some(staging_buffer) = &staging_buffer {
            Self::update_pending_images(staging_buffer, self.pending_images, queue);
            Self::update_pending_buffers(staging_buffer, self.pending_buffers, queue);
            let device_memory = &staging_buffer.device_memory;
            if !device_memory
                .memory_type
                .property_flags
                .contains(MemoryPropertyFlags::HOST_COHERENT)
            {
                mapped_ranges.add_range(device_memory, 0, device_memory.size);
            }
        }
        let regions = self
            .no_coherent_regions
            .into_iter()
            .map(|(arc_device_memory, regions)| (arc_device_memory, regions))
            .collect::<Vec<_>>();
        let locks = regions
            .iter()
            .map(|(device_memory, region)| (device_memory.get_device_memory(), region.as_slice()))
            .collect::<Vec<_>>();
        locks.iter().for_each(|(device_memory, regions)| {
            for (offset, size) in *regions {
                mapped_ranges.add_range(device_memory, *offset, *size);
            }
        });
        mapped_ranges.flush().unwrap();
        queue.simple_submit().unwrap();
    }
}
