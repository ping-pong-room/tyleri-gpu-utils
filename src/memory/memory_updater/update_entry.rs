use crate::memory::MemoryResource;

use std::sync::Arc;
use yarvk::barrier::{BufferMemoryBarrier, ImageMemoryBarrier};
use yarvk::command::command_buffer::CommandBuffer;
use yarvk::command::command_buffer::Level::SECONDARY;
use yarvk::command::command_buffer::RenderPassScope::OUTSIDE;
use yarvk::command::command_buffer::State::RECORDING;

use yarvk::image_subresource_range::{ImageSubresourceRange};
use yarvk::pipeline::pipeline_stage_flags::{PipelineStageFlag, PipelineStageFlags};
use yarvk::BufferCopy;
use yarvk::IBuffer;
use yarvk::{AccessFlags, BufferImageCopy, ImageLayout};
use yarvk::{Buffer};
use yarvk::{DependencyFlags, Image};

pub trait CmdCopyBufferTo {
    type Region;
    type TargetInfo;
    fn copy_buffer_to(
        self: &Arc<Self>,
        staging_buffer: Arc<IBuffer>,
        command_buffer: &mut CommandBuffer<{ SECONDARY }, { RECORDING }, { OUTSIDE }>,
        pipeline_stage_flags: PipelineStageFlags,
        regions: &mut Vec<(Self::TargetInfo, Self::Region)>,
    );
}

pub struct ImageTargetInfo {
    pub(crate) access_mask: AccessFlags,
    pub(crate) image_layout: ImageLayout,
}

impl CmdCopyBufferTo for MemoryResource<Image> {
    type Region = BufferImageCopy;
    type TargetInfo = ImageTargetInfo;

    fn copy_buffer_to(
        self: &Arc<Self>,
        staging_buffer: Arc<IBuffer>,
        command_buffer: &mut CommandBuffer<{ SECONDARY }, { RECORDING }, { OUTSIDE }>,
        pipeline_stage_flags: PipelineStageFlags,
        regions: &mut Vec<(Self::TargetInfo, Self::Region)>,
    ) {
        let mut begin_barriers = Vec::with_capacity(regions.len());
        let mut end_barriers = Vec::with_capacity(regions.len());
        let mut buffer_image_copies = Vec::with_capacity(regions.len());
        while let Some((target_info, region)) = regions.pop() {
            let image_subresource = region.image_subresource;
            let subresource_range = ImageSubresourceRange::builder()
                .aspect_mask(image_subresource.aspect_mask)
                .base_mip_level(image_subresource.mip_level)
                .level_count(1)
                .base_array_layer(image_subresource.base_array_layer)
                .layer_count(image_subresource.layer_count)
                .build();
            let begin_barrier = ImageMemoryBarrier::builder(self.clone())
                .dst_access_mask(AccessFlags::TRANSFER_WRITE)
                .new_layout(ImageLayout::TRANSFER_DST_OPTIMAL)
                .subresource_range(subresource_range.clone())
                .build();
            begin_barriers.push(begin_barrier);

            let end_barrier = ImageMemoryBarrier::builder(self.clone())
                .src_access_mask(AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(target_info.access_mask)
                .old_layout(ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(target_info.image_layout)
                .subresource_range(subresource_range)
                .build();
            end_barriers.push(end_barrier);
            buffer_image_copies.push(region);
        }
        command_buffer.cmd_pipeline_barrier(
            PipelineStageFlags::new(PipelineStageFlag::BottomOfPipe),
            PipelineStageFlags::new(PipelineStageFlag::Transfer),
            DependencyFlags::empty(),
            [],
            [],
            begin_barriers,
        );
        command_buffer.cmd_copy_buffer_to_image(
            staging_buffer,
            self.clone(),
            ImageLayout::TRANSFER_DST_OPTIMAL,
            buffer_image_copies.as_slice(),
        );
        command_buffer.cmd_pipeline_barrier(
            PipelineStageFlags::new(PipelineStageFlag::Transfer),
            pipeline_stage_flags,
            DependencyFlags::empty(),
            [],
            [],
            end_barriers,
        );
    }
}

pub struct BufferTargetInfo {
    pub(crate) access_mask: AccessFlags,
}

impl CmdCopyBufferTo for MemoryResource<Buffer> {
    type Region = BufferCopy;
    type TargetInfo = BufferTargetInfo;

    fn copy_buffer_to(
        self: &Arc<Self>,
        staging_buffer: Arc<IBuffer>,
        command_buffer: &mut CommandBuffer<{ SECONDARY }, { RECORDING }, { OUTSIDE }>,
        pipeline_stage_flags: PipelineStageFlags,
        regions: &mut Vec<(Self::TargetInfo, Self::Region)>,
    ) {
        let mut begin_barriers = Vec::with_capacity(regions.len());
        let mut end_barriers = Vec::with_capacity(regions.len());
        let mut buffer_copies = Vec::with_capacity(regions.len());
        while let Some((target_info, region)) = regions.pop() {
            let begin_barrier = BufferMemoryBarrier::builder(self.clone())
                .dst_access_mask(AccessFlags::TRANSFER_WRITE)
                .offset(region.dst_offset)
                .size(region.size)
                .build();
            begin_barriers.push(begin_barrier);

            let end_barrier = BufferMemoryBarrier::builder(self.clone())
                .src_access_mask(AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(target_info.access_mask)
                .offset(region.dst_offset)
                .size(region.size)
                .build();
            end_barriers.push(end_barrier);
            buffer_copies.push(region);
        }
        command_buffer.cmd_pipeline_barrier(
            PipelineStageFlags::new(PipelineStageFlag::BottomOfPipe),
            PipelineStageFlags::new(PipelineStageFlag::Transfer),
            DependencyFlags::empty(),
            [],
            begin_barriers,
            [],
        );
        command_buffer.cmd_copy_buffer(
            staging_buffer.clone(),
            self.clone(),
            buffer_copies.as_slice(),
        );
        command_buffer.cmd_pipeline_barrier(
            PipelineStageFlags::new(PipelineStageFlag::Transfer),
            pipeline_stage_flags,
            DependencyFlags::empty(),
            [],
            end_barriers,
            [],
        );
    }
}
